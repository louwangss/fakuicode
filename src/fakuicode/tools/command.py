"""Controlled local command execution for the workspace agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import signal
import subprocess
import tempfile
import io
from threading import Event
from time import monotonic
from typing import BinaryIO

from fakuicode.errors import (
    RequestCancelled,
    ToolExecutionError,
    ToolOutputStorageError,
    ToolPolicyError,
)
from fakuicode.context import ContextPolicy, build_stored_tool_result_preview
from fakuicode.context_artifacts import ContextArtifactStore, ContextArtifactRef
from fakuicode.models import ToolDefinition
from fakuicode.permissions.safety import serialize_command
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments
from fakuicode.tools.policy import WorkspacePolicy


DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0


class RunCommandTool:
    def __init__(
        self,
        policy: WorkspacePolicy,
        *,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        artifact_store: ContextArtifactStore | None = None,
    ) -> None:
        self.policy = policy
        self.timeout_seconds = timeout_seconds
        self.artifact_store = artifact_store

    def set_artifact_store(self, artifact_store: ContextArtifactStore | None) -> None:
        self.artifact_store = artifact_store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "run_command",
            "在工作区内以 shell=False 直接运行策略允许的 argv 命令。仅当专用工具不适用时使用；"
            "文件发现优先使用 find_files。不得调用 cmd /c、PowerShell -Command、sh -c 或 bash -c "
            "等通用 shell 包装器，它们会被安全边界拒绝。",
            {
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "array", "minItems": 1, "items": {"type": "string"}}},
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"command"}:
            if "command" not in arguments:
                raise ToolExecutionError("Tool 'run_command' requires a string command list.")
            raise ToolExecutionError("Tool 'run_command' received unexpected arguments.")
        value = arguments.get("command")
        if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
            raise ToolExecutionError("Tool 'run_command' requires a string command list.")
        validated = self.policy.validate_command(value)
        return ToolPreparation(freeze_arguments({"command": validated}), serialize_command(validated))

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        value = arguments.get("command")
        if not isinstance(value, tuple) or not all(isinstance(part, str) for part in value):
            raise ToolExecutionError("Prepared run_command arguments are invalid.")
        return self._run_validated(value, cancel_event=cancel_event)

    def run(self, command: Sequence[str], *, cancel_event: Event | None = None) -> ToolExecution:
        validated = self.policy.validate_command(command)
        return self._run_validated(validated, cancel_event=cancel_event)

    def _run_validated(self, validated: tuple[str, ...], *, cancel_event: Event | None = None) -> ToolExecution:
        if validated[0].casefold() == "ls":
            return self._list_directory(validated[1:])
        process_group: dict[str, object]
        if os.name == "nt":
            process_group = {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        else:
            process_group = {"start_new_session": True}
        with tempfile.TemporaryFile(mode="w+b") as stdout_stream, tempfile.TemporaryFile(
            mode="w+b"
        ) as stderr_stream:
            try:
                process = subprocess.Popen(
                    validated,
                    cwd=self.policy.workspace,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    shell=False,
                    **process_group,
                )
            except OSError as error:
                raise ToolExecutionError("Unable to start the requested local command.") from error

            windows_job = _create_windows_kill_job(process)
            try:
                deadline = monotonic() + self.timeout_seconds
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        _terminate_process_tree(process, windows_job)
                        process.wait()
                        raise RequestCancelled()
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        _terminate_process_tree(process, windows_job)
                        process.wait()
                        raise ToolExecutionError("Command timed out.")
                    try:
                        process.wait(timeout=min(0.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                _close_windows_job(windows_job)

            success = process.returncode == 0
            return self._execution_from_streams(
                stdout_stream,
                stderr_stream,
                exit_code=process.returncode,
                success=success,
                summary=f"command exited with exit {process.returncode}",
            )

    def _execution_from_streams(
        self,
        stdout_stream: BinaryIO,
        stderr_stream: BinaryIO,
        *,
        exit_code: int,
        success: bool,
        summary: str,
    ) -> ToolExecution:
        if self.artifact_store is None:
            stdout = _read_command_stream(stdout_stream)
            stderr = _read_command_stream(stderr_stream)
            output = f"stdout:\n{stdout}\nstderr:\n{stderr}\nexit_code: {exit_code}"
            metadata = None
        else:
            try:
                reference = self.artifact_store.write_command_result_streams(
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    exit_code=exit_code,
                    success=success,
                )
                output = _artifact_visible_output(self.artifact_store, reference)
            except (OSError, ValueError) as error:
                raise ToolOutputStorageError(
                    "Command output could not be stored safely."
                ) from error
            metadata = {"context_artifact": _artifact_metadata(reference)}
        return ToolExecution(success, output, summary, metadata=metadata)

    def _list_directory(self, arguments: Sequence[str]) -> ToolExecution:
        show_hidden, long_format, target = _parse_ls_arguments(arguments)
        path = self.policy.resolve_path(target)
        if not path.is_dir():
            raise ToolExecutionError("The requested ls target is not a directory.")
        try:
            entries = sorted(path.iterdir(), key=lambda entry: entry.name.casefold())
        except OSError as error:
            raise ToolExecutionError("Unable to list the requested workspace directory.") from error

        with tempfile.TemporaryFile(mode="w+b") as stdout_stream, tempfile.TemporaryFile(
            mode="w+b"
        ) as stderr_stream:
            wrote_entry = False
            for entry in entries:
                if not show_hidden and entry.name.startswith("."):
                    continue
                try:
                    safe_entry = self.policy.resolve_path(str(entry))
                except ToolPolicyError:
                    continue
                if long_format:
                    entry_type = "d" if safe_entry.is_dir() else "-"
                    size = "-" if safe_entry.is_dir() else str(safe_entry.stat().st_size)
                    line = f"{entry_type} {size:>8} {safe_entry.name}"
                else:
                    line = safe_entry.name
                if wrote_entry:
                    stdout_stream.write(b"\n")
                stdout_stream.write(line.encode("utf-8", errors="replace"))
                wrote_entry = True
            return self._execution_from_streams(
                stdout_stream,
                stderr_stream,
                exit_code=0,
                success=True,
                summary=f"listed {self.policy.relative_target(path)}",
            )


CommandTools = RunCommandTool


def _read_command_stream(stream) -> str:
    stream.seek(0)
    text_stream = io.TextIOWrapper(
        stream,
        encoding="utf-8",
        errors="replace",
        newline=None,
    )
    try:
        return text_stream.read()
    finally:
        text_stream.detach()


def _artifact_visible_output(
    store: ContextArtifactStore,
    reference: ContextArtifactRef,
) -> str:
    policy = ContextPolicy()
    if reference.estimated_tokens <= policy.single_tool_result_tokens:
        path = store.resolve_read_path(reference)
        return path.read_text(encoding="utf-8")
    edge_bytes = policy.tool_preview_max_tokens * 4
    head, tail = store.read_text_edges(reference, edge_bytes=edge_bytes)
    return build_stored_tool_result_preview(
        head=head,
        tail=tail,
        original_bytes=reference.byte_size,
        original_tokens=reference.estimated_tokens,
        success=reference.success,
        read_path=reference.read_path,
        budget_tokens=policy.tool_preview_max_tokens,
    )


def _artifact_metadata(reference: ContextArtifactRef) -> dict[str, object]:
    return {
        "conversation_id": reference.conversation_id,
        "read_path": reference.read_path,
        "content_sha256": reference.content_sha256,
        "byte_size": reference.byte_size,
        "estimated_tokens": reference.estimated_tokens,
        "success": reference.success,
    }


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    windows_job: int | None,
) -> None:
    """Best-effort termination of the isolated process tree for one command."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        if windows_job is not None:
            import ctypes

            if ctypes.windll.kernel32.TerminateJobObject(windows_job, 1):
                return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
            )
        except OSError:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def _create_windows_kill_job(process: subprocess.Popen[bytes]) -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    process_handle = kernel32.OpenProcess(0x0001 | 0x0100, False, process.pid)
    if not process_handle:
        kernel32.CloseHandle(job)
        return None
    try:
        assigned = configured and kernel32.AssignProcessToJobObject(job, process_handle)
    finally:
        kernel32.CloseHandle(process_handle)
    if not assigned:
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _close_windows_job(job: int | None) -> None:
    if job is None or os.name != "nt":
        return
    import ctypes

    ctypes.windll.kernel32.CloseHandle(job)


def _parse_ls_arguments(arguments: Sequence[str]) -> tuple[bool, bool, str]:
    show_hidden = False
    long_format = False
    target: str | None = None
    for argument in arguments:
        if argument.startswith("-"):
            if argument in {"--all", "--almost-all"}:
                show_hidden = True
            elif argument == "--long":
                long_format = True
            elif len(argument) > 1 and set(argument[1:]) <= {"a", "l"}:
                show_hidden = show_hidden or "a" in argument
                long_format = long_format or "l" in argument
            else:
                raise ToolExecutionError("Portable ls supports only -a/--all and -l/--long options.")
        elif target is None:
            target = argument
        else:
            raise ToolExecutionError("Portable ls accepts at most one workspace path.")
    return show_hidden, long_format, target or "."
