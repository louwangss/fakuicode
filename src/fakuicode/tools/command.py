"""Controlled local command execution for the workspace agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
from threading import Event
from time import monotonic

from fakuicode.errors import RequestCancelled, ToolExecutionError, ToolPolicyError
from fakuicode.models import ToolDefinition
from fakuicode.permissions.safety import serialize_command
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments
from fakuicode.tools.policy import WorkspacePolicy


DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0


class RunCommandTool:
    def __init__(self, policy: WorkspacePolicy, *, timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS) -> None:
        self.policy = policy
        self.timeout_seconds = timeout_seconds

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
        try:
            process = subprocess.Popen(
                validated,
                cwd=self.policy.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except OSError as error:
            raise ToolExecutionError("Unable to start the requested local command.") from error

        deadline = monotonic() + self.timeout_seconds
        while True:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                process.communicate()
                raise RequestCancelled()
            remaining = deadline - monotonic()
            if remaining <= 0:
                process.terminate()
                process.communicate()
                raise ToolExecutionError("Command timed out.")
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        output = f"stdout:\n{stdout}\nstderr:\n{stderr}\nexit_code: {process.returncode}"
        return ToolExecution(process.returncode == 0, output, f"command exited with exit {process.returncode}")

    def _list_directory(self, arguments: Sequence[str]) -> ToolExecution:
        show_hidden, long_format, target = _parse_ls_arguments(arguments)
        path = self.policy.resolve_path(target)
        if not path.is_dir():
            raise ToolExecutionError("The requested ls target is not a directory.")
        try:
            entries = sorted(path.iterdir(), key=lambda entry: entry.name.casefold())
        except OSError as error:
            raise ToolExecutionError("Unable to list the requested workspace directory.") from error

        lines: list[str] = []
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
                lines.append(f"{entry_type} {size:>8} {safe_entry.name}")
            else:
                lines.append(safe_entry.name)
        listing = "\n".join(lines)
        output = f"stdout:\n{listing}\nstderr:\n\nexit_code: 0"
        return ToolExecution(True, output, f"listed {self.policy.relative_target(path)}")


CommandTools = RunCommandTool


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
