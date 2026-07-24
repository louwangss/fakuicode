"""Workspace-scoped file, discovery, and code-search tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import difflib
from pathlib import Path
from threading import Event
from time import monotonic

from fakuicode.errors import ToolExecutionError, ToolPolicyError
from fakuicode.models import ToolDefinition
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments
from fakuicode.tools.policy import WorkspacePolicy


_MAX_DIFF = 4_000
_MAX_MATCHES = 200
_SCAN_TIMEOUT_SECONDS = 5.0
_IGNORED_DISCOVERY_DIRECTORIES = {"__pycache__", ".pytest_cache"}
_FILE_RESULTS_NOTICE = "\n… file results truncated; use a narrower pattern or path"
_SEARCH_RESULTS_NOTICE = "\n… search results truncated; use a narrower query or path"


class ReadFileTool:
    def __init__(self, policy: WorkspacePolicy) -> None:
        self.policy = policy

    @property
    def definition(self) -> ToolDefinition:
        return _definition("read_file", "读取工作区内一个 UTF-8 文件的内容。修改已有文件前应先使用本工具读取相关内容。", ("path",), {"path": _path_schema()})

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        _validate_arguments(arguments, required={"path"})
        path = self.policy.resolve_path(
            _string(arguments, "path"),
            allow_context_artifact_read=True,
        )
        return ToolPreparation(freeze_arguments({"path": path}), self.policy.relative_target(path))

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        del cancel_event
        path = _prepared_path(
            arguments,
            "path",
            self.policy,
            allow_context_artifact_read=True,
        )
        content = _read_text(path)
        numbered = "".join(f"{number}: {line}" for number, line in enumerate(content.splitlines(keepends=True), start=1))
        if content and not numbered:
            numbered = "1: "
        return ToolExecution(True, numbered, f"read {self.policy.relative_target(path)}")


class WriteFileTool:
    def __init__(self, policy: WorkspacePolicy) -> None:
        self.policy = policy

    @property
    def definition(self) -> ToolDefinition:
        return _definition(
            "write_file",
            "创建一个 UTF-8 工作区文件并自动创建父目录，或有意完整替换其内容。"
            "局部修改已有文件时优先使用 edit_file。",
            ("path", "content"),
            {"path": _path_schema(), "content": {"type": "string"}},
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        _validate_arguments(arguments, required={"path", "content"})
        path = self.policy.resolve_path(_string(arguments, "path"))
        prepared = freeze_arguments({"path": path, "content": _string(arguments, "content")})
        return ToolPreparation(prepared, self.policy.relative_target(path))

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        del cancel_event
        path = _prepared_path(arguments, "path", self.policy)
        content = _string(arguments, "content")
        previous = _read_text(path) if path.exists() else ""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as error:
            raise ToolExecutionError("Unable to write the requested workspace file.") from error
        return ToolExecution(True, "", _diff_summary(path.name, previous, content))


class EditFileTool:
    def __init__(self, policy: WorkspacePolicy) -> None:
        self.policy = policy

    @property
    def definition(self) -> ToolDefinition:
        return _definition(
            "edit_file",
            "精确替换工作区文件中唯一匹配的一段文本。调用前必须先读取目标文件并确认旧文本。",
            ("path", "old_text", "new_text"),
            {"path": _path_schema(), "old_text": {"type": "string"}, "new_text": {"type": "string"}},
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        _validate_arguments(arguments, required={"path", "old_text", "new_text"})
        old_text = _string(arguments, "old_text")
        if not old_text:
            raise ToolExecutionError("The expected text must not be empty.")
        path = self.policy.resolve_path(_string(arguments, "path"))
        prepared = freeze_arguments(
            {
                "path": path,
                "old_text": old_text,
                "new_text": _string(arguments, "new_text"),
            }
        )
        return ToolPreparation(prepared, self.policy.relative_target(path))

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        del cancel_event
        old_text = _string(arguments, "old_text")
        path = _prepared_path(arguments, "path", self.policy)
        previous = _read_text(path)
        matches = previous.count(old_text)
        if matches != 1:
            raise ToolExecutionError(f"Expected text matched {matches} times; it must match exactly once.")
        updated = previous.replace(old_text, _string(arguments, "new_text"), 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as error:
            raise ToolExecutionError("Unable to edit the requested workspace file.") from error
        return ToolExecution(True, "", _diff_summary(path.name, previous, updated))


class FindFilesTool:
    def __init__(self, policy: WorkspacePolicy) -> None:
        self.policy = policy

    @property
    def definition(self) -> ToolDefinition:
        return _definition(
            "find_files",
            "按 glob 模式查找工作区中的非生成文件。全项目清单使用 **/*；聚焦检查时使用更具体的模式或路径。",
            ("pattern",),
            {"pattern": {"type": "string", "minLength": 1}, "path": _path_schema()},
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        _validate_arguments(arguments, required={"pattern"}, optional={"path"})
        pattern = _string(arguments, "pattern")
        _validate_pattern(pattern)
        scope = _prepare_scope(self.policy, arguments)
        prepared = freeze_arguments({"pattern": pattern, "scope": scope})
        return ToolPreparation(prepared, self.policy.relative_target(scope))

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        del cancel_event
        pattern = _string(arguments, "pattern")
        scope = _prepared_path(arguments, "scope", self.policy)
        try:
            candidates = scope.glob(pattern) if scope.is_dir() else (scope if scope.match(pattern) else None,)
        except (OSError, ValueError) as error:
            raise ToolExecutionError("Unable to search the requested workspace scope.") from error
        paths, limited = _matching_paths(self.policy, candidates)
        output = "\n".join(paths)
        if limited:
            output = _with_limit_notice(output, _FILE_RESULTS_NOTICE)
        return ToolExecution(True, output, f"found {len(paths)} file(s)")


class SearchCodeTool:
    def __init__(self, policy: WorkspacePolicy) -> None:
        self.policy = policy

    @property
    def definition(self) -> ToolDefinition:
        return _definition(
            "search_code",
            "在工作区的 UTF-8 文件中搜索字面文本。已知范围时通过 path 限定搜索位置。",
            ("query",),
            {"query": {"type": "string", "minLength": 1}, "path": _path_schema()},
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        _validate_arguments(arguments, required={"query"}, optional={"path"})
        query = _string(arguments, "query")
        if not query:
            raise ToolExecutionError("Search query must not be empty.")
        scope = _prepare_scope(self.policy, arguments)
        prepared = freeze_arguments({"query": query, "scope": scope})
        return ToolPreparation(prepared, self.policy.relative_target(scope))

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        del cancel_event
        query = _string(arguments, "query")
        scope = _prepared_path(arguments, "scope", self.policy)
        deadline = monotonic() + _SCAN_TIMEOUT_SECONDS
        matches: list[str] = []
        limited = False
        for path in _candidate_files(scope):
            if monotonic() >= deadline or len(matches) >= _MAX_MATCHES:
                limited = True
                break
            if _is_ignored_discovery_path(path, self.policy):
                continue
            try:
                safe_path = self.policy.resolve_path(str(path))
                content = _read_text(safe_path)
            except (ToolPolicyError, ToolExecutionError):
                continue
            if "\0" in content:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if query in line:
                    relative = self.policy.relative_target(safe_path)
                    matches.append(f"{relative}:{line_number}: {line}")
                    if len(matches) >= _MAX_MATCHES:
                        limited = True
                        break
            if limited:
                break
        output = "\n".join(matches)
        if limited:
            output = _with_limit_notice(output, _SEARCH_RESULTS_NOTICE)
        return ToolExecution(True, output, f"found {len(matches)} match(es)")


def _definition(name: str, description: str, required: tuple[str, ...], properties: dict[str, object]) -> ToolDefinition:
    return ToolDefinition(name, description, {"type": "object", "required": list(required), "properties": properties})


def _path_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


def _string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolExecutionError(f"Tool requires string argument '{name}'.")
    return value


def _path(arguments: Mapping[str, object], name: str) -> Path:
    value = arguments.get(name)
    if not isinstance(value, Path):
        raise ToolExecutionError(f"Prepared tool arguments require path '{name}'.")
    return value


def _prepared_path(
    arguments: Mapping[str, object],
    name: str,
    policy: WorkspacePolicy,
    *,
    allow_context_artifact_read: bool = False,
) -> Path:
    """Recheck a frozen path immediately before I/O to catch post-approval replacement."""

    authorized = _path(arguments, name)
    current = (
        policy.resolve_path(str(authorized), allow_context_artifact_read=True)
        if allow_context_artifact_read
        else policy.resolve_path(str(authorized))
    )
    if current != authorized:
        raise ToolPolicyError("The prepared workspace path changed before execution.")
    return current


def _prepare_scope(policy: WorkspacePolicy, arguments: Mapping[str, object]) -> Path:
    value = arguments.get("path", ".")
    if not isinstance(value, str):
        raise ToolExecutionError("Tool requires optional string argument 'path'.")
    scope = policy.resolve_path(value)
    if not scope.exists():
        raise ToolExecutionError("The requested workspace scope does not exist.")
    return scope


def _validate_arguments(
    arguments: Mapping[str, object], *, required: set[str], optional: set[str] | None = None
) -> None:
    allowed = required | (optional or set())
    missing = required - set(arguments)
    if missing:
        names = ", ".join(sorted(missing))
        raise ToolExecutionError(f"Tool requires argument(s): {names}.")
    unexpected = set(arguments) - allowed
    if unexpected:
        raise ToolExecutionError("Tool received unexpected arguments.")


def _validate_pattern(pattern: str) -> None:
    parsed = Path(pattern)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ToolExecutionError("File patterns must stay within the workspace scope.")


def _matching_paths(policy: WorkspacePolicy, candidates: Iterable[Path | None]) -> tuple[list[str], bool]:
    matches: list[str] = []
    deadline = monotonic() + _SCAN_TIMEOUT_SECONDS
    for candidate in candidates:
        if candidate is None:
            continue
        if monotonic() >= deadline or len(matches) >= _MAX_MATCHES:
            break
        try:
            safe_path = policy.resolve_path(str(candidate))
        except ToolPolicyError:
            continue
        if _is_ignored_discovery_path(safe_path, policy):
            continue
        if safe_path.is_file():
            matches.append(policy.relative_target(safe_path))
    return matches, monotonic() >= deadline or len(matches) >= _MAX_MATCHES


def _candidate_files(scope: Path) -> Iterable[Path]:
    if scope.is_file():
        return (scope,)
    try:
        return scope.rglob("*")
    except OSError as error:
        raise ToolExecutionError("Unable to search the requested workspace scope.") from error


def _is_ignored_discovery_path(path: Path, policy: WorkspacePolicy) -> bool:
    try:
        relative = Path(policy.relative_target(path))
    except (ValueError, ToolPolicyError):
        return True
    folded = tuple(part.casefold() for part in relative.parts)
    return (
        any(part in _IGNORED_DISCOVERY_DIRECTORIES for part in folded)
        or folded[:2] in {
            (".fakuicode", "worktrees"),
            (".fakuicode", "worktree-state"),
        }
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ToolExecutionError("Unable to read the requested workspace file.") from error


def _bounded(content: str, *, limit: int, suffix: str = "\n… output truncated") -> str:
    return content if len(content) <= limit else content[:limit] + suffix


def _with_limit_notice(content: str, notice: str) -> str:
    return content + notice


def _diff_summary(name: str, before: str, after: str) -> str:
    diff = "".join(
        difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=name, tofile=name)
    )
    return _bounded(diff, limit=_MAX_DIFF) or f"no content change in {name}"
