"""Compatibility parsing for models that emit DeepSeek-style tool markup as text."""

from __future__ import annotations

from dataclasses import dataclass
import re

from fakuicode.models import ToolCall


TOOL_CALL_PREFIX = "<｜｜DSML｜｜tool_calls>"
_TOOL_CALL_SUFFIX = "</｜｜DSML｜｜tool_calls>"
_INVOKE_PATTERN = re.compile(
    r'<｜｜DSML｜｜invoke\s+name="(?P<name>[^"]+)">(?P<body>.*?)</｜｜DSML｜｜invoke>',
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(
    r'<｜｜DSML｜｜parameter\s+name="(?P<name>[^"]+)"\s+string="true">(?P<value>.*?)</｜｜DSML｜｜parameter>',
    re.DOTALL,
)


@dataclass
class ToolMarkupAccumulator:
    """Separate streamed visible text from one trailing DSML tool-call block."""

    _visible_tail: str = ""
    _raw_markup: str | None = None

    def append(self, text: str) -> tuple[str, ...]:
        if self._raw_markup is not None:
            self._raw_markup += text
            return ()
        content = self._visible_tail + text
        marker = content.find(TOOL_CALL_PREFIX)
        if marker >= 0:
            visible = content[:marker]
            self._visible_tail = ""
            self._raw_markup = content[marker:]
            return (visible,) if visible else ()
        prefix_length = _trailing_prefix_length(content)
        visible = content[:-prefix_length] if prefix_length else content
        self._visible_tail = content[-prefix_length:] if prefix_length else ""
        return (visible,) if visible else ()

    def finish(self) -> tuple[tuple[str, ...], list[ToolCall] | None]:
        visible = [self._visible_tail] if self._visible_tail else []
        if self._raw_markup is None:
            return tuple(visible), None
        calls = parse_tool_calls(self._raw_markup)
        if calls is None:
            visible.append(self._raw_markup)
        return tuple(visible), calls


def parse_tool_calls(content: str) -> list[ToolCall] | None:
    """Convert supported DSML tool markup to the public Fakuicode tool contract."""
    if not content.startswith(TOOL_CALL_PREFIX) or not content.endswith(_TOOL_CALL_SUFFIX):
        return None
    calls: list[ToolCall] = []
    for index, invocation in enumerate(_INVOKE_PATTERN.finditer(content), start=1):
        parameters = {
            parameter.group("name"): parameter.group("value")
            for parameter in _PARAMETER_PATTERN.finditer(invocation.group("body"))
        }
        translated = _translate_tool(invocation.group("name"), parameters)
        if translated is not None:
            name, arguments = translated
            calls.append(ToolCall(f"dsml-{index}", name, arguments))
    return calls or None


def _translate_tool(name: str, parameters: dict[str, str]) -> tuple[str, dict[str, str]] | None:
    if name == "list_dir":
        return "find_files", {"pattern": "**/*", "path": _first(parameters, "dirPath") or "."}
    if name == "read_file":
        path = _first(parameters, "path", "filePath", "file_path")
        return ("read_file", {"path": path}) if path is not None else None
    if name == "find_files":
        pattern = _first(parameters, "pattern", "glob")
        if pattern is None:
            return None
        path = _first(parameters, "path", "dirPath", "directory")
        # DSML models often use `find_files("*")` as a directory-listing
        # request.  In the workspace root that otherwise hides every source
        # file below `src/`, leaving the model unable to identify an entry
        # point in its single allowed tool batch.
        if pattern in {"*", "./*"} and path in {None, "", ".", "./"}:
            pattern = "**/*"
        arguments = {"pattern": pattern}
        if path is not None:
            arguments["path"] = path
        return "find_files", arguments
    if name in {"search_code", "grep"}:
        query = _first(parameters, "query", "pattern", "text")
        if query is None:
            return None
        arguments = {"query": query}
        path = _first(parameters, "path", "dirPath", "directory")
        if path is not None:
            arguments["path"] = path
        return "search_code", arguments
    return None


def _first(parameters: dict[str, str], *names: str) -> str | None:
    normalized = {_normalize_parameter_name(name): value for name, value in parameters.items()}
    return next(
        (normalized[key] for name in names if (key := _normalize_parameter_name(name)) in normalized),
        None,
    )


def _normalize_parameter_name(name: str) -> str:
    """Treat common DSML casing and separator variants as the same field."""
    return re.sub(r"[-_]", "", name).casefold()


def _trailing_prefix_length(content: str) -> int:
    maximum = min(len(content), len(TOOL_CALL_PREFIX) - 1)
    for length in range(maximum, 0, -1):
        if content.endswith(TOOL_CALL_PREFIX[:length]):
            return length
    return 0
