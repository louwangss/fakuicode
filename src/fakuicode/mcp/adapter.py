"""Expose discovered MCP tools through fakuiCode's existing Tool contract."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
import hashlib
import json
import re
from threading import Event
import time

from jsonschema import Draft202012Validator, ValidationError, SchemaError

from fakuicode.errors import RequestCancelled, ToolExecutionError
from fakuicode.mcp.models import McpRemoteTool, McpToolBinding, ResolvedServerConfig
from fakuicode.mcp.runtime import McpClientManager
from fakuicode.models import ToolDefinition
from fakuicode.permissions.models import PermissionScope
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments


_ALL_ARGUMENTS = "__all_arguments__"
_MAX_SUMMARY = 120


class McpToolAdapter:
    def __init__(self, binding: McpToolBinding, manager: McpClientManager) -> None:
        self.binding = binding
        self.manager = manager
        self._definition = ToolDefinition(
            binding.public_name, binding.description, binding.input_schema
        )
        self._validator = Draft202012Validator(dict(binding.input_schema))

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        copied = dict(arguments)
        try:
            self._validator.validate(copied)
        except ValidationError as error:
            path = ".".join(str(item) for item in error.absolute_path)
            label = f" at '{path}'" if path else ""
            raise ToolExecutionError(f"MCP tool arguments are invalid{label}.") from error
        return ToolPreparation(
            freeze_arguments(copied), _ALL_ARGUMENTS, PermissionScope.TOOL
        )

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelled("MCP tool call cancelled")
        started = time.monotonic()
        result = self.manager.call_tool(
            self.binding.server_name,
            self.binding.remote_name,
            dict(arguments),
            cancel_event=cancel_event,
        )
        duration = time.monotonic() - started
        output = _render_result(result)
        summary = result.public_summary or ("MCP tool reported an error." if result.is_error else "MCP tool completed.")
        return ToolExecution(
            not result.is_error,
            output,
            summary[:_MAX_SUMMARY],
            duration,
        )

    def execute(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        prepared = self.prepare(arguments)
        return self.execute_prepared(prepared.arguments, cancel_event=cancel_event)


def build_adapters(
    manager: McpClientManager,
    configs: tuple[ResolvedServerConfig, ...],
) -> tuple[McpToolAdapter, ...]:
    configs_by_name = {config.name: config for config in configs}
    adapters: list[McpToolAdapter] = []
    used_names: set[str] = set()
    for server_name, tools in sorted(manager.discovered_tools().items()):
        config = configs_by_name.get(server_name)
        if config is None:
            continue
        name_counts = Counter(
            remote.name for remote in tools if isinstance(remote.name, str) and remote.name
        )
        for remote in tools:
            if (
                not isinstance(remote.name, str)
                or not remote.name
                or name_counts[remote.name] != 1
            ):
                continue
            if config.enabled_tools is not None and remote.name not in config.enabled_tools:
                continue
            if remote.name in config.disabled_tools:
                continue
            public_name = _public_name(server_name, remote.name)
            if public_name in used_names:
                continue
            used_names.add(public_name)
            schema = _schema(remote.input_schema)
            if schema is None:
                continue
            binding = McpToolBinding(
                public_name,
                server_name,
                remote.name,
                _description(server_name, remote),
                schema,
            )
            adapters.append(McpToolAdapter(binding, manager))
    return tuple(sorted(adapters, key=lambda adapter: adapter.definition.name))


def _public_name(server_name: str, remote_name: str) -> str:
    lowered = remote_name.lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_") or "tool"
    if not re.match(r"[a-z_]", slug):
        slug = f"tool_{slug}"
    base = f"mcp__{server_name}__{slug}"
    changed = slug != remote_name
    if changed or len(base) > 64:
        digest = hashlib.sha256(remote_name.encode("utf-8")).hexdigest()[:8]
        prefix_length = 64 - len(f"mcp__{server_name}___{digest}")
        slug = slug[: max(prefix_length, 1)].rstrip("_") or "tool"
        base = f"mcp__{server_name}__{slug}_{digest}"
    return base[:64]


def _description(server_name: str, remote: McpRemoteTool) -> str:
    if isinstance(remote.description, str) and remote.description.strip():
        return remote.description.strip()[:2_000]
    return f"MCP tool '{remote.name}' provided by server '{server_name}'."


def _schema(value: object) -> Mapping[str, object] | None:
    fallback: Mapping[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    if value is None:
        return fallback
    if not isinstance(value, Mapping):
        return None
    schema = dict(value)
    try:
        serialized = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > 65_536 or _depth(schema) > 32:
            return None
        Draft202012Validator.check_schema(schema)
    except (SchemaError, TypeError, ValueError, RecursionError):
        return None
    if schema.get("type") != "object":
        return None
    return schema


def _depth(value: object, current: int = 0) -> int:
    if current > 32:
        return current
    if isinstance(value, Mapping):
        return max((_depth(item, current + 1) for item in value.values()), default=current)
    if isinstance(value, (list, tuple)):
        return max((_depth(item, current + 1) for item in value), default=current)
    return current


def _render_result(result: object) -> str:
    from fakuicode.mcp.models import McpCallResult

    assert isinstance(result, McpCallResult)
    pieces: list[str] = []
    for block in result.content:
        if block.kind == "text" and block.text is not None:
            pieces.append(block.text)
        else:
            pieces.append(f"[{block.kind} content omitted]")
    if result.structured_content is not None:
        try:
            pieces.append(
                json.dumps(
                    result.structured_content,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError):
            pieces.append("[structured content omitted]")
    if not pieces and result.public_summary:
        pieces.append(result.public_summary)
    return "\n".join(pieces)
