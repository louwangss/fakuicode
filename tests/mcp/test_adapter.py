from __future__ import annotations

from types import MappingProxyType

import pytest

from fakuicode.errors import ToolExecutionError
from fakuicode.mcp.adapter import build_adapters
from fakuicode.mcp.models import (
    McpCallResult,
    McpContentBlock,
    McpRemoteTool,
    ResolvedStdioServerConfig,
)
from fakuicode.permissions.models import PermissionScope


class _Manager:
    def __init__(self, tools: dict[str, tuple[McpRemoteTool, ...]]) -> None:
        self.tools = tools
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def discovered_tools(self):
        return MappingProxyType(self.tools)

    def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, object],
        *,
        cancel_event: object = None,
    ) -> McpCallResult:
        del cancel_event
        self.calls.append((server, tool, arguments))
        return McpCallResult(
            (McpContentBlock("text", "first"), McpContentBlock("image")),
            {"z": 1, "a": 2},
        )


def _config(**changes: object) -> ResolvedStdioServerConfig:
    values = {"name": "server", "command": "x", "args": (), "environment": {}}
    values.update(changes)
    return ResolvedStdioServerConfig(**values)


def test_names_filters_schema_and_stable_order() -> None:
    manager = _Manager(
        {
            "server": (
                McpRemoteTool("Z Tool", None, None),
                McpRemoteTool("alpha", "Alpha", {"type": "object", "properties": {"n": {"type": "integer"}}}),
                McpRemoteTool(7, "invalid", {}),
            )
        }
    )
    adapters = build_adapters(manager, (_config(disabled_tools=frozenset({"alpha"})),))  # type: ignore[arg-type]
    assert len(adapters) == 1
    assert adapters[0].definition.name.startswith("mcp__server__z_tool_")
    assert len(adapters[0].definition.name) <= 64
    assert adapters[0].definition.input_schema["type"] == "object"
    assert adapters[0].definition.input_schema["additionalProperties"] is False


def test_adapter_validates_arguments_calls_original_name_and_formats_result() -> None:
    manager = _Manager(
        {"server": (McpRemoteTool("remote-name", "Does work", {"type": "object", "required": ["value"]}),)}
    )
    adapter = build_adapters(manager, (_config(),))[0]  # type: ignore[arg-type]
    with pytest.raises(ToolExecutionError):
        adapter.prepare({})
    prepared = adapter.prepare({"value": 3})
    execution = adapter.execute_prepared(prepared.arguments)
    assert prepared.permission_scope is PermissionScope.TOOL
    assert prepared.target == "__all_arguments__"
    assert manager.calls == [("server", "remote-name", {"value": 3})]
    assert execution.success
    assert execution.output == 'first\n[image content omitted]\n{"a":2,"z":1}'
    assert execution.duration_seconds is not None


def test_adapter_preserves_text_beyond_the_old_character_limit() -> None:
    class LongResultManager(_Manager):
        def call_tool(
            self,
            server: str,
            tool: str,
            arguments: dict[str, object],
            *,
            cancel_event: object = None,
        ) -> McpCallResult:
            del server, tool, arguments, cancel_event
            return McpCallResult((McpContentBlock("text", "x" * 13_000 + "tail-marker"),))

    manager = LongResultManager(
        {"server": (McpRemoteTool("long", "Long result", {"type": "object"}),)}
    )

    execution = build_adapters(manager, (_config(),))[0].execute({})

    assert execution.output.endswith("tail-marker")
    assert "[truncated]" not in execution.output


def test_disabled_filter_wins_over_enabled_filter() -> None:
    manager = _Manager({"server": (McpRemoteTool("one", "", {"type": "object"}),)})
    adapters = build_adapters(
        manager,
        (_config(enabled_tools=frozenset({"one"}), disabled_tools=frozenset({"one"})),),  # type: ignore[arg-type]
    )
    assert adapters == ()


def test_invalid_schema_and_all_duplicate_remote_names_are_skipped() -> None:
    manager = _Manager(
        {
            "server": (
                McpRemoteTool("duplicate", "", {"type": "object"}),
                McpRemoteTool("duplicate", "", {"type": "object"}),
                McpRemoteTool("invalid", "", "not-a-schema"),
            )
        }
    )
    assert build_adapters(manager, (_config(),)) == ()  # type: ignore[arg-type]
