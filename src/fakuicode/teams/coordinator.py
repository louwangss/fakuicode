"""Coordinator-mode tool scoping."""

from __future__ import annotations

from collections.abc import Iterable

from fakuicode.tools.registry import ToolRegistry


_COORDINATOR_READ_TOOLS = frozenset(
    {
        "read_file",
        "find_files",
        "search_code",
    }
)


def apply_coordinator_scope(
    registry: ToolRegistry,
    team_tool_names: Iterable[str],
) -> frozenset[str]:
    """Limit a Lead to read-only inspection and explicit Team control tools."""

    requested_team_tools = frozenset(team_tool_names)
    missing = requested_team_tools.difference(registry.all_names())
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Coordinator 工具尚未注册：{names}")
    visible = (_COORDINATOR_READ_TOOLS | requested_team_tools).intersection(
        registry.all_names()
    )
    registry.set_tool_ceiling(visible, include_system_tools=False)
    registry.set_visible_tools(visible, include_system_tools=False)
    return frozenset(visible)
