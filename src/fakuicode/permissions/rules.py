"""Strict parsing and matching for ``tool(pattern)`` permission rules."""

from __future__ import annotations

import re
from collections.abc import Iterable

from fakuicode.permissions.models import Rule, RuleEffect, RuleSource
from fakuicode.matching import GlobSyntaxError, compile_glob


KNOWN_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "find_files",
        "search_code",
        "team_create",
        "team_task_create",
        "team_task_get",
        "team_task_list",
        "team_task_update",
        "team_task_delete",
        "team_message_send",
        "team_inbox_list",
        "team_plan_submit",
        "team_plan_review",
        "team_member_start",
        "team_member_assign",
        "team_member_resume",
        "team_member_stop",
        "team_task_complete",
        "team_integrate_task",
        "team_finalize_prepare",
        "team_finalize",
    }
)
_RULE_PATTERN = re.compile(r"([a-z_][a-z0-9_]{0,63})\((.+)\)", re.ASCII)
_MCP_TOOL_PATTERN = re.compile(r"mcp__[a-z][a-z0-9_]{0,31}__[a-z_][a-z0-9_]*", re.ASCII)
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


class RuleSyntaxError(ValueError):
    """Raised when a rule is ambiguous or outside the supported grammar."""


def parse_rule(expression: str, effect: RuleEffect, source: RuleSource) -> Rule:
    if not isinstance(expression, str) or expression != expression.strip() or _CONTROL_PATTERN.search(expression):
        raise RuleSyntaxError("Permission rules must be one trimmed printable line.")
    match = _RULE_PATTERN.fullmatch(expression)
    if match is None:
        raise RuleSyntaxError("Permission rules must use tool(pattern) syntax.")
    tool_name, pattern = match.groups()
    if tool_name not in KNOWN_TOOLS and (
        len(tool_name) > 64 or _MCP_TOOL_PATTERN.fullmatch(tool_name) is None
    ):
        raise RuleSyntaxError(f"Unknown permission tool '{tool_name}'.")
    try:
        matcher, exact = compile_glob(pattern)
    except GlobSyntaxError as error:
        raise RuleSyntaxError(str(error)) from error
    return Rule(expression, tool_name, pattern, RuleEffect(effect), RuleSource(source), exact, matcher)


def escape_exact_target(target: str) -> str:
    """Escape every supported glob metacharacter for one exact generated rule."""

    if not target or _CONTROL_PATTERN.search(target):
        raise RuleSyntaxError("Exact permission targets must be one non-empty printable line.")
    escaped: list[str] = []
    for character in target:
        if character in {"\\", "*", "?", "[", "]"}:
            escaped.append("\\")
        escaped.append(character)
    return "".join(escaped)


def select_rule(rules: Iterable[Rule], tool_name: str, target: str) -> Rule | None:
    matches = [rule for rule in rules if rule.tool_name == tool_name and rule.matches(target)]
    if not matches:
        return None
    return min(matches, key=lambda rule: (not rule.exact, rule.effect is not RuleEffect.DENY))
