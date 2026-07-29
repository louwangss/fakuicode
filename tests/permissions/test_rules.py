from __future__ import annotations

import pytest

from fakuicode.permissions.models import ApprovalChoice, DecisionKind, PermissionMode, RuleEffect, RuleSource
from fakuicode.permissions.rules import RuleSyntaxError, escape_exact_target, parse_rule, select_rule


def test_permission_enums_reject_unknown_values() -> None:
    assert PermissionMode("strict") is PermissionMode.STRICT
    assert DecisionKind("ask") is DecisionKind.ASK
    assert ApprovalChoice("session") is ApprovalChoice.SESSION
    with pytest.raises(ValueError):
        PermissionMode("bypass")


@pytest.mark.parametrize(
    ("expression", "target"),
    [
        ("run_command(git status)", "git status"),
        ("read_file(src/main.py)", "src/main.py"),
    ],
)
def test_rule_without_wildcards_is_exact(expression: str, target: str) -> None:
    rule = parse_rule(expression, RuleEffect.ALLOW, RuleSource.USER)

    assert rule.exact is True
    assert rule.matches(target)
    assert not rule.matches(target + ".bak")


@pytest.mark.parametrize(
    ("expression", "matching", "not_matching"),
    [
        ("run_command(git *)", "git status", "python -m pytest"),
        ("read_file(src/**)", "src/pkg/main.py", "tests/test_main.py"),
        ("read_file(src/?.py)", "src/a.py", "src/main.py"),
        ("read_file(src/[ab].py)", "src/a.py", "src/c.py"),
    ],
)
def test_rule_with_wildcards_uses_glob_matching(
    expression: str, matching: str, not_matching: str
) -> None:
    rule = parse_rule(expression, RuleEffect.ALLOW, RuleSource.USER)

    assert rule.exact is False
    assert rule.matches(matching)
    assert not rule.matches(not_matching)


@pytest.mark.parametrize(
    "expression",
    [
        "Bash(git *)",
        "unknown_tool(*)",
        "run_command",
        "run_command()",
        "run_command(git *)) trailing",
        "run_command(git [abc)",
        r"run_command(git \)",
        "run_command(git\nstatus)",
    ],
)
def test_invalid_or_ambiguous_rules_are_rejected(expression: str) -> None:
    with pytest.raises(RuleSyntaxError):
        parse_rule(expression, RuleEffect.ALLOW, RuleSource.USER)


def test_exact_rule_escapes_glob_metacharacters_without_broadening() -> None:
    target = "python -c \"print('*?[x]\\\\')\""
    expression = f"run_command({escape_exact_target(target)})"

    rule = parse_rule(expression, RuleEffect.ALLOW, RuleSource.SESSION)

    assert rule.exact is True
    assert rule.matches(target)
    assert not rule.matches("python -c \"print('anything')\"")


def test_exact_match_wins_over_glob_in_the_same_source() -> None:
    rules = (
        parse_rule("run_command(git *)", RuleEffect.DENY, RuleSource.PROJECT_LOCAL),
        parse_rule("run_command(git status)", RuleEffect.ALLOW, RuleSource.PROJECT_LOCAL),
    )

    selected = select_rule(rules, "run_command", "git status")

    assert selected is not None
    assert selected.effect is RuleEffect.ALLOW
    assert selected.exact is True


def test_deny_wins_when_matches_have_equal_specificity() -> None:
    rules = (
        parse_rule("read_file(src/*)", RuleEffect.ALLOW, RuleSource.USER),
        parse_rule("read_file(src/*)", RuleEffect.DENY, RuleSource.USER),
    )

    selected = select_rule(rules, "read_file", "src/main.py")

    assert selected is not None
    assert selected.effect is RuleEffect.DENY


def test_select_rule_ignores_other_tools_and_non_matches() -> None:
    rules = (
        parse_rule("write_file(src/*)", RuleEffect.DENY, RuleSource.USER),
        parse_rule("read_file(tests/*)", RuleEffect.ALLOW, RuleSource.USER),
    )

    assert select_rule(rules, "read_file", "src/main.py") is None


def test_dynamic_mcp_tool_rule_is_strictly_namespaced() -> None:
    rule = parse_rule(
        "mcp__docs__lookup(__all_arguments__)", RuleEffect.ALLOW, RuleSource.USER
    )
    assert rule.exact
    assert rule.matches("__all_arguments__")
    with pytest.raises(RuleSyntaxError):
        parse_rule("mcp__Docs__lookup(*)", RuleEffect.ALLOW, RuleSource.USER)


@pytest.mark.parametrize(
    "tool_name",
    [
        "team_create",
        "team_member_start",
        "team_task_create",
        "team_message_send",
        "team_integrate_task",
        "team_finalize",
    ],
)
def test_team_tools_support_explicit_permission_rules(tool_name: str) -> None:
    rule = parse_rule(
        f"{tool_name}(team:alpha)",
        RuleEffect.ALLOW,
        RuleSource.USER,
    )

    assert rule.matches("team:alpha")
