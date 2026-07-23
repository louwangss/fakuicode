from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.permissions.config import PermissionConfigSnapshot
from fakuicode.permissions.engine import PermissionEngine
from fakuicode.permissions.models import (
    DecisionKind,
    PermissionMode,
    PermissionSubject,
    RuleEffect,
    RuleSource,
)
from fakuicode.permissions.rules import parse_rule
from fakuicode.permissions.safety import DangerousCommandGuard


def _rule(expression: str, effect: RuleEffect, source: RuleSource):
    return parse_rule(expression, effect, source)


def _engine(tmp_path: Path, snapshot: PermissionConfigSnapshot) -> PermissionEngine:
    return PermissionEngine(snapshot, DangerousCommandGuard(tmp_path))


def test_hard_deny_cannot_be_overridden_by_rules_or_trusted_mode(tmp_path: Path) -> None:
    snapshot = PermissionConfigSnapshot(
        mode=PermissionMode.TRUSTED,
        user_rules=(_rule("run_command(*)", RuleEffect.ALLOW, RuleSource.USER),),
    )
    subject = PermissionSubject("run_command", "bash -lc pwd", read_only=False)

    decision = _engine(tmp_path, snapshot).decide(subject)

    assert decision.kind is DecisionKind.DENY
    assert decision.layer == "dangerous_command"


def test_global_deny_is_checked_before_more_specific_session_allow(tmp_path: Path) -> None:
    snapshot = PermissionConfigSnapshot(
        user_rules=(_rule("run_command(git *)", RuleEffect.DENY, RuleSource.USER),)
    )
    session = (_rule("run_command(git status)", RuleEffect.ALLOW, RuleSource.SESSION),)

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject("run_command", "git status", read_only=False), session_rules=session
    )

    assert decision.kind is DecisionKind.DENY
    assert decision.layer == "user_global_deny"


def test_rule_sources_are_checked_from_session_to_user(tmp_path: Path) -> None:
    snapshot = PermissionConfigSnapshot(
        user_rules=(_rule("write_file(*)", RuleEffect.ALLOW, RuleSource.USER),),
        project_shared_rules=(
            _rule("write_file(src/*)", RuleEffect.DENY, RuleSource.PROJECT_SHARED),
        ),
        project_local_rules=(
            _rule("write_file(src/main.py)", RuleEffect.ALLOW, RuleSource.PROJECT_LOCAL),
        ),
        project_trusted=True,
    )
    session = (_rule("write_file(src/main.py)", RuleEffect.ALLOW, RuleSource.SESSION),)

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject("write_file", "src/main.py", read_only=False), session_rules=session
    )

    assert decision.kind is DecisionKind.ALLOW
    assert decision.rule is session[0]


def test_untrusted_project_ignores_shared_allow_but_keeps_shared_deny(tmp_path: Path) -> None:
    allow = _rule("write_file(src/*)", RuleEffect.ALLOW, RuleSource.PROJECT_SHARED)
    deny = _rule("write_file(dist/*)", RuleEffect.DENY, RuleSource.PROJECT_SHARED)
    snapshot = PermissionConfigSnapshot(project_shared_rules=(allow, deny), project_trusted=False)
    engine = _engine(tmp_path, snapshot)

    source = engine.decide(PermissionSubject("write_file", "src/main.py", read_only=False))
    distribution = engine.decide(PermissionSubject("write_file", "dist/app.js", read_only=False))

    assert source.kind is DecisionKind.ASK
    assert distribution.kind is DecisionKind.DENY
    assert distribution.rule is deny


def test_trusted_project_enables_shared_allow(tmp_path: Path) -> None:
    allow = _rule("write_file(src/*)", RuleEffect.ALLOW, RuleSource.PROJECT_SHARED)
    snapshot = PermissionConfigSnapshot(project_shared_rules=(allow,), project_trusted=True)

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject("write_file", "src/main.py", read_only=False)
    )

    assert decision.kind is DecisionKind.ALLOW
    assert decision.rule is allow


@pytest.mark.parametrize("mode", list(PermissionMode))
def test_safe_reads_are_allowed_in_every_mode(tmp_path: Path, mode: PermissionMode) -> None:
    snapshot = PermissionConfigSnapshot(mode=mode)

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject("read_file", "src/main.py", read_only=True)
    )

    assert decision.kind is DecisionKind.ALLOW
    assert decision.layer == "safe_read"


def test_explicit_deny_still_blocks_safe_read(tmp_path: Path) -> None:
    deny = _rule("read_file(src/*)", RuleEffect.DENY, RuleSource.PROJECT_LOCAL)
    snapshot = PermissionConfigSnapshot(project_local_rules=(deny,))

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject("read_file", "src/main.py", read_only=True)
    )

    assert decision.kind is DecisionKind.DENY


@pytest.mark.parametrize(
    ("mode", "tool", "expected"),
    [
        (PermissionMode.STRICT, "write_file", DecisionKind.DENY),
        (PermissionMode.STRICT, "run_command", DecisionKind.DENY),
        (PermissionMode.DEFAULT, "write_file", DecisionKind.ASK),
        (PermissionMode.DEFAULT, "run_command", DecisionKind.ASK),
        (PermissionMode.TRUSTED, "write_file", DecisionKind.ALLOW),
        (PermissionMode.TRUSTED, "edit_file", DecisionKind.ALLOW),
        (PermissionMode.TRUSTED, "run_command", DecisionKind.ASK),
    ],
)
def test_permission_modes_apply_only_after_rules(
    tmp_path: Path, mode: PermissionMode, tool: str, expected: DecisionKind
) -> None:
    snapshot = PermissionConfigSnapshot(mode=mode)

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject(tool, "target", read_only=False)
    )

    assert decision.kind is expected


def test_locked_state_allows_safe_reads_but_never_asks_for_side_effects(tmp_path: Path) -> None:
    snapshot = PermissionConfigSnapshot(mode=PermissionMode.STRICT, locked=True)
    engine = _engine(tmp_path, snapshot)

    read = engine.decide(PermissionSubject("read_file", "src/main.py", read_only=True))
    write = engine.decide(PermissionSubject("write_file", "src/main.py", read_only=False))

    assert read.kind is DecisionKind.ALLOW
    assert write.kind is DecisionKind.DENY
    assert write.layer == "locked_config"


def test_plan_task_boundary_rejects_side_effect_even_with_allow_rule(tmp_path: Path) -> None:
    snapshot = PermissionConfigSnapshot(
        user_rules=(_rule("write_file(*)", RuleEffect.ALLOW, RuleSource.USER),)
    )

    decision = _engine(tmp_path, snapshot).decide(
        PermissionSubject("write_file", "src/main.py", read_only=False), read_only_task=True
    )

    assert decision.kind is DecisionKind.DENY
    assert decision.layer == "task_mode"
