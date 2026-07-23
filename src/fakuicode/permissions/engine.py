"""Deterministic layered permission decisions without UI or filesystem writes."""

from __future__ import annotations

from collections.abc import Iterable

from fakuicode.permissions.config import PermissionConfigSnapshot
from fakuicode.permissions.models import (
    Decision,
    DecisionKind,
    PermissionMode,
    PermissionSubject,
    Rule,
    RuleEffect,
)
from fakuicode.permissions.rules import select_rule
from fakuicode.permissions.safety import DangerousCommandGuard


class PermissionEngine:
    def __init__(self, snapshot: PermissionConfigSnapshot, command_guard: DangerousCommandGuard) -> None:
        self.snapshot = snapshot
        self.command_guard = command_guard

    def decide(
        self,
        subject: PermissionSubject,
        *,
        session_rules: Iterable[Rule] = (),
        mode: PermissionMode | None = None,
        read_only_task: bool = False,
    ) -> Decision:
        if read_only_task and not subject.read_only:
            return Decision(DecisionKind.DENY, "Plan mode permits only read-only tools.", "task_mode")

        if subject.tool_name == "run_command":
            reason = self.command_guard.reason(subject.target)
            if reason is not None:
                return Decision(DecisionKind.DENY, reason, "dangerous_command")

        global_deny = _select_deny(self.snapshot.user_rules, subject)
        if global_deny is not None:
            return _rule_decision(global_deny, "user_global_deny")

        selected = self._select_layered_rule(subject, tuple(session_rules))
        if selected is not None and selected.effect is RuleEffect.DENY:
            return _rule_decision(selected, "rule")

        if self.snapshot.locked and not subject.read_only:
            return Decision(
                DecisionKind.DENY,
                "Permission configuration is invalid; side effects are locked until restart.",
                "locked_config",
            )

        if selected is not None:
            return _rule_decision(selected, "rule")

        if subject.read_only:
            return Decision(DecisionKind.ALLOW, "Safe workspace read is allowed.", "safe_read")

        active_mode = mode or self.snapshot.mode
        if active_mode is PermissionMode.STRICT:
            return Decision(DecisionKind.DENY, "Strict mode requires an explicit allow rule.", "mode")
        if active_mode is PermissionMode.TRUSTED and subject.tool_name in {"write_file", "edit_file"}:
            return Decision(DecisionKind.ALLOW, "Trusted mode allows workspace file changes.", "mode")
        return Decision(DecisionKind.ASK, "This action requires user confirmation.", "mode")

    def _select_layered_rule(self, subject: PermissionSubject, session_rules: tuple[Rule, ...]) -> Rule | None:
        shared = self.snapshot.project_shared_rules
        if not self.snapshot.project_trusted:
            shared = tuple(rule for rule in shared if rule.effect is RuleEffect.DENY)
        for rules in (
            session_rules,
            self.snapshot.project_local_rules,
            shared,
            self.snapshot.user_rules,
        ):
            selected = select_rule(rules, subject.tool_name, subject.target)
            if selected is not None:
                return selected
        return None


def _select_deny(rules: Iterable[Rule], subject: PermissionSubject) -> Rule | None:
    return select_rule(
        (rule for rule in rules if rule.effect is RuleEffect.DENY),
        subject.tool_name,
        subject.target,
    )


def _rule_decision(rule: Rule, layer: str) -> Decision:
    kind = DecisionKind.ALLOW if rule.effect is RuleEffect.ALLOW else DecisionKind.DENY
    return Decision(kind, f"Matched {rule.source.value} {rule.effect.value} rule.", layer, rule)
