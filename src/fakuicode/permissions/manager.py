"""Stateful permission orchestration around the pure decision engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Event, RLock
from typing import Protocol
from uuid import uuid4

from fakuicode.errors import PermissionPersistenceError
from fakuicode.permissions.config import PermissionConfigRepository, PermissionConfigSnapshot
from fakuicode.permissions.engine import PermissionEngine
from fakuicode.permissions.models import (
    ApprovalChoice,
    Decision,
    DecisionKind,
    PermissionMode,
    PermissionRequest,
    PermissionSubject,
    Rule,
    RuleEffect,
    RuleSource,
)
from fakuicode.permissions.rules import escape_exact_target, parse_rule
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.tools.base import PreparedToolCall


class ApprovalHandler(Protocol):
    def request(
        self, request: PermissionRequest, *, cancel_event: Event | None = None
    ) -> ApprovalChoice: ...


class RejectingApprovalHandler:
    """Fail closed when no interactive approval channel exists."""

    def request(
        self, request: PermissionRequest, *, cancel_event: Event | None = None
    ) -> ApprovalChoice:
        del request, cancel_event
        return ApprovalChoice.DENY


@dataclass
class _PendingApproval:
    request: PermissionRequest
    completed: Event
    choice: ApprovalChoice | None = None


class ApprovalBroker:
    """Bridge synchronous tool workers to one main-thread approval dialog."""

    def __init__(self, *, poll_seconds: float = 0.05) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive.")
        self._poll_seconds = poll_seconds
        self._queue: deque[_PendingApproval] = deque()
        self._active: _PendingApproval | None = None
        self._closed = False
        self._lock = RLock()

    def request(
        self, request: PermissionRequest, *, cancel_event: Event | None = None
    ) -> ApprovalChoice:
        pending = _PendingApproval(request, Event())
        with self._lock:
            if self._closed:
                return ApprovalChoice.DENY
            self._queue.append(pending)
        while not pending.completed.wait(self._poll_seconds):
            if cancel_event is not None and cancel_event.is_set():
                self._complete(pending, ApprovalChoice.DENY)
                break
        return pending.choice if pending.choice is not None else ApprovalChoice.DENY

    def next_request(self) -> PermissionRequest | None:
        with self._lock:
            if self._closed or self._active is not None:
                return None
            while self._queue:
                candidate = self._queue.popleft()
                if candidate.completed.is_set():
                    continue
                self._active = candidate
                return candidate.request
        return None

    def resolve(self, request_id: str, choice: ApprovalChoice) -> bool:
        with self._lock:
            pending = self._active
            if pending is None or pending.request.request_id != request_id or pending.completed.is_set():
                return False
            self._active = None
            pending.choice = ApprovalChoice(choice)
            pending.completed.set()
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = ([self._active] if self._active is not None else []) + list(self._queue)
            self._active = None
            self._queue.clear()
            for item in pending:
                item.choice = ApprovalChoice.DENY
                item.completed.set()

    def _complete(self, pending: _PendingApproval, choice: ApprovalChoice) -> None:
        with self._lock:
            if pending.completed.is_set():
                return
            if self._active is pending:
                self._active = None
            else:
                try:
                    self._queue.remove(pending)
                except ValueError:
                    pass
            pending.choice = choice
            pending.completed.set()


class PermissionManager:
    def __init__(
        self,
        snapshot: PermissionConfigSnapshot,
        command_guard: DangerousCommandGuard,
        *,
        approval_handler: ApprovalHandler | None = None,
        repository: PermissionConfigRepository | None = None,
        session_rules: tuple[Rule, ...] = (),
        request_source: str | None = None,
        owns_approval_handler: bool = True,
        parent_manager: PermissionManager | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._guard = command_guard
        self._approval_handler = approval_handler or RejectingApprovalHandler()
        self._repository = repository
        self._mode = snapshot.mode
        self._session_rules = list(session_rules)
        self._session_capabilities: set[str] = set()
        self._rejected_targets: set[tuple[str, str]] = set()
        self._request_source = request_source
        self._owns_approval_handler = owns_approval_handler
        self._parent_manager = parent_manager
        self._lock = RLock()

    @property
    def snapshot(self) -> PermissionConfigSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def mode(self) -> PermissionMode:
        with self._lock:
            return self._mode

    @property
    def session_rules(self) -> tuple[Rule, ...]:
        with self._lock:
            return tuple(self._session_rules)

    @property
    def session_capabilities(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._session_capabilities))

    def grant_session_capability(self, capability: str) -> None:
        normalized = _normalize_session_capability(capability)
        with self._lock:
            self._session_capabilities.add(normalized)

    def begin_request(self) -> None:
        with self._lock:
            self._rejected_targets.clear()

    def set_mode(self, mode: PermissionMode) -> None:
        with self._lock:
            if self._snapshot.locked:
                raise ValueError("Permission configuration is locked in strict mode.")
            self._mode = PermissionMode(mode)

    def spawn_child(
        self,
        *,
        mode: PermissionMode | None = None,
        approval_handler: ApprovalHandler | None = None,
        request_source: str | None = None,
        command_guard: DangerousCommandGuard | None = None,
    ) -> PermissionManager:
        """Copy the parent's permission ledger without sharing mutable decisions."""

        with self._lock:
            requested = self._mode if mode is None else PermissionMode(mode)
            effective_mode = _narrower_mode(self._mode, requested)
            child = PermissionManager(
                self._snapshot,
                command_guard or self._guard,
                approval_handler=approval_handler,
                repository=self._repository,
                session_rules=tuple(self._session_rules),
                request_source=request_source,
                owns_approval_handler=False,
                parent_manager=self,
            )
            child._mode = effective_mode
            return child

    def authorize(
        self,
        prepared: PreparedToolCall,
        *,
        read_only_task: bool = False,
        cancel_event: Event | None = None,
    ) -> Decision:
        subject = PermissionSubject(prepared.name, prepared.target, prepared.read_only)
        key = (prepared.name, prepared.target)
        parent = self._parent_manager
        if parent is not None:
            inherited = parent._explicit_ledger_decision(subject)
            if inherited is not None:
                return inherited
        with self._lock:
            preauthorized = (
                prepared.permission_capability is not None
                and self._has_session_capability(prepared.permission_capability)
            )
            decision = self._decide(
                subject,
                read_only_task=read_only_task,
                preauthorized=preauthorized,
            )
            if decision.kind is not DecisionKind.ASK:
                return decision
            if key in self._rejected_targets:
                return Decision(
                    DecisionKind.DENY,
                    "This exact action was already rejected in the current request.",
                    "rejection_cache",
                )
            exact_rule = f"{prepared.name}({escape_exact_target(prepared.target)})"
            request = PermissionRequest(
                str(uuid4()),
                prepared.id,
                prepared.name,
                prepared.target,
                decision.reason,
                exact_rule,
                prepared.permission_scope,
                self._request_source,
            )

        choice = self._approval_handler.request(request, cancel_event=cancel_event)
        with self._lock:
            if choice is ApprovalChoice.DENY:
                self._rejected_targets.add(key)
                return Decision(DecisionKind.DENY, "The user denied this action.", "user_confirmation")
            if choice is ApprovalChoice.ONCE:
                return Decision(DecisionKind.ALLOW, "The user allowed this call once.", "user_confirmation")
            if choice is ApprovalChoice.SESSION:
                self._add_session_rule(exact_rule)
                return Decision(
                    DecisionKind.ALLOW,
                    "The user allowed this tool for the current session."
                    if prepared.permission_scope.value == "tool"
                    else "The user allowed this exact target for the current session.",
                    "user_confirmation",
                )
            if choice is ApprovalChoice.PERMANENT:
                return self._save_permanent(exact_rule)
            self._rejected_targets.add(key)
            return Decision(DecisionKind.DENY, "The approval response was invalid.", "user_confirmation")

    def set_project_trusted(self, trusted: bool) -> PermissionConfigSnapshot:
        with self._lock:
            if self._repository is None:
                raise PermissionPersistenceError("Permission configuration storage is unavailable.")
            updated = self._repository.set_project_trusted(trusted)
            self._snapshot = replace(
                self._snapshot,
                project_trusted=updated.project_trusted,
                warnings=updated.warnings,
            )
            return self._snapshot

    def close(self) -> None:
        if self._owns_approval_handler:
            closer = getattr(self._approval_handler, "close", None)
            if callable(closer):
                closer()
        with self._lock:
            self._session_rules.clear()
            self._session_capabilities.clear()
            self._rejected_targets.clear()

    def _decide(
        self,
        subject: PermissionSubject,
        *,
        read_only_task: bool,
        preauthorized: bool,
    ) -> Decision:
        return PermissionEngine(self._snapshot, self._guard).decide(
            subject,
            session_rules=self._session_rules,
            mode=self._mode,
            read_only_task=read_only_task,
            preauthorized=preauthorized,
        )

    def _has_session_capability(self, capability: str) -> bool:
        with self._lock:
            if capability in self._session_capabilities:
                return True
            parent = self._parent_manager
        return parent is not None and parent._has_session_capability(capability)

    def _explicit_ledger_decision(
        self,
        subject: PermissionSubject,
    ) -> Decision | None:
        """Expose current explicit parent rules without inheriting its mode fallback."""

        with self._lock:
            decision = PermissionEngine(self._snapshot, self._guard).decide(
                subject,
                session_rules=tuple(self._session_rules),
                mode=PermissionMode.STRICT,
            )
        if decision.kind is DecisionKind.ALLOW and decision.rule is not None:
            return Decision(
                DecisionKind.ALLOW,
                decision.reason,
                "parent_ledger",
                decision.rule,
            )
        return None

    def _add_session_rule(self, expression: str) -> None:
        rule = parse_rule(expression, RuleEffect.ALLOW, RuleSource.SESSION)
        if not any(existing.expression == rule.expression for existing in self._session_rules):
            self._session_rules.append(rule)

    def _save_permanent(self, expression: str) -> Decision:
        if self._repository is None:
            return Decision(
                DecisionKind.DENY,
                "The permanent permission could not be saved because storage is unavailable.",
                "permission_storage",
            )
        try:
            updated = self._repository.save_project_local_allow(self._snapshot, expression)
        except PermissionPersistenceError:
            return Decision(
                DecisionKind.DENY,
                "The permanent permission could not be saved; the action was not executed.",
                "permission_storage",
            )
        self._snapshot = updated
        return Decision(
            DecisionKind.ALLOW,
            "The user permanently allowed this exact target.",
            "user_confirmation",
        )


def _narrower_mode(parent: PermissionMode, requested: PermissionMode) -> PermissionMode:
    rank = {
        PermissionMode.STRICT: 0,
        PermissionMode.DEFAULT: 1,
        PermissionMode.TRUSTED: 2,
    }
    return parent if rank[parent] <= rank[requested] else requested


def _normalize_session_capability(capability: str) -> str:
    if (
        not isinstance(capability, str)
        or not capability
        or capability != capability.strip()
        or not capability.isprintable()
    ):
        raise ValueError("Session capability must be one non-empty printable line.")
    return capability
