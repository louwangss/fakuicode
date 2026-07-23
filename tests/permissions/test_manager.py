from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from fakuicode.errors import PermissionPersistenceError
from fakuicode.permissions.config import PermissionConfigRepository, PermissionConfigSnapshot, PermissionPaths
from fakuicode.permissions.manager import ApprovalBroker, PermissionManager, RejectingApprovalHandler
from fakuicode.permissions.models import (
    ApprovalChoice,
    DecisionKind,
    PermissionMode,
    PermissionScope,
    RuleEffect,
    RuleSource,
)
from fakuicode.permissions.rules import parse_rule
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.tools.base import PreparedToolCall


class RecordingApprovalHandler:
    def __init__(self, *choices: ApprovalChoice) -> None:
        self.choices = list(choices)
        self.requests = []

    def request(self, request, *, cancel_event: Event | None = None) -> ApprovalChoice:
        del cancel_event
        self.requests.append(request)
        return self.choices.pop(0)


def _prepared(tool: str = "write_file", target: str = "src/main.py", *, call_id: str = "call-1"):
    return PreparedToolCall(call_id, tool, {}, target, read_only=tool in {"read_file", "find_files", "search_code"})


def _manager(
    tmp_path: Path,
    handler=None,
    *,
    snapshot: PermissionConfigSnapshot | None = None,
    repository: PermissionConfigRepository | None = None,
) -> PermissionManager:
    return PermissionManager(
        snapshot or PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
        approval_handler=handler,
        repository=repository,
    )


def test_rejecting_handler_fails_closed_without_waiting(tmp_path: Path) -> None:
    manager = _manager(tmp_path, RejectingApprovalHandler())

    decision = manager.authorize(_prepared())

    assert decision.kind is DecisionKind.DENY
    assert decision.layer == "user_confirmation"


def test_once_allows_only_the_current_call(tmp_path: Path) -> None:
    handler = RecordingApprovalHandler(ApprovalChoice.ONCE, ApprovalChoice.DENY)
    manager = _manager(tmp_path, handler)

    first = manager.authorize(_prepared(call_id="call-1"))
    second = manager.authorize(_prepared(call_id="call-2"))

    assert first.kind is DecisionKind.ALLOW
    assert second.kind is DecisionKind.DENY
    assert len(handler.requests) == 2


def test_session_choice_adds_an_exact_session_rule(tmp_path: Path) -> None:
    handler = RecordingApprovalHandler(ApprovalChoice.SESSION, ApprovalChoice.DENY)
    manager = _manager(tmp_path, handler)

    first = manager.authorize(_prepared(target="src/*literal.py", call_id="call-1"))
    same = manager.authorize(_prepared(target="src/*literal.py", call_id="call-2"))
    other = manager.authorize(_prepared(target="src/anything.py", call_id="call-3"))

    assert first.kind is DecisionKind.ALLOW
    assert same.kind is DecisionKind.ALLOW
    assert other.kind is DecisionKind.DENY
    assert len(handler.requests) == 2
    assert manager.session_rules[0].exact is True


def test_mcp_session_permission_covers_whole_tool_without_persisting_arguments(tmp_path: Path) -> None:
    handler = RecordingApprovalHandler(ApprovalChoice.SESSION)
    manager = _manager(tmp_path, handler)
    first = PreparedToolCall(
        "call-1",
        "mcp__docs__lookup",
        {"secret": "first"},
        "__all_arguments__",
        False,
        PermissionScope.TOOL,
    )
    second = PreparedToolCall(
        "call-2",
        "mcp__docs__lookup",
        {"secret": "second"},
        "__all_arguments__",
        False,
        PermissionScope.TOOL,
    )
    assert manager.authorize(first).kind is DecisionKind.ALLOW
    assert manager.authorize(second).kind is DecisionKind.ALLOW
    assert len(handler.requests) == 1
    assert handler.requests[0].scope is PermissionScope.TOOL
    assert "first" not in manager.session_rules[0].expression
    assert "second" not in manager.session_rules[0].expression


def test_rejection_cache_suppresses_same_target_until_next_request(tmp_path: Path) -> None:
    handler = RecordingApprovalHandler(ApprovalChoice.DENY, ApprovalChoice.ONCE, ApprovalChoice.ONCE)
    manager = _manager(tmp_path, handler)

    denied = manager.authorize(_prepared(call_id="call-1"))
    cached = manager.authorize(_prepared(call_id="call-2"))
    changed = manager.authorize(_prepared(target="src/other.py", call_id="call-3"))
    manager.begin_request()
    retried = manager.authorize(_prepared(call_id="call-4"))

    assert denied.kind is DecisionKind.DENY
    assert cached.kind is DecisionKind.DENY
    assert cached.layer == "rejection_cache"
    assert changed.kind is DecisionKind.ALLOW
    assert retried.kind is DecisionKind.ALLOW
    assert len(handler.requests) == 3


def test_permanent_choice_persists_before_allowing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = PermissionConfigRepository(
        PermissionPaths.for_workspace(workspace, home=tmp_path / "home"), workspace
    )
    handler = RecordingApprovalHandler(ApprovalChoice.PERMANENT)
    manager = _manager(workspace, handler, snapshot=repository.load(), repository=repository)

    decision = manager.authorize(_prepared(target="src/main.py"))
    reloaded = repository.load()

    assert decision.kind is DecisionKind.ALLOW
    assert len(reloaded.project_local_rules) == 1
    assert reloaded.project_local_rules[0].matches("src/main.py")


def test_permanent_save_failure_denies_without_memory_grant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = PermissionConfigRepository(
        PermissionPaths.for_workspace(workspace, home=tmp_path / "home"), workspace
    )
    handler = RecordingApprovalHandler(ApprovalChoice.PERMANENT, ApprovalChoice.DENY)
    manager = _manager(workspace, handler, snapshot=repository.load(), repository=repository)

    def fail_save(snapshot, expression):
        del snapshot, expression
        raise PermissionPersistenceError("simulated")

    monkeypatch.setattr(repository, "save_project_local_allow", fail_save)

    failed = manager.authorize(_prepared(call_id="call-1"))
    repeated = manager.authorize(_prepared(call_id="call-2"))

    assert failed.kind is DecisionKind.DENY
    assert "saved" in failed.reason.lower()
    assert repeated.kind is DecisionKind.DENY
    assert manager.session_rules == ()


def test_hard_and_global_denies_never_call_approval_handler(tmp_path: Path) -> None:
    handler = RecordingApprovalHandler(ApprovalChoice.ONCE)
    global_deny = parse_rule("write_file(*)", RuleEffect.DENY, RuleSource.USER)
    manager = _manager(
        tmp_path,
        handler,
        snapshot=PermissionConfigSnapshot(
            mode=PermissionMode.TRUSTED,
            user_rules=(global_deny,),
        ),
    )

    command = manager.authorize(_prepared("run_command", "bash -lc pwd"))
    write = manager.authorize(_prepared("write_file", "src/main.py"))

    assert command.kind is DecisionKind.DENY
    assert write.kind is DecisionKind.DENY
    assert handler.requests == []


def test_read_only_task_boundary_cannot_be_approved(tmp_path: Path) -> None:
    allow = parse_rule("write_file(*)", RuleEffect.ALLOW, RuleSource.USER)
    handler = RecordingApprovalHandler(ApprovalChoice.ONCE)
    manager = _manager(tmp_path, handler, snapshot=PermissionConfigSnapshot(user_rules=(allow,)))

    decision = manager.authorize(_prepared(), read_only_task=True)

    assert decision.kind is DecisionKind.DENY
    assert decision.layer == "task_mode"
    assert handler.requests == []


def test_locked_snapshot_cannot_switch_runtime_mode(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        snapshot=PermissionConfigSnapshot(mode=PermissionMode.STRICT, locked=True),
    )

    with pytest.raises(ValueError, match="locked"):
        manager.set_mode(PermissionMode.TRUSTED)

    assert manager.mode is PermissionMode.STRICT


def test_child_permission_state_copies_parent_rules_without_sharing_mutations(tmp_path: Path) -> None:
    parent_handler = RecordingApprovalHandler(ApprovalChoice.SESSION)
    parent = _manager(tmp_path, parent_handler)
    assert parent.authorize(_prepared(target="src/parent.py")).kind is DecisionKind.ALLOW
    child_handler = RecordingApprovalHandler(ApprovalChoice.SESSION)

    child = parent.spawn_child(
        approval_handler=child_handler,
        request_source="reviewer",
    )

    inherited = child.authorize(_prepared(target="src/parent.py", call_id="child-1"))
    added = child.authorize(_prepared(target="src/child.py", call_id="child-2"))
    assert inherited.kind is DecisionKind.ALLOW
    assert added.kind is DecisionKind.ALLOW
    assert len(child_handler.requests) == 1
    assert child_handler.requests[0].source == "reviewer"
    assert len(child.session_rules) == 2
    assert len(parent.session_rules) == 1


def test_child_observes_parent_rules_approved_after_spawn(tmp_path: Path) -> None:
    parent_handler = RecordingApprovalHandler(ApprovalChoice.SESSION)
    parent = _manager(tmp_path, parent_handler)
    child_handler = RecordingApprovalHandler(ApprovalChoice.DENY)
    child = parent.spawn_child(approval_handler=child_handler)
    prepared = _prepared(target="src/shared.py")

    assert parent.authorize(prepared).kind is DecisionKind.ALLOW
    inherited = child.authorize(
        _prepared(target="src/shared.py", call_id="child-late-rule")
    )

    assert inherited.kind is DecisionKind.ALLOW
    assert inherited.layer == "parent_ledger"
    assert child_handler.requests == []
    assert child.session_rules == ()


def test_child_permission_mode_cannot_elevate_parent_mode(tmp_path: Path) -> None:
    parent = _manager(
        tmp_path,
        snapshot=PermissionConfigSnapshot(mode=PermissionMode.STRICT),
    )

    child = parent.spawn_child(mode=PermissionMode.TRUSTED)
    decision = child.authorize(_prepared())

    assert child.mode is PermissionMode.STRICT
    assert decision.kind is DecisionKind.DENY
    assert decision.layer == "mode"


def test_child_close_does_not_close_shared_approval_handler(tmp_path: Path) -> None:
    class CloseTrackingHandler(RecordingApprovalHandler):
        def __init__(self) -> None:
            super().__init__(ApprovalChoice.DENY)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    handler = CloseTrackingHandler()
    parent = _manager(tmp_path, handler)
    child = parent.spawn_child(approval_handler=handler)

    child.close()

    assert handler.closed is False
    parent.close()
    assert handler.closed is True


def test_trust_change_does_not_hot_reload_other_permission_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = PermissionConfigRepository(
        PermissionPaths.for_workspace(workspace, home=tmp_path / "home"), workspace
    )
    original_rule = parse_rule("write_file(src/main.py)", RuleEffect.DENY, RuleSource.USER)
    original = PermissionConfigSnapshot(user_rules=(original_rule,))
    manager = _manager(workspace, snapshot=original, repository=repository)
    reloaded = PermissionConfigSnapshot(
        mode=PermissionMode.TRUSTED,
        project_trusted=True,
        warnings=("new trust warning",),
    )
    monkeypatch.setattr(repository, "set_project_trusted", lambda trusted: reloaded)

    updated = manager.set_project_trusted(True)

    assert updated.project_trusted is True
    assert updated.user_rules == (original_rule,)
    assert updated.mode is PermissionMode.DEFAULT
    assert manager.mode is PermissionMode.DEFAULT


def _wait_for_broker_request(broker: ApprovalBroker):
    deadline = monotonic() + 1
    while monotonic() < deadline:
        request = broker.next_request()
        if request is not None:
            return request
        sleep(0.01)
    raise AssertionError("approval request did not arrive")


def test_approval_broker_wakes_the_matching_background_request() -> None:
    broker = ApprovalBroker()
    permission_request = _manager_request("request-1")
    choices = []

    worker = Thread(target=lambda: choices.append(broker.request(permission_request)), daemon=True)
    worker.start()
    visible = _wait_for_broker_request(broker)

    assert visible == permission_request
    assert broker.resolve("request-1", ApprovalChoice.SESSION) is True
    worker.join(timeout=1)

    assert choices == [ApprovalChoice.SESSION]


def test_approval_broker_exposes_only_one_request_and_preserves_queue_order() -> None:
    broker = ApprovalBroker()
    choices = {}
    first = Thread(
        target=lambda: choices.__setitem__("first", broker.request(_manager_request("first"))), daemon=True
    )
    first.start()
    assert _wait_for_broker_request(broker).request_id == "first"
    second = Thread(
        target=lambda: choices.__setitem__("second", broker.request(_manager_request("second"))), daemon=True
    )
    second.start()

    assert broker.next_request() is None
    assert broker.resolve("first", ApprovalChoice.ONCE) is True
    assert _wait_for_broker_request(broker).request_id == "second"
    assert broker.resolve("second", ApprovalChoice.DENY) is True
    first.join(timeout=1)
    second.join(timeout=1)

    assert choices == {"first": ApprovalChoice.ONCE, "second": ApprovalChoice.DENY}


def test_approval_broker_cancel_and_close_release_waiters() -> None:
    broker = ApprovalBroker(poll_seconds=0.01)
    cancelled = Event()
    choices = []
    first = Thread(
        target=lambda: choices.append(broker.request(_manager_request("cancelled"), cancel_event=cancelled)),
        daemon=True,
    )
    first.start()
    _wait_for_broker_request(broker)
    cancelled.set()
    first.join(timeout=1)

    second = Thread(target=lambda: choices.append(broker.request(_manager_request("closed"))), daemon=True)
    second.start()
    _wait_for_broker_request(broker)
    broker.close()
    second.join(timeout=1)

    assert choices == [ApprovalChoice.DENY, ApprovalChoice.DENY]
    assert not first.is_alive()
    assert not second.is_alive()
    assert broker.resolve("closed", ApprovalChoice.ONCE) is False


def _manager_request(request_id: str):
    from fakuicode.permissions.models import PermissionRequest

    return PermissionRequest(
        request_id,
        f"call-{request_id}",
        "write_file",
        "src/main.py",
        "confirmation needed",
        "write_file(src/main.py)",
    )
