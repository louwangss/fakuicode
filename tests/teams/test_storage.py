from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from fakuicode.teams.models import (
    ActorContext,
    BackendType,
    MemberStatus,
    MessageType,
    TaskStatus,
    TeamMember,
    TeamMessage,
    TeamRecord,
    TeamTask,
)
from fakuicode.teams.storage import (
    DuplicateTeamError,
    TeamStore,
)


def _create_store(tmp_path: Path) -> tuple[TeamStore, str, TeamMember, TeamMember]:
    store = TeamStore(tmp_path / "teams")
    team = store.create_team(
        name="refactor-auth",
        lead_conversation_id="lead-conversation",
        repository_fingerprint="repo-1",
        target_branch="feature/example",
        target_sha="a" * 40,
    )
    lead = TeamMember.create(
        name="lead",
        role="负责人",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="lead-conversation",
        member_id=team.lead_member_id,
    )
    alice = TeamMember.create(
        name="alice",
        role="实现",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=True,
        conversation_id="alice-conversation",
    )
    store.add_member(team.team_id, lead)
    store.add_member(team.team_id, alice)
    return store, team.team_id, lead, alice


def test_create_team_rejects_case_insensitive_collision(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.create_team(
        name="alpha",
        lead_conversation_id="lead",
        repository_fingerprint="repo",
        target_branch="main",
        target_sha="a" * 40,
    )

    with pytest.raises(DuplicateTeamError):
        store.create_team(
            name="ALPHA",
            lead_conversation_id="lead",
            repository_fingerprint="repo",
            target_branch="main",
            target_sha="a" * 40,
        )


def test_registry_rejects_duplicate_member_name(tmp_path: Path) -> None:
    store, team_id, _, alice = _create_store(tmp_path)

    duplicate = TeamMember.create(
        name="alice",
        role="另一个成员",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="different",
    )

    with pytest.raises(ValueError):
        store.add_member(team_id, duplicate)
    assert store.resolve_member(team_id, "alice") == alice.member_id


def test_mailbox_injects_sender_and_preserves_concurrent_messages(tmp_path: Path) -> None:
    store, team_id, lead, alice = _create_store(tmp_path)
    actor = ActorContext(team_id=team_id, member_id=lead.member_id, member_name=lead.name)

    def send(index: int) -> str:
        message = store.send_message(
            actor,
            to=alice.name,
            body=f"message-{index}",
            summary=f"summary-{index}",
            message_type=MessageType.TEXT,
        )
        return message.message_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = set(executor.map(send, range(20)))

    messages = store.list_messages(team_id, alice.member_id, unread_only=True)

    assert len(messages) == 20
    assert {message.message_id for message in messages} == ids
    assert {message.sender_id for message in messages} == {lead.member_id}
    assert all(message.read is False for message in messages)


def test_mailbox_read_receipt_is_idempotent(tmp_path: Path) -> None:
    store, team_id, lead, alice = _create_store(tmp_path)
    actor = ActorContext(team_id=team_id, member_id=lead.member_id, member_name=lead.name)
    message = store.send_message(
        actor,
        to=alice.member_id,
        body="开始工作",
        summary="新任务",
    )

    store.mark_messages_read(team_id, alice.member_id, (message.message_id,))
    store.mark_messages_read(team_id, alice.member_id, (message.message_id,))

    listed = store.list_messages(team_id, alice.member_id)
    assert listed[0].read is True
    assert store.list_messages(team_id, alice.member_id, unread_only=True) == ()


def test_task_graph_rejects_cycles_and_claims_atomically(tmp_path: Path) -> None:
    store, team_id, _, alice = _create_store(tmp_path)
    first = store.create_task(
        team_id,
        TeamTask.create(title="first", description="", created_by="lead"),
    )
    second = store.create_task(
        team_id,
        TeamTask.create(
            title="second",
            description="",
            created_by="lead",
            blocked_by=(first.task_id,),
        ),
    )

    with pytest.raises(ValueError):
        store.update_task_dependencies(team_id, first.task_id, (second.task_id,))

    with pytest.raises(ValueError):
        store.claim_task(team_id, second.task_id, alice.member_id)

    claimed = store.claim_task(team_id, first.task_id, alice.member_id)
    assert claimed.assignee_id == alice.member_id


def test_legacy_json_is_imported_once_and_sqlite_becomes_authoritative(tmp_path: Path) -> None:
    root = tmp_path / "teams"
    team = TeamRecord.create(
        name="legacy-team",
        lead_conversation_id="lead-conversation",
        repository_fingerprint="repo-1",
        target_branch="main",
        target_sha="a" * 40,
    )
    lead = TeamMember.create(
        name="lead",
        role="lead",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="lead-conversation",
        member_id=team.lead_member_id,
    )
    task = TeamTask.create(title="legacy task", description="", created_by=lead.member_id)
    message = TeamMessage(
        message_id=str(uuid4()),
        message_type=MessageType.TEXT,
        sender_id=lead.member_id,
        sender_name=lead.name,
        recipient_id=lead.member_id,
        recipient_name=lead.name,
        body="legacy message",
        summary="legacy",
        created_at=team.created_at,
    )
    team_dir = root / team.name
    for child in ("members", "tasks", "mailboxes", "locks"):
        (team_dir / child).mkdir(parents=True, exist_ok=True)
    (team_dir / "config.json").write_text(
        json.dumps(team.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    member_path = team_dir / "members" / f"{lead.member_id}.json"
    member_path.write_text(json.dumps(lead.to_dict(), ensure_ascii=False), encoding="utf-8")
    (team_dir / "tasks" / f"{task.task_id}.json").write_text(
        json.dumps(task.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    mailbox_path = team_dir / "mailboxes" / f"{lead.member_id}.jsonl"
    mailbox_path.write_text(
        "\n".join(
            (
                json.dumps({"event": "message", "message": message.to_dict()}),
                json.dumps({"event": "read", "message_ids": [message.message_id]}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_bytes = {
        path: path.read_bytes()
        for path in (team_dir / "config.json", member_path, mailbox_path)
    }

    store = TeamStore(root)

    assert store.database_path == root / "teams.sqlite3"
    assert store.get_team(team.team_id) == team
    assert store.list_members(team.team_id) == (lead,)
    assert store.list_tasks(team.team_id) == (task,)
    assert store.list_messages(team.team_id, lead.member_id) == (
        TeamMessage.from_dict(message.to_dict(), read=True),
    )
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {path: path.read_bytes() for path in legacy_bytes} == legacy_bytes

    changed = TeamMember.from_dict({**lead.to_dict(), "role": "tampered legacy"})
    member_path.write_text(json.dumps(changed.to_dict()), encoding="utf-8")
    reopened = TeamStore(root)

    assert reopened.list_members(team.team_id) == (lead,)
    assert len(reopened.list_messages(team.team_id, lead.member_id)) == 1


def test_workflow_transition_rolls_back_all_records_when_message_insert_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, team_id, lead, alice = _create_store(tmp_path)
    task = store.create_task(
        team_id,
        TeamTask.create(title="transaction", description="", created_by=lead.member_id),
    )
    claimed = store.claim_task(team_id, task.task_id, alice.member_id)
    request_id = str(uuid4())
    planned = claimed.revise(
        status=TaskStatus.PLANNING,
        plan_request_id=request_id,
        plan_revision=1,
    )
    waiting = TeamMember.from_dict(
        {
            **alice.to_dict(),
            "status": MemberStatus.WAITING_APPROVAL.value,
            "current_task_id": task.task_id,
        }
    )
    actor = ActorContext(team_id, alice.member_id, alice.name)

    def fail_insert(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected message failure")

    monkeypatch.setattr(store, "_insert_message", fail_insert)

    with pytest.raises(RuntimeError, match="injected message failure"):
        store.commit_workflow_transition(
            actor,
            task=planned,
            member=waiting,
            to=lead.member_id,
            body="plan",
            summary="plan request",
            message_type=MessageType.PLAN_REQUEST,
            correlation_id=request_id,
        )

    assert store.get_task(team_id, task.task_id) == claimed
    assert store.resolve_member(team_id, alice.name) == alice.member_id
    persisted_alice = next(
        member for member in store.list_members(team_id) if member.member_id == alice.member_id
    )
    assert persisted_alice == alice
    assert store.list_messages(team_id, lead.member_id) == ()


def test_workflow_transition_serializes_conflicting_task_revisions(tmp_path: Path) -> None:
    store, team_id, lead, alice = _create_store(tmp_path)
    task = store.create_task(
        team_id,
        TeamTask.create(title="concurrent", description="", created_by=lead.member_id),
    )
    claimed = store.claim_task(team_id, task.task_id, alice.member_id)
    waiting = TeamMember.from_dict(
        {
            **alice.to_dict(),
            "status": MemberStatus.WAITING_APPROVAL.value,
            "current_task_id": task.task_id,
        }
    )
    actor = ActorContext(team_id, alice.member_id, alice.name)
    requests = (str(uuid4()), str(uuid4()))

    def submit(request_id: str) -> str:
        candidate = claimed.revise(
            status=TaskStatus.PLANNING,
            plan_request_id=request_id,
            plan_revision=1,
        )
        return store.commit_workflow_transition(
            actor,
            task=candidate,
            member=waiting,
            to=lead.member_id,
            body=f"plan {request_id}",
            summary="plan request",
            message_type=MessageType.PLAN_REQUEST,
            correlation_id=request_id,
        ).message_id

    successes: list[str] = []
    failures: list[Exception] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit, request_id) for request_id in requests]
        for future in futures:
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append(error)

    persisted = store.get_task(team_id, task.task_id)
    messages = store.list_messages(team_id, lead.member_id)
    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], ValueError)
    assert persisted.revision == claimed.revision + 1
    assert persisted.plan_request_id in requests
    assert len(messages) == 1
    assert messages[0].correlation_id == persisted.plan_request_id
