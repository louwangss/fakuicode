from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.teams.models import (
    BackendType,
    MemberStatus,
    TaskStatus,
    TeamMember,
)
from fakuicode.teams.service import TeamService
from fakuicode.teams.storage import TeamStore


def _service(tmp_path: Path) -> TeamService:
    return TeamService(
        TeamStore(tmp_path / "teams"),
        lead_conversation_id="lead-conversation",
        repository_fingerprint="repo-1",
        target_branch="feature/example",
        target_sha="a" * 40,
        lead_profile="default",
    )


def test_create_team_registers_fixed_lead_and_attaches_it(tmp_path: Path) -> None:
    service = _service(tmp_path)

    team = service.create_team("alpha")

    assert service.active_team_id == team.team_id
    lead = service.actor()
    assert lead.team_id == team.team_id
    assert lead.member_name == "lead"


def test_attach_rejects_different_repository_or_lead(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_team("alpha")

    wrong_repo = TeamService(
        service.store,
        lead_conversation_id="lead-conversation",
        repository_fingerprint="repo-2",
        target_branch="feature/example",
        target_sha="a" * 40,
        lead_profile="default",
    )
    with pytest.raises(ValueError):
        wrong_repo.attach_team("alpha")

    wrong_lead = TeamService(
        service.store,
        lead_conversation_id="another-conversation",
        repository_fingerprint="repo-1",
        target_branch="feature/example",
        target_sha="a" * 40,
        lead_profile="default",
    )
    with pytest.raises(ValueError):
        wrong_lead.attach_team("alpha")


def test_team_members_can_message_directly_but_cannot_spoof_actor(tmp_path: Path) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    alice = TeamMember.create(
        name="alice",
        role="实现",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="alice-conversation",
    )
    bob = TeamMember.create(
        name="bob",
        role="复核",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="bob-conversation",
    )
    service.store.add_member(team.team_id, alice)
    service.store.add_member(team.team_id, bob)

    service.send_message(
        service.actor_for_member(alice.member_id),
        to="bob",
        body="接口已经完成",
        summary="接口完成",
    )

    message = service.store.list_messages(team.team_id, bob.member_id)[0]
    assert message.sender_id == alice.member_id
    with pytest.raises(ValueError):
        service.actor_for_member("00000000-0000-0000-0000-000000000000")


def test_plan_review_requires_exact_request_and_revision(tmp_path: Path) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    alice = TeamMember.create(
        name="alice",
        role="实现",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=True,
        conversation_id="alice-conversation",
    )
    service.store.add_member(team.team_id, alice)
    task = service.create_task(
        service.actor(),
        title="实现认证",
        description="",
    )
    service.claim_task(service.actor(), task.task_id, alice.member_id)

    request = service.submit_plan(
        service.actor_for_member(alice.member_id),
        task_id=task.task_id,
        plan="先补测试，再实现认证。",
        summary="认证计划",
    )

    assert request["revision"] == 1
    assert service.store.get_task(team.team_id, task.task_id).status is TaskStatus.PLANNING
    waiting = next(
        member
        for member in service.store.list_members(team.team_id)
        if member.member_id == alice.member_id
    )
    assert waiting.status is MemberStatus.WAITING_APPROVAL

    with pytest.raises(ValueError):
        service.review_plan(
            service.actor(),
            task_id=task.task_id,
            request_id=request["request_id"],
            revision=2,
            approved=True,
            feedback="",
        )

    reviewed = service.review_plan(
        service.actor(),
        task_id=task.task_id,
        request_id=request["request_id"],
        revision=1,
        approved=True,
        feedback="按此执行",
    )

    assert reviewed.status is TaskStatus.WORKING
    running = next(
        member
        for member in service.store.list_members(team.team_id)
        if member.member_id == alice.member_id
    )
    assert running.status is MemberStatus.IDLE

    with pytest.raises(ValueError):
        service.review_plan(
            service.actor(),
            task_id=task.task_id,
            request_id=request["request_id"],
            revision=1,
            approved=True,
            feedback="重复批准",
        )
