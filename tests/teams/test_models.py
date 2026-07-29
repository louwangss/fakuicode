from __future__ import annotations

from uuid import UUID

import pytest

from fakuicode.teams.models import (
    BackendType,
    MemberStatus,
    TaskStatus,
    TeamMember,
    TeamTask,
    normalize_team_name,
)


def test_normalize_team_name_rejects_ambiguous_or_unsafe_names() -> None:
    assert normalize_team_name("refactor-auth") == "refactor-auth"

    for value in ("", "Refactor Auth", "../team", ".hidden", "a" * 33):
        with pytest.raises(ValueError):
            normalize_team_name(value)


def test_team_member_round_trips_without_serializing_secrets() -> None:
    member = TeamMember.create(
        name="alice",
        role="负责认证模块",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=True,
        conversation_id="conversation-1",
    )

    restored = TeamMember.from_dict(member.to_dict())

    assert restored == member
    assert UUID(member.member_id)
    assert restored.status is MemberStatus.IDLE
    assert "api_key" not in member.to_dict()


def test_task_dependencies_are_one_way_and_reverse_edges_are_derived() -> None:
    first = TeamTask.create(title="定义接口", description="", created_by="lead")
    second = TeamTask.create(
        title="实现接口",
        description="",
        created_by="lead",
        blocked_by=(first.task_id,),
    )

    assert second.status is TaskStatus.PENDING
    assert second.blocked_by == (first.task_id,)
    assert "blocks" not in second.to_dict()
