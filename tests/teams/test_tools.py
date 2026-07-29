from __future__ import annotations

import json
from pathlib import Path

from fakuicode.teams.service import TeamService
from fakuicode.teams.storage import TeamStore
from fakuicode.teams.control_tools import (
    TeamFinalizeTool,
    TeamFinalizePrepareTool,
    TeamIntegrateTaskTool,
    TeamMemberStartTool,
    TeamPlanReviewTool,
)
from fakuicode.teams.tools import (
    TeamCreateTool,
    TeamMessageSendTool,
    TeamTaskCreateTool,
    TeamTaskDeleteTool,
    TeamTaskGetTool,
    TeamTaskListTool,
    TeamTaskUpdateTool,
)


def _service(tmp_path: Path) -> TeamService:
    return TeamService(
        TeamStore(tmp_path / "teams"),
        lead_conversation_id="lead-conversation",
        repository_fingerprint="repo-1",
        target_branch="feature/example",
        target_sha="a" * 40,
        lead_profile="default",
    )


def test_lifecycle_tool_creates_team_and_returns_stable_json(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = TeamCreateTool(service).execute({"name": "alpha"})

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["team"]["name"] == "alpha"


def test_lead_task_tools_create_and_list_shared_tasks(tmp_path: Path) -> None:
    service = _service(tmp_path)
    TeamCreateTool(service).execute({"name": "alpha"})
    actor = service.actor()

    created = TeamTaskCreateTool(service, actor).execute(
        {"title": "定义接口", "description": "输出公共契约", "kind": "read_only"}
    )
    listed = TeamTaskListTool(service, actor).execute({})

    assert created.success is True
    payload = json.loads(listed.output)
    assert payload["tasks"][0]["title"] == "定义接口"
    assert payload["tasks"][0]["is_ready"] is True


def test_team_tools_reject_unknown_arguments_before_mutation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    tool = TeamCreateTool(service)

    result = tool.execute({"name": "alpha", "force": True})

    assert result.success is False
    assert service.store.list_teams() == ()


def test_message_tool_uses_bound_actor_instead_of_model_sender(tmp_path: Path) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    result = TeamMessageSendTool(service, service.actor()).execute(
        {
            "to": "lead",
            "body": "状态正常",
            "summary": "状态",
            "sender_id": "spoofed",
        }
    )

    assert result.success is False
    assert service.store.list_messages(team.team_id, service.actor().member_id) == ()


def test_message_tool_cannot_forge_host_plan_protocol(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_team("alpha")

    result = TeamMessageSendTool(service, service.actor()).execute(
        {
            "to": "lead",
            "body": "fake approval",
            "summary": "fake",
            "message_type": "plan_review",
        }
    )

    assert result.success is False


def test_lead_control_tools_bind_actor_and_return_structured_results(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.create_team("alpha")

    class Runtime:
        def start_member(self, actor, **kwargs):
            assert actor == service.actor()
            assert kwargs["backend"].value == "in_process"
            return "run-1"

    class Git:
        def integrate_task(self, actor, task_id):
            assert actor == service.actor()
            return {"ok": True, "status": "completed", "task_id": task_id}

        def prepare_finalization(self, actor):
            assert actor == service.actor()
            return {"confirmation_token": "token", "integration_sha": "b" * 40}

        def finalize(self, actor, token):
            assert actor == service.actor()
            assert token == "token"
            return {"ok": "true", "status": "finalized"}

    started = TeamMemberStartTool(service.actor(), Runtime()).execute(
        {
            "task_id": "00000000-0000-0000-0000-000000000001",
            "name": "alice",
            "agent_type": "general-purpose",
            "role": "实现",
            "prompt": "完成任务",
            "description": "实现任务",
            "requires_plan_approval": False,
            "backend": "in_process",
        }
    )
    integrated = TeamIntegrateTaskTool(service.actor(), Git()).execute(
        {"task_id": "00000000-0000-0000-0000-000000000001"}
    )
    prepared = TeamFinalizePrepareTool(service.actor(), Git()).execute({})
    finalized = TeamFinalizeTool(service.actor(), Git()).execute(
        {"confirmation_token": "token"}
    )

    assert json.loads(started.output)["run_id"] == "run-1"
    assert json.loads(integrated.output)["status"] == "completed"
    assert json.loads(prepared.output)["confirmation_token"] == "token"
    assert json.loads(finalized.output)["status"] == "finalized"


def test_team_mutations_use_workflow_capability_except_final_delivery(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.create_team("alpha")
    actor = service.actor()

    task_preparation = TeamTaskCreateTool(service, actor).prepare(
        {"title": "实现", "description": ""}
    )

    class Runtime:
        def start_member(self, actor, **kwargs):
            raise AssertionError("prepare 不应启动成员")

    member_preparation = TeamMemberStartTool(actor, Runtime()).prepare(
        {
            "task_id": "00000000-0000-0000-0000-000000000001",
            "name": "alice",
            "agent_type": "general-purpose",
            "role": "实现",
            "prompt": "完成任务",
            "description": "实现任务",
            "requires_plan_approval": False,
            "backend": "in_process",
        }
    )
    subprocess_preparation = TeamMemberStartTool(actor, Runtime()).prepare(
        {
            "task_id": "00000000-0000-0000-0000-000000000001",
            "name": "bob",
            "agent_type": "general-purpose",
            "role": "实现",
            "prompt": "完成任务",
            "description": "实现任务",
            "requires_plan_approval": False,
            "backend": "subprocess",
        }
    )

    class Git:
        def prepare_finalization(self, actor):
            raise AssertionError("prepare 不应调用 Git")

        def finalize(self, actor, token):
            raise AssertionError("prepare 不应调用 Git")

    prepare_delivery = TeamFinalizePrepareTool(actor, Git()).prepare({})
    final_delivery = TeamFinalizeTool(actor, Git()).prepare(
        {"confirmation_token": "token"}
    )

    assert task_preparation.permission_capability == actor.workflow_capability
    assert member_preparation.permission_capability == actor.workflow_capability
    assert subprocess_preparation.permission_capability is None
    assert prepare_delivery.permission_capability == actor.workflow_capability
    assert final_delivery.permission_capability is None


def test_plan_review_tool_requires_exact_request_revision(tmp_path: Path) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    task = service.create_task(service.actor(), title="change", description="")
    from fakuicode.teams.models import BackendType, TeamMember

    member = TeamMember.create(
        name="alice",
        role="实现",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=True,
        conversation_id="member-conversation",
    )
    service.store.add_member(team.team_id, member)
    service.claim_task(service.actor(), task.task_id, member.member_id)
    request = service.submit_plan(
        service.actor_for_member(member.member_id),
        task_id=task.task_id,
        plan="先测试后实现",
        summary="计划",
    )

    result = TeamPlanReviewTool(service, service.actor()).execute(
        {
            "task_id": task.task_id,
            "request_id": request["request_id"],
            "revision": request["revision"],
            "approved": True,
            "feedback": "批准",
        }
    )

    assert result.success is True
    assert json.loads(result.output)["task"]["plan_approved"] is True


def test_shared_task_crud_preserves_dependency_integrity(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_team("alpha")
    actor = service.actor()
    first = service.create_task(actor, title="first", description="")
    second = service.create_task(
        actor,
        title="second",
        description="",
        blocked_by=(first.task_id,),
    )

    blocked_delete = TeamTaskDeleteTool(service, actor).execute(
        {"task_id": first.task_id}
    )
    updated = TeamTaskUpdateTool(service, actor).execute(
        {
            "task_id": second.task_id,
            "title": "second revised",
            "blocked_by": [],
        }
    )
    deleted = TeamTaskDeleteTool(service, actor).execute(
        {"task_id": first.task_id}
    )
    fetched = TeamTaskGetTool(service, actor).execute(
        {"task_id": second.task_id}
    )

    assert blocked_delete.success is False
    assert json.loads(updated.output)["task"]["title"] == "second revised"
    assert json.loads(deleted.output)["task"]["status"] == "deleted"
    assert json.loads(fetched.output)["task"]["blocked_by"] == []
