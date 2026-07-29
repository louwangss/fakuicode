from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

from fakuicode.subagents.models import AgentDefinition, AgentSource
from fakuicode.subagents.runtime import ChildRunResult
from fakuicode.subagents.tasks import TaskManager
from fakuicode.teams.models import (
    BackendType,
    MemberStatus,
    MessageType,
    TaskKind,
    TaskStatus,
)
from fakuicode.teams.runtime import TeamRuntimeManager
from fakuicode.teams.service import TeamService
from fakuicode.teams.storage import TeamStore


@dataclass
class FakeSession:
    id: str
    name: str
    role: str
    profile_name: str
    conversation_id: str
    prompts: list[str]
    execution: dict[str, object]
    outcome: ChildRunResult = ChildRunResult("done", "completed")
    started: Event | None = None
    release: Event | None = None

    def run_to_completion(self, prompt: str, *, event_sink=None) -> ChildRunResult:
        del event_sink
        self.prompts.append(prompt)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(5)
        return self.outcome

    def cancel(self) -> None:
        if self.release is not None:
            self.release.set()

    def touch(self) -> None:
        return None

    def close(self, *, status: str = "completed") -> None:
        del status


class FakeFactory:
    def __init__(
        self,
        *,
        outcome: ChildRunResult = ChildRunResult("done", "completed"),
        block: bool = False,
        on_create=None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.sessions: list[FakeSession] = []
        self.outcome = outcome
        self.block = block
        self.on_create = on_create

    def create_defined(self, definition, **kwargs):
        self.calls.append(dict(kwargs))
        session = FakeSession(
            id=str(kwargs["session_id"]),
            name=str(kwargs["name"]),
            role=definition.name,
            profile_name=str(kwargs.get("profile_override") or "default"),
            conversation_id=str(
                kwargs.get("conversation_id")
                or kwargs.get("create_conversation_id")
                or f"conv-{len(self.calls)}"
            ),
            prompts=[],
            execution={"isolation": "shared"},
            outcome=self.outcome,
            started=Event() if self.block else None,
            release=Event() if self.block else None,
        )
        self.sessions.append(session)
        if self.on_create is not None:
            self.on_create(kwargs)
        return session


class FakeCatalog:
    def resolve(self, name: str) -> AgentDefinition:
        if name != "general-purpose":
            raise KeyError(name)
        return AgentDefinition(
            "general-purpose",
            "通用成员",
            "完成任务",
            AgentSource.BUILTIN,
            Path("general-purpose.md"),
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


def test_in_process_member_becomes_idle_and_can_resume_after_manager_restart(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="调研接口",
        description="",
        kind=TaskKind.READ_ONLY,
    )
    first_factory = FakeFactory()
    first_tasks = TaskManager(max_concurrent=1)
    first_runtime = TeamRuntimeManager(
        service,
        FakeCatalog(),
        first_factory,
        first_tasks,
    )

    run_id = first_runtime.start_member(
        service.actor(),
        task_id=task.task_id,
        name="alice",
        agent_type="general-purpose",
        role="调研",
        prompt="检查接口",
        description="调研接口",
        profile=None,
        requires_plan_approval=False,
    )
    snapshot = first_tasks.wait(run_id, timeout=5)

    assert snapshot is not None and snapshot.status == "completed"
    alice = next(
        member for member in service.store.list_members(team.team_id) if member.name == "alice"
    )
    assert alice.status is MemberStatus.IDLE
    assert service.store.list_messages(
        team.team_id, service.actor().member_id, unread_only=True
    )[0].summary == "alice 已空闲"
    first_tasks.close()

    second_factory = FakeFactory()
    second_tasks = TaskManager(max_concurrent=1)
    second_runtime = TeamRuntimeManager(
        service,
        FakeCatalog(),
        second_factory,
        second_tasks,
    )
    resumed_id = second_runtime.resume_member(
        service.actor(),
        member_name="alice",
        prompt="继续复核",
        description="续派",
    )
    resumed = second_tasks.wait(resumed_id, timeout=5)

    assert resumed is not None and resumed.status == "completed"
    assert second_factory.calls[0]["conversation_id"] == alice.conversation_id
    assert second_factory.calls[0]["session_id"] == alice.member_id
    second_tasks.close()


def test_worktree_task_passes_task_owned_lease_to_child_runtime(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="实现功能",
        description="",
        kind=TaskKind.TASK_WORKTREE,
    )
    sentinel = object()

    class Git:
        def prepare_task(self, actor, task_id):
            assert actor == service.actor()
            assert task_id == task.task_id
            return sentinel

        def task_lease(self, actor, task_id):
            raise AssertionError("新任务不应恢复旧 Worktree")

    factory = FakeFactory()
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(
        service,
        FakeCatalog(),
        factory,
        tasks,
        git_coordinator=Git(),
    )

    run_id = runtime.start_member(
        service.actor(),
        task_id=task.task_id,
        name="alice",
        agent_type="general-purpose",
        role="实现",
        prompt="完成任务",
        description="实现功能",
        profile=None,
        requires_plan_approval=False,
    )
    tasks.wait(run_id, timeout=5)

    assert factory.calls[0]["execution_lease"] is sentinel
    selection = runtime.selection_for_run(run_id)
    assert selection is not None
    assert selection["requested_backend"] == "auto"
    assert selection["selected_backend"] == "in_process"
    tasks.close()


def test_explicit_subprocess_request_fails_without_silent_downgrade(
    tmp_path: Path,
) -> None:
    import pytest

    service = _service(tmp_path)
    service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="调研",
        description="",
        kind=TaskKind.READ_ONLY,
    )
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(service, FakeCatalog(), FakeFactory(), tasks)

    with pytest.raises(ValueError, match="未降级"):
        runtime.start_member(
            service.actor(),
            task_id=task.task_id,
            name="alice",
            agent_type="general-purpose",
            role="调研",
            prompt="检查",
            description="调研",
            profile=None,
            requires_plan_approval=False,
            backend=BackendType.SUBPROCESS,
        )

    assert len(service.store.list_members(service.actor().team_id)) == 1
    tasks.close()


def test_member_start_failure_leaves_member_and_task_in_failed_state(
    tmp_path: Path,
) -> None:
    import pytest

    class FailingFactory:
        def create_defined(self, definition, **kwargs):
            del definition, kwargs
            raise RuntimeError("factory failed")

    service = _service(tmp_path)
    team = service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="调研",
        description="",
        kind=TaskKind.READ_ONLY,
    )
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(service, FakeCatalog(), FailingFactory(), tasks)

    with pytest.raises(RuntimeError, match="factory failed"):
        runtime.start_member(
            service.actor(),
            task_id=task.task_id,
            name="alice",
            agent_type="general-purpose",
            role="调研",
            prompt="检查",
            description="调研",
            profile=None,
            requires_plan_approval=False,
        )

    alice = next(
        member
        for member in service.store.list_members(team.team_id)
        if member.name == "alice"
    )
    assert alice.status is MemberStatus.FAILED
    assert service.store.get_task(team.team_id, task.task_id).status.value == "failed"
    tasks.close()


def test_member_run_failure_keeps_inbox_unread_and_never_sends_idle_notice(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="research",
        description="",
        kind=TaskKind.READ_ONLY,
    )

    def send_before_run(kwargs) -> None:
        service.send_message(
            service.actor(),
            to=str(kwargs["session_id"]),
            body="important evidence",
            summary="evidence",
        )

    factory = FakeFactory(
        outcome=ChildRunResult("", "failed", "provider failed"),
        on_create=send_before_run,
    )
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(service, FakeCatalog(), factory, tasks)

    run_id = runtime.start_member(
        service.actor(),
        task_id=task.task_id,
        name="alice",
        agent_type="general-purpose",
        role="research",
        prompt="inspect",
        description="research",
        profile=None,
        requires_plan_approval=False,
    )
    snapshot = tasks.wait(run_id, timeout=5)

    assert snapshot is not None and snapshot.status == "failed"
    alice = next(
        member for member in service.store.list_members(team.team_id) if member.name == "alice"
    )
    assert alice.status is MemberStatus.FAILED
    assert alice.runtime_id is None
    assert service.store.get_task(team.team_id, task.task_id).status is TaskStatus.FAILED
    unread = service.store.list_messages(
        team.team_id,
        alice.member_id,
        unread_only=True,
    )
    assert [message.summary for message in unread] == ["evidence"]
    lead_messages = service.store.list_messages(
        team.team_id,
        service.actor().member_id,
        unread_only=True,
    )
    assert all(message.message_type is not MessageType.IDLE_NOTICE for message in lead_messages)
    tasks.close()


def test_stop_member_wins_over_a_late_completed_result(tmp_path: Path) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="research",
        description="",
        kind=TaskKind.READ_ONLY,
    )
    factory = FakeFactory(block=True)
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(service, FakeCatalog(), factory, tasks)

    run_id = runtime.start_member(
        service.actor(),
        task_id=task.task_id,
        name="alice",
        agent_type="general-purpose",
        role="research",
        prompt="inspect",
        description="research",
        profile=None,
        requires_plan_approval=False,
    )
    session = factory.sessions[0]
    assert session.started is not None and session.started.wait(5)

    assert runtime.stop_member(service.actor(), "alice") is True
    snapshot = tasks.wait(run_id, timeout=5)

    assert snapshot is not None and snapshot.status == "cancelled"
    alice = next(
        member for member in service.store.list_members(team.team_id) if member.name == "alice"
    )
    assert alice.status is MemberStatus.STOPPED
    assert alice.runtime_id is None
    assert service.store.get_task(team.team_id, task.task_id).status is TaskStatus.CANCELLED
    lead_messages = service.store.list_messages(
        team.team_id,
        service.actor().member_id,
        unread_only=True,
    )
    assert all(message.message_type is not MessageType.IDLE_NOTICE for message in lead_messages)
    tasks.close()


def test_approved_planning_member_is_rebound_to_task_worktree(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    task = service.create_task(
        service.actor(),
        title="实现",
        description="",
        kind=TaskKind.TASK_WORKTREE,
    )
    sentinel = object()

    class Git:
        def prepare_task(self, actor, task_id):
            assert task_id == task.task_id
            return sentinel

        def task_lease(self, actor, task_id):
            raise AssertionError("首次绑定不应恢复 Worktree")

    factory = FakeFactory()
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(
        service,
        FakeCatalog(),
        factory,
        tasks,
        git_coordinator=Git(),
    )
    planning_run = runtime.start_member(
        service.actor(),
        task_id=task.task_id,
        name="alice",
        agent_type="general-purpose",
        role="实现",
        prompt="先提交计划",
        description="规划",
        profile=None,
        requires_plan_approval=True,
    )
    tasks.wait(planning_run, timeout=5)
    alice = next(
        member
        for member in service.store.list_members(team.team_id)
        if member.name == "alice"
    )
    request = service.submit_plan(
        service.actor_for_member(alice.member_id),
        task_id=task.task_id,
        plan="计划",
        summary="计划",
    )
    service.review_plan(
        service.actor(),
        task_id=task.task_id,
        request_id=str(request["request_id"]),
        revision=int(request["revision"]),
        approved=True,
        feedback="批准",
    )

    resumed_run = runtime.resume_member(
        service.actor(),
        member_name="alice",
        prompt="开始实施",
        description="实施",
    )
    tasks.wait(resumed_run, timeout=5)

    assert len(factory.calls) == 2
    assert factory.calls[1]["execution_lease"] is sentinel
    tasks.close()


def test_idle_member_can_receive_new_task_with_same_conversation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    team = service.create_team("alpha")
    first = service.create_task(
        service.actor(),
        title="first",
        description="",
        kind=TaskKind.READ_ONLY,
    )
    second = service.create_task(
        service.actor(),
        title="second",
        description="",
        kind=TaskKind.READ_ONLY,
    )
    factory = FakeFactory()
    tasks = TaskManager(max_concurrent=1)
    runtime = TeamRuntimeManager(service, FakeCatalog(), factory, tasks)
    first_run = runtime.start_member(
        service.actor(),
        task_id=first.task_id,
        name="alice",
        agent_type="general-purpose",
        role="调研",
        prompt="first",
        description="first",
        profile=None,
        requires_plan_approval=False,
    )
    tasks.wait(first_run, timeout=5)
    alice = next(
        member
        for member in service.store.list_members(team.team_id)
        if member.name == "alice"
    )

    second_run = runtime.assign_member(
        service.actor(),
        member_name="alice",
        task_id=second.task_id,
        prompt="second",
        description="second",
    )
    tasks.wait(second_run, timeout=5)

    assert len(factory.calls) == 2
    assert factory.calls[1]["conversation_id"] == alice.conversation_id
    assert factory.calls[1]["session_id"] == alice.member_id
    tasks.close()
