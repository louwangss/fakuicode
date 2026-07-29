"""Long-lived logical Team members on the existing bounded child runtime."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Protocol
from uuid import uuid4

from fakuicode.subagents.models import AgentDefinition
from fakuicode.subagents.runtime import ChildRunResult
from fakuicode.subagents.tasks import TaskManager
from fakuicode.teams.models import (
    ActorContext,
    BackendType,
    MemberStatus,
    MessageType,
    TaskKind,
    TaskStatus,
    TeamMember,
    TeamTask,
)
from fakuicode.teams.service import TeamService
from fakuicode.teams.control_tools import TeamTaskCompleteTool
from fakuicode.teams.tools import (
    TeamInboxListTool,
    TeamMessageSendTool,
    TeamPlanSubmitTool,
    TeamTaskCreateTool,
    TeamTaskDeleteTool,
    TeamTaskGetTool,
    TeamTaskListTool,
    TeamTaskUpdateTool,
)


class AgentCatalogProtocol(Protocol):
    def resolve(self, name: str) -> AgentDefinition: ...


class ChildRuntimeFactoryProtocol(Protocol):
    def create_defined(self, definition: AgentDefinition, **kwargs): ...


class TeamGitProtocol(Protocol):
    def prepare_task(self, actor: ActorContext, task_id: str): ...

    def task_lease(self, actor: ActorContext, task_id: str): ...


class _ManagedMemberSession:
    def __init__(
        self,
        service: TeamService,
        member: TeamMember,
        task_id: str | None,
        delegate,
    ) -> None:
        self.service = service
        self.member = member
        self.task_id = task_id
        self.delegate = delegate
        self.id = delegate.id
        self.name = delegate.name
        self.role = delegate.role
        self.profile_name = delegate.profile_name
        self.conversation_id = delegate.conversation_id
        self.execution = delegate.execution

    def run_to_completion(self, prompt: str, *, event_sink=None) -> ChildRunResult:
        actor = self.service.actor_for_member(self.member.member_id)
        inbox = self.service.list_inbox(actor, unread_only=True)
        delivered_ids = tuple(message.message_id for message in inbox)
        enriched = _prepend_inbox(prompt, inbox)
        try:
            result = self.delegate.run_to_completion(enriched, event_sink=event_sink)
        except Exception:
            result = ChildRunResult("", "failed", "成员运行时发生内部错误。")

        accepted = False
        effective_status: MemberStatus | None = None

        def finish_member(current: TeamMember) -> TeamMember:
            nonlocal accepted, effective_status
            if current.runtime_id != self.member.runtime_id:
                return current
            accepted = True
            if current.status is MemberStatus.WAITING_APPROVAL:
                effective_status = MemberStatus.WAITING_APPROVAL
                return current
            if current.status in {MemberStatus.STOPPING, MemberStatus.STOPPED}:
                status = MemberStatus.STOPPED
            elif result.status == "failed":
                status = MemberStatus.FAILED
            elif result.status == "cancelled":
                status = MemberStatus.STOPPED
            else:
                status = MemberStatus.IDLE
            effective_status = status
            return TeamMember.from_dict(
                {
                    **current.to_dict(),
                    "status": status.value,
                    "runtime_id": None,
                }
            )

        current = self.service.store.update_member(
            actor.team_id,
            self.member.member_id,
            finish_member,
        )
        if not accepted:
            return result
        assert effective_status is not None
        if (
            delivered_ids
            and result.status == "completed"
            and effective_status
            in {MemberStatus.IDLE, MemberStatus.WAITING_APPROVAL}
        ):
            self.service.mark_inbox_read(actor, delivered_ids)
        if self.task_id is not None:
            def finish_task(task: TeamTask) -> TeamTask:
                if task.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.DELETED,
                }:
                    return task
                if effective_status is MemberStatus.STOPPED:
                    return task.revise(
                        status=TaskStatus.CANCELLED,
                        result_summary=result.error or "成员运行已停止。",
                    )
                if effective_status is MemberStatus.FAILED:
                    return task.revise(
                        status=TaskStatus.FAILED,
                        result_summary=result.error or "成员执行失败。",
                    )
                if (
                    effective_status is MemberStatus.IDLE
                    and result.status == "completed"
                    and task.kind is TaskKind.READ_ONLY
                ):
                    return task.revise(
                        status=TaskStatus.COMPLETED,
                        result_summary=result.text[:2_000],
                    )
                return task

            self.service.store.update_task(actor.team_id, self.task_id, finish_task)
        lead_id = self.service.actor().member_id
        if effective_status is MemberStatus.IDLE:
            self.service.send_message(
                actor,
                to=lead_id,
                body=f"成员 {current.name} 已完成当前运行，可以继续指派。",
                summary=f"{current.name} 已空闲",
                message_type=MessageType.IDLE_NOTICE,
            )
        elif effective_status is MemberStatus.FAILED:
            self.service.send_message(
                actor,
                to=lead_id,
                body=f"成员 {current.name} 运行失败：{result.error or '未知错误'}",
                summary=f"{current.name} 运行失败",
                message_type=MessageType.TASK_EVENT,
            )
        elif effective_status is MemberStatus.STOPPED:
            self.service.send_message(
                actor,
                to=lead_id,
                body=f"成员 {current.name} 的当前运行已停止。",
                summary=f"{current.name} 已停止",
                message_type=MessageType.TASK_EVENT,
            )
        return result

    def cancel(self) -> None:
        self.delegate.cancel()

    def touch(self) -> None:
        self.delegate.touch()

    def close(self, *, status: str = "completed") -> None:
        self.delegate.close(status=status)


class TeamRuntimeManager:
    def __init__(
        self,
        service: TeamService,
        catalog: AgentCatalogProtocol,
        child_runtime: ChildRuntimeFactoryProtocol,
        task_manager: TaskManager,
        git_coordinator: TeamGitProtocol | None = None,
    ) -> None:
        self.service = service
        self.catalog = catalog
        self.child_runtime = child_runtime
        self.task_manager = task_manager
        self.git_coordinator = git_coordinator
        self._lock = RLock()
        self._sessions: dict[str, _ManagedMemberSession] = {}
        self._backend_selections: dict[str, dict[str, str]] = {}

    def start_member(
        self,
        actor: ActorContext,
        *,
        task_id: str,
        name: str,
        agent_type: str,
        role: str,
        prompt: str,
        description: str,
        profile: str | None,
        requires_plan_approval: bool,
        backend: BackendType = BackendType.AUTO,
    ) -> str:
        self._require_lead(actor)
        requested_backend = backend
        if backend is BackendType.AUTO:
            backend = BackendType.IN_PROCESS
            selection_reason = "当前运行环境仅配置了进程内成员运行器。"
        elif backend is BackendType.SUBPROCESS:
            raise ValueError(
                "subprocess 后端未配置可用的独立终端运行器；已拒绝启动，未降级。"
            )
        else:
            selection_reason = "调用方明确选择了进程内成员运行器。"
        definition = self.catalog.resolve(agent_type)
        session_id = str(uuid4())
        conversation_id = str(uuid4())
        member_actor = ActorContext(actor.team_id, session_id, name)
        effective_definition = (
            _planning_definition(definition) if requires_plan_approval else definition
        )
        member_profile = profile or definition.profile
        if member_profile == "inherit":
            member_profile = self.service.lead_profile
        member = TeamMember.create(
            member_id=session_id,
            name=member_actor.member_name,
            role=role,
            agent_type=agent_type,
            profile=member_profile,
            backend=backend,
            requires_plan_approval=requires_plan_approval,
            conversation_id=conversation_id,
        )
        try:
            self.service.store.add_member(actor.team_id, member)
            claimed = self.service.claim_task(actor, task_id, member.member_id)
            execution_lease = None
            if (
                claimed.kind is TaskKind.TASK_WORKTREE
                and not requires_plan_approval
            ):
                if self.git_coordinator is None:
                    raise ValueError("任务 Worktree 运行需要 Team Git 协调器。")
                execution_lease = self.git_coordinator.prepare_task(
                    actor,
                    task_id,
                )
            session = self.child_runtime.create_defined(
                effective_definition,
                profile_override=profile,
                name=member_actor.member_name,
                create_conversation_id=conversation_id,
                session_id=session_id,
                execution_lease=execution_lease,
                registry_configurator=lambda registry: _register_member_tools(
                    registry,
                    self.service,
                    member_actor,
                    include_plan_submit=requires_plan_approval,
                    git_coordinator=self.git_coordinator,
                    delivery_notifier=self.notify_messages,
                ),
                instruction_suffix=_team_instructions(
                    actor.team_id,
                    member_actor,
                    task_id,
                    requires_plan_approval,
                ),
            )
            running = TeamMember.from_dict(
                {
                    **member.to_dict(),
                    "status": (
                        MemberStatus.PLANNING.value
                        if requires_plan_approval
                        else MemberStatus.RUNNING.value
                    ),
                    "current_task_id": claimed.task_id,
                    "runtime_id": str(uuid4()),
                }
            )
            self.service.store.save_member(actor.team_id, running)
            managed = _ManagedMemberSession(self.service, running, task_id, session)
            run_id = self.task_manager.launch(
                managed,
                prompt,
                description,
                notify_on_done=True,
            )
        except Exception:
            if "session" in locals():
                session.close(status="error")
            self._mark_start_failure(actor.team_id, member, task_id)
            raise
        with self._lock:
            self._sessions[member.member_id] = managed
            self._backend_selections[run_id] = {
                "requested_backend": requested_backend.value,
                "selected_backend": backend.value,
                "selection_reason": selection_reason,
            }
        return run_id

    def selection_for_run(self, run_id: str) -> dict[str, str] | None:
        with self._lock:
            selection = self._backend_selections.get(run_id)
        return None if selection is None else dict(selection)

    def resume_member(
        self,
        actor: ActorContext,
        *,
        member_name: str,
        prompt: str,
        description: str,
    ) -> str:
        self._require_lead(actor)
        member_id = self.service.store.resolve_member(actor.team_id, member_name)
        member = next(
            item
            for item in self.service.store.list_members(actor.team_id)
            if item.member_id == member_id
        )
        if member.member_id == actor.member_id:
            raise ValueError("Lead 不能把自己作为成员续派。")
        current_task = (
            None
            if member.current_task_id is None
            else self.service.store.get_task(actor.team_id, member.current_task_id)
        )
        requires_worktree_rebind = (
            current_task is not None
            and current_task.kind is TaskKind.TASK_WORKTREE
            and current_task.workspace is None
            and (
                not member.requires_plan_approval
                or current_task.plan_approved
            )
        )
        with self._lock:
            managed = self._sessions.get(member.member_id)
        if managed is not None and not requires_worktree_rebind:
            return self.task_manager.send_message(member.name, prompt)
        if managed is not None:
            managed.close(status="completed")
            with self._lock:
                self._sessions.pop(member.member_id, None)
        definition = self.catalog.resolve(member.agent_type)
        effective_definition = (
            _planning_definition(definition)
            if (
                member.requires_plan_approval
                and current_task is not None
                and not current_task.plan_approved
            )
            else definition
        )
        member_actor = self.service.actor_for_member(member.member_id)
        execution_lease = None
        if current_task is not None:
            task = current_task
            if task.kind is TaskKind.TASK_WORKTREE:
                if self.git_coordinator is None:
                    raise ValueError("任务 Worktree 恢复需要 Team Git 协调器。")
                if task.workspace is None:
                    execution_lease = self.git_coordinator.prepare_task(
                        actor,
                        task.task_id,
                    )
                else:
                    execution_lease = self.git_coordinator.task_lease(
                        actor,
                        task.task_id,
                    )
        try:
            session = self.child_runtime.create_defined(
                effective_definition,
                profile_override=member.profile,
                name=member.name,
                conversation_id=member.conversation_id,
                session_id=member.member_id,
                execution_lease=execution_lease,
                registry_configurator=lambda registry: _register_member_tools(
                    registry,
                    self.service,
                    member_actor,
                    include_plan_submit=member.requires_plan_approval,
                    git_coordinator=self.git_coordinator,
                    delivery_notifier=self.notify_messages,
                ),
                instruction_suffix=_team_instructions(
                    actor.team_id,
                    member_actor,
                    member.current_task_id,
                    member.requires_plan_approval,
                ),
            )
            running = TeamMember.from_dict(
                {
                    **member.to_dict(),
                    "status": MemberStatus.RUNNING.value,
                    "runtime_id": str(uuid4()),
                }
            )
            self.service.store.save_member(actor.team_id, running)
            managed = _ManagedMemberSession(
                self.service,
                running,
                member.current_task_id,
                session,
            )
            run_id = self.task_manager.launch(
                managed,
                prompt,
                description,
                notify_on_done=True,
            )
        except Exception:
            if "session" in locals():
                session.close(status="error")
            if member.current_task_id is not None:
                self._mark_start_failure(
                    actor.team_id,
                    member,
                    member.current_task_id,
                )
            raise
        with self._lock:
            self._sessions[member.member_id] = managed
        return run_id

    def assign_member(
        self,
        actor: ActorContext,
        *,
        member_name: str,
        task_id: str,
        prompt: str,
        description: str,
    ) -> str:
        """Assign a new ready task while preserving the member conversation."""

        self._require_lead(actor)
        member_id = self.service.store.resolve_member(actor.team_id, member_name)
        member = next(
            item
            for item in self.service.store.list_members(actor.team_id)
            if item.member_id == member_id
        )
        if member.member_id == actor.member_id:
            raise ValueError("Lead 不能领取成员任务。")
        if member.status not in {
            MemberStatus.IDLE,
            MemberStatus.STOPPED,
            MemberStatus.FAILED,
        }:
            raise ValueError("只有空闲、停止或失败的成员可重新分配。")
        claimed = self.service.claim_task(actor, task_id, member.member_id)
        assigned = TeamMember.from_dict(
            {
                **member.to_dict(),
                "status": (
                    MemberStatus.PLANNING.value
                    if member.requires_plan_approval
                    else MemberStatus.IDLE.value
                ),
                "current_task_id": claimed.task_id,
                "workspace": None,
                "runtime_id": None,
            }
        )
        self.service.store.save_member(actor.team_id, assigned)
        with self._lock:
            managed = self._sessions.pop(member.member_id, None)
        if managed is not None:
            managed.close(status="completed")
        return self.resume_member(
            actor,
            member_name=member.name,
            prompt=prompt,
            description=description,
        )

    def stop_member(self, actor: ActorContext, member_name: str) -> bool:
        self._require_lead(actor)
        member_id = self.service.store.resolve_member(actor.team_id, member_name)
        member = next(
            item
            for item in self.service.store.list_members(actor.team_id)
            if item.member_id == member_id
        )
        if member.member_id == actor.member_id:
            raise ValueError("不能停止 Team Lead。")
        stop_requested = False
        stopped_runtime_id: str | None = None
        current_task_id: str | None = None

        def mark_stopping(current: TeamMember) -> TeamMember:
            nonlocal stop_requested, stopped_runtime_id, current_task_id
            current_task_id = current.current_task_id
            if current.status in {
                MemberStatus.STARTING,
                MemberStatus.PLANNING,
                MemberStatus.WAITING_APPROVAL,
                MemberStatus.RUNNING,
                MemberStatus.STOPPING,
            }:
                stop_requested = True
                stopped_runtime_id = current.runtime_id
                if current.status is MemberStatus.STOPPING:
                    return current
                return TeamMember.from_dict(
                    {
                        **current.to_dict(),
                        "status": MemberStatus.STOPPING.value,
                    }
                )
            return TeamMember.from_dict(
                {
                    **current.to_dict(),
                    "status": MemberStatus.STOPPED.value,
                    "runtime_id": None,
                }
            )

        self.service.store.update_member(actor.team_id, member_id, mark_stopping)
        stopped = self.task_manager.stop_by_name(member.name)

        def mark_stopped(current: TeamMember) -> TeamMember:
            if current.status is not MemberStatus.STOPPING:
                return current
            if current.runtime_id != stopped_runtime_id:
                return current
            return TeamMember.from_dict(
                {
                    **current.to_dict(),
                    "status": MemberStatus.STOPPED.value,
                    "runtime_id": None,
                }
            )

        self.service.store.update_member(
            actor.team_id,
            member_id,
            mark_stopped,
        )
        if stop_requested and current_task_id is not None:
            self.service.store.update_task(
                actor.team_id,
                current_task_id,
                lambda task: (
                    task
                    if task.status
                    in {
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.DELETED,
                    }
                    else task.revise(
                        status=TaskStatus.CANCELLED,
                        result_summary="成员运行已由 Lead 停止。",
                    )
                ),
            )
        return stopped

    def notify_messages(self, messages: tuple[object, ...]) -> None:
        """Wake idle in-process recipients from their persisted conversations."""

        lead = self.service.actor()
        members = {
            member.member_id: member
            for member in self.service.store.list_members(lead.team_id)
        }
        for message in messages:
            recipient_id = getattr(message, "recipient_id", None)
            member = members.get(recipient_id)
            if (
                member is None
                or member.member_id == lead.member_id
                or member.backend is not BackendType.IN_PROCESS
                or member.status not in {MemberStatus.IDLE, MemberStatus.STOPPED}
            ):
                continue
            self.resume_member(
                lead,
                member_name=member.name,
                prompt="读取 Team 邮箱中的新消息并继续协作。",
                description=f"邮箱唤醒：{member.name}",
            )

    def _require_lead(self, actor: ActorContext) -> None:
        lead = self.service.actor()
        if actor != lead:
            raise ValueError("只有固定 Lead 可以启动或续派团队成员。")

    def _mark_start_failure(
        self,
        team_id: str,
        member: TeamMember,
        task_id: str,
    ) -> None:
        """Keep failed startup evidence in a recoverable terminal state."""

        try:
            current_member = next(
                item
                for item in self.service.store.list_members(team_id)
                if item.member_id == member.member_id
            )
            self.service.store.save_member(
                team_id,
                TeamMember.from_dict(
                    {
                        **current_member.to_dict(),
                        "status": MemberStatus.FAILED.value,
                        "runtime_id": None,
                    }
                ),
            )
            task = self.service.store.get_task(team_id, task_id)
            if task.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
                TaskStatus.DELETED,
            }:
                self.service.store.save_task(
                    team_id,
                    task.revise(
                        status=TaskStatus.FAILED,
                        result_summary="成员运行时启动失败。",
                    ),
                )
        except (OSError, RuntimeError, ValueError, StopIteration):
            return


def _planning_definition(definition: AgentDefinition) -> AgentDefinition:
    return replace(
        definition,
        tools=("read_file", "find_files", "search_code"),
        disallowed_tools=(),
        isolation=None,
    )


def _register_member_tools(
    registry,
    service: TeamService,
    actor: ActorContext,
    *,
    include_plan_submit: bool,
    git_coordinator: TeamGitProtocol | None,
    delivery_notifier,
) -> set[str]:
    tools = [
        TeamTaskCreateTool(service, actor),
        TeamTaskGetTool(service, actor),
        TeamTaskListTool(service, actor),
        TeamTaskUpdateTool(service, actor),
        TeamTaskDeleteTool(service, actor),
        TeamMessageSendTool(
            service,
            actor,
            delivery_notifier=delivery_notifier,
        ),
        TeamInboxListTool(service, actor),
    ]
    if include_plan_submit:
        tools.append(TeamPlanSubmitTool(service, actor))
    if git_coordinator is not None:
        tools.append(TeamTaskCompleteTool(actor, git_coordinator))
    for tool in tools:
        registry.register(tool)
    return {tool.definition.name for tool in tools}


def _team_instructions(
    team_id: str,
    actor: ActorContext,
    task_id: str | None,
    requires_plan_approval: bool,
) -> str:
    approval = (
        "开始实施前必须调用 team_plan_submit 提交结构化计划；提交后停止本轮并等待 Lead。"
        if requires_plan_approval
        else "按照共享任务与邮箱直接同其他成员协作。"
    )
    return (
        "## Team 成员边界\n\n"
        f"- team_id：{team_id}\n"
        f"- member_id：{actor.member_id}\n"
        f"- 成员名：{actor.member_name}\n"
        f"- 当前任务：{task_id or '未绑定'}\n"
        "- 普通文本回复不会自动发送给其他成员；需要通信时调用 team_message_send。\n"
        f"- {approval}"
    )


def _prepend_inbox(prompt: str, messages) -> str:
    if not messages:
        return prompt
    lines = ["<incoming-team-messages>"]
    for message in messages:
        lines.append(
            f"[{message.message_id}] 来自 {message.sender_name}：{message.summary}\n"
            f"{message.body}"
        )
    lines.append("</incoming-team-messages>")
    lines.append("")
    lines.append(prompt)
    return "\n".join(lines)
