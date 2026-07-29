"""Authorized application service for one Lead's active Team."""

from __future__ import annotations

from uuid import uuid4

from fakuicode.teams.models import (
    ActorContext,
    BackendType,
    MemberStatus,
    MessageType,
    TaskKind,
    TaskStatus,
    TeamMember,
    TeamMessage,
    TeamRecord,
    TeamTask,
)
from fakuicode.teams.storage import TeamStore


class TeamService:
    def __init__(
        self,
        store: TeamStore,
        *,
        lead_conversation_id: str,
        repository_fingerprint: str,
        target_branch: str,
        target_sha: str,
        lead_profile: str,
    ) -> None:
        self.store = store
        self.lead_conversation_id = lead_conversation_id
        self.repository_fingerprint = repository_fingerprint
        self.target_branch = target_branch
        self.target_sha = target_sha
        self.lead_profile = lead_profile
        self.active_team_id: str | None = None

    def create_team(self, name: str) -> TeamRecord:
        if self.active_team_id is not None:
            raise ValueError("当前 Lead 已附着一个活跃团队。")
        team = self.store.create_team(
            name=name,
            lead_conversation_id=self.lead_conversation_id,
            repository_fingerprint=self.repository_fingerprint,
            target_branch=self.target_branch,
            target_sha=self.target_sha,
        )
        lead = TeamMember.create(
            name="lead",
            role="团队负责人",
            profile=self.lead_profile,
            backend=BackendType.IN_PROCESS,
            requires_plan_approval=False,
            conversation_id=self.lead_conversation_id,
            member_id=team.lead_member_id,
        )
        try:
            self.store.add_member(team.team_id, lead)
        except Exception:
            # The partially-created Team remains inspectable instead of deleting
            # coordination evidence behind the caller's back.
            raise
        self.active_team_id = team.team_id
        return team

    def attach_team(self, name: str) -> TeamRecord:
        team = self.store.get_team(name)
        if team.lead_conversation_id != self.lead_conversation_id:
            raise ValueError("团队绑定到另一个 Lead 会话。")
        if team.repository_fingerprint != self.repository_fingerprint:
            raise ValueError("团队所属仓库与当前仓库不一致。")
        if self.active_team_id not in {None, team.team_id}:
            raise ValueError("当前 Lead 已附着另一个团队。")
        self.active_team_id = team.team_id
        return team

    def detach_team(self) -> None:
        self.active_team_id = None

    def actor(self) -> ActorContext:
        team = self._active_team()
        lead = self._member(team.lead_member_id)
        return ActorContext(team.team_id, lead.member_id, lead.name)

    def actor_for_member(self, member_id: str) -> ActorContext:
        team = self._active_team()
        member = self._member(member_id)
        return ActorContext(team.team_id, member.member_id, member.name)

    def create_task(
        self,
        actor: ActorContext,
        *,
        title: str,
        description: str,
        blocked_by: tuple[str, ...] = (),
        kind: TaskKind = TaskKind.TASK_WORKTREE,
    ) -> TeamTask:
        self._authorize(actor)
        task = TeamTask.create(
            title=title,
            description=description,
            created_by=actor.member_id,
            blocked_by=blocked_by,
            kind=kind,
        )
        return self.store.create_task(actor.team_id, task)

    def list_tasks(self, actor: ActorContext) -> tuple[dict[str, object], ...]:
        self._authorize(actor)
        tasks = self.store.list_tasks(actor.team_id)
        by_id = {task.task_id: task for task in tasks}
        return tuple(
            {
                **task.to_dict(),
                "blocks": [
                    candidate.task_id
                    for candidate in tasks
                    if task.task_id in candidate.blocked_by
                ],
                "is_ready": all(
                    by_id[dependency].status is TaskStatus.COMPLETED
                    for dependency in task.blocked_by
                ),
            }
            for task in tasks
        )

    def get_task(self, actor: ActorContext, task_id: str) -> dict[str, object]:
        self._authorize(actor)
        task = self.store.get_task(actor.team_id, task_id)
        return next(
            item for item in self.list_tasks(actor) if item["task_id"] == task.task_id
        )

    def update_task(
        self,
        actor: ActorContext,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        blocked_by: tuple[str, ...] | None = None,
    ) -> TeamTask:
        self._authorize(actor)
        normalized_title = None
        if title is not None:
            normalized_title = title.strip()
            if not normalized_title:
                raise ValueError("任务标题不能为空。")
            if len(normalized_title) > 500:
                raise ValueError("任务标题不能超过 500 个字符。")
        normalized_description = None
        if description is not None:
            normalized_description = description.strip()
        return self.store.update_pending_task(
            actor.team_id,
            task_id,
            title=normalized_title,
            description=normalized_description,
            blocked_by=blocked_by,
        )

    def delete_task(self, actor: ActorContext, task_id: str) -> TeamTask:
        self._authorize(actor)
        return self.store.delete_pending_task(actor.team_id, task_id)

    def claim_task(
        self,
        actor: ActorContext,
        task_id: str,
        member_id: str,
    ) -> TeamTask:
        team = self._active_team()
        if actor.member_id != team.lead_member_id:
            raise ValueError("只有 Lead 可以分配团队任务。")
        self._authorize(actor)
        self._member(member_id)
        return self.store.claim_task(team.team_id, task_id, member_id)

    def submit_plan(
        self,
        actor: ActorContext,
        *,
        task_id: str,
        plan: str,
        summary: str,
    ) -> dict[str, object]:
        member = self._authorize(actor)
        if not member.requires_plan_approval:
            raise ValueError("该成员不需要提交审批计划。")
        task = self.store.get_task(actor.team_id, task_id)
        if task.assignee_id != actor.member_id:
            raise ValueError("只能为分配给自己的任务提交计划。")
        if task.status not in {TaskStatus.CLAIMED, TaskStatus.PLANNING}:
            raise ValueError("当前任务状态不能提交计划。")
        request_id = str(uuid4())
        revision = task.plan_revision + 1
        planned = task.revise(
            status=TaskStatus.PLANNING,
            plan_request_id=request_id,
            plan_revision=revision,
            plan_approved=False,
        )
        waiting = TeamMember.from_dict(
            {
                **member.to_dict(),
                "status": MemberStatus.WAITING_APPROVAL.value,
                "current_task_id": task.task_id,
            }
        )
        team = self._active_team()
        self.store.commit_workflow_transition(
            actor,
            task=planned,
            member=waiting,
            to=team.lead_member_id,
            body=plan,
            summary=summary,
            message_type=MessageType.PLAN_REQUEST,
            correlation_id=request_id,
            payload={
                "request_id": request_id,
                "task_id": task.task_id,
                "revision": revision,
            },
        )
        return {
            "request_id": request_id,
            "task_id": task.task_id,
            "revision": revision,
        }

    def review_plan(
        self,
        actor: ActorContext,
        *,
        task_id: str,
        request_id: str,
        revision: int,
        approved: bool,
        feedback: str,
    ) -> TeamTask:
        team = self._active_team()
        if actor.member_id != team.lead_member_id:
            raise ValueError("只有 Lead 可以审批成员计划。")
        self._authorize(actor)
        task = self.store.get_task(actor.team_id, task_id)
        if (
            task.status is not TaskStatus.PLANNING
            or task.plan_request_id != request_id
            or task.plan_revision != revision
            or task.plan_approved
            or task.assignee_id is None
        ):
            raise ValueError("计划审批与当前 request_id 或 revision 不匹配。")
        assignee = self._member(task.assignee_id)
        reviewed = task.revise(
            status=TaskStatus.WORKING if approved else TaskStatus.PLANNING,
            plan_approved=approved,
        )
        updated_member = TeamMember.from_dict(
            {
                **assignee.to_dict(),
                "status": (
                    MemberStatus.IDLE.value
                    if approved
                    else MemberStatus.PLANNING.value
                ),
                "current_task_id": task.task_id,
            }
        )
        self.store.commit_workflow_transition(
            actor,
            task=reviewed,
            member=updated_member,
            to=assignee.member_id,
            body=feedback or ("计划已批准。" if approved else "计划已驳回，请修订。"),
            summary="计划已批准" if approved else "计划需修订",
            message_type=MessageType.PLAN_REVIEW,
            correlation_id=request_id,
            payload={
                "request_id": request_id,
                "task_id": task.task_id,
                "revision": revision,
                "approved": approved,
            },
        )
        return reviewed

    def send_message(
        self,
        actor: ActorContext,
        *,
        to: str,
        body: str,
        summary: str,
        message_type: MessageType = MessageType.TEXT,
        payload: dict[str, object] | None = None,
    ) -> tuple[TeamMessage, ...]:
        self._authorize(actor)
        team = self._active_team()
        if message_type is MessageType.PLAN_REVIEW and actor.member_id != team.lead_member_id:
            raise ValueError("只有 Lead 可以发送计划审批消息。")
        if message_type in {MessageType.PLAN_REQUEST, MessageType.IDLE_NOTICE} and (
            actor.member_id == team.lead_member_id
        ):
            raise ValueError("Lead 不能伪造成员工作流消息。")
        if message_type is MessageType.SHUTDOWN_RESPONSE and to not in {
            team.lead_member_id,
            "lead",
        }:
            raise ValueError("shutdown_response 只能发送给 Lead。")
        if to != "*":
            return (
                self.store.send_message(
                    actor,
                    to=to,
                    body=body,
                    summary=summary,
                    message_type=message_type,
                    payload=payload,
                ),
            )
        correlation_id = str(uuid4())
        return tuple(
            self.store.send_message(
                actor,
                to=member.member_id,
                body=body,
                summary=summary,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id,
            )
            for member in self.store.list_members(actor.team_id)
            if member.member_id != actor.member_id
        )

    def list_inbox(
        self,
        actor: ActorContext,
        *,
        unread_only: bool = True,
    ) -> tuple[TeamMessage, ...]:
        self._authorize(actor)
        return self.store.list_messages(
            actor.team_id,
            actor.member_id,
            unread_only=unread_only,
        )

    def mark_inbox_read(
        self,
        actor: ActorContext,
        message_ids: tuple[str, ...],
    ) -> None:
        self._authorize(actor)
        self.store.mark_messages_read(actor.team_id, actor.member_id, message_ids)

    def _active_team(self) -> TeamRecord:
        if self.active_team_id is None:
            raise ValueError("当前 Lead 尚未附着团队。")
        return self.store.get_team(self.active_team_id)

    def _member(self, member_id: str) -> TeamMember:
        team = self._active_team()
        for member in self.store.list_members(team.team_id):
            if member.member_id == member_id:
                return member
        raise ValueError("成员不属于当前团队。")

    def _authorize(self, actor: ActorContext) -> TeamMember:
        team = self._active_team()
        if actor.team_id != team.team_id:
            raise ValueError("调用方不属于当前团队。")
        member = self._member(actor.member_id)
        if member.name != actor.member_name:
            raise ValueError("调用方身份与团队注册表不一致。")
        return member
