"""Task-owned Worktrees and controlled Team integration."""

from __future__ import annotations

from hashlib import sha256
import re
from threading import RLock
from uuid import UUID, uuid4

from fakuicode.teams.models import (
    ActorContext,
    MemberStatus,
    TaskKind,
    TaskStatus,
    TeamMember,
)
from fakuicode.teams.service import TeamService
from fakuicode.worktrees.git import GitCommandError
from fakuicode.worktrees.manager import WorktreeManager
from fakuicode.worktrees.models import WorktreeIdentity, WorktreeLease


_SHA = re.compile(r"[0-9a-f]{40,64}\Z")


class TeamGitCoordinator:
    def __init__(
        self,
        service: TeamService,
        worktree_manager: WorktreeManager,
    ) -> None:
        self.service = service
        self.worktree_manager = worktree_manager
        self._lock = RLock()
        self._integration: dict[str, WorktreeLease] = {}
        self._tasks: dict[str, WorktreeLease] = {}
        self._finalizations: dict[str, dict[str, str]] = {}

    def integration_lease(self, actor: ActorContext) -> WorktreeLease:
        self._require_lead(actor)
        return self._ensure_integration(actor.team_id)

    def task_lease(self, actor: ActorContext, task_id: str) -> WorktreeLease:
        self._require_lead(actor)
        task = self.service.store.get_task(actor.team_id, task_id)
        if task.worktree_branch is None or task.workspace is None:
            raise ValueError("任务尚未准备 Worktree。")
        return self._task_lease(task_id)

    def prepare_task(self, actor: ActorContext, task_id: str) -> WorktreeLease:
        self._require_lead(actor)
        task = self.service.store.get_task(actor.team_id, task_id)
        if task.kind is not TaskKind.TASK_WORKTREE:
            raise ValueError("只为 task_worktree 任务创建 Worktree。")
        preparable = task.status is TaskStatus.CLAIMED or (
            task.status is TaskStatus.WORKING
            and task.plan_approved
            and task.workspace is None
        )
        if not preparable or task.assignee_id is None:
            raise ValueError("任务必须先原子分配给成员。")
        integration = self._ensure_integration(actor.team_id)
        identity = WorktreeIdentity.for_role(UUID(task.task_id), "team-task")
        lease = self.worktree_manager.create(
            identity,
            base_ref=integration.branch,
        )
        prepared = task.revise(
            status=TaskStatus.WORKING,
            base_sha=lease.base_sha,
            worktree_branch=lease.branch,
            workspace=str(lease.execution_workspace),
        )
        self.service.store.save_task(actor.team_id, prepared)
        member = self._member(actor.team_id, task.assignee_id)
        active_member = TeamMember.from_dict(
            {
                **member.to_dict(),
                "status": MemberStatus.RUNNING.value,
                "current_task_id": task.task_id,
                "workspace": str(lease.execution_workspace),
            }
        )
        self.service.store.save_member(actor.team_id, active_member)
        with self._lock:
            self._tasks[task.task_id] = lease
        return lease

    def record_completion(
        self,
        actor: ActorContext,
        task_id: str,
        completion_sha: str,
        *,
        verification_summary: str,
    ) -> None:
        member = self._authorize_member(actor)
        task = self.service.store.get_task(actor.team_id, task_id)
        if task.assignee_id != member.member_id:
            raise ValueError("只能提交分配给自己的任务。")
        if task.status is not TaskStatus.WORKING:
            raise ValueError("任务当前不处于工作状态。")
        if (
            _SHA.fullmatch(completion_sha) is None
            or not verification_summary.strip()
            or task.base_sha is None
            or task.worktree_branch is None
        ):
            raise ValueError("完成提交或验证摘要无效。")
        lease = self._task_lease(task_id)
        git = self.worktree_manager.git
        timeout = self.worktree_manager.limits.metadata_timeout_seconds
        status = git.run(
            lease.execution_workspace,
            ("status", "--porcelain", "--untracked-files=all"),
            timeout=timeout,
        ).stdout
        head = git.run(
            lease.execution_workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        ancestor = git.run(
            lease.execution_workspace,
            ("merge-base", "--is-ancestor", task.base_sha, completion_sha),
            timeout=timeout,
            check=False,
        )
        if status or head != completion_sha or ancestor.returncode != 0:
            raise ValueError("任务 Worktree 不干净或完成提交不属于任务分支。")
        completed = task.revise(
            status=TaskStatus.INTEGRATING,
            completion_sha=completion_sha,
            verification_summary=verification_summary.strip(),
        )
        self.service.store.save_task(actor.team_id, completed)

    def integrate_task(
        self,
        actor: ActorContext,
        task_id: str,
    ) -> dict[str, object]:
        self._require_lead(actor)
        task = self.service.store.get_task(actor.team_id, task_id)
        if (
            task.status is not TaskStatus.INTEGRATING
            or task.worktree_branch is None
            or task.completion_sha is None
        ):
            raise ValueError("任务尚未完成可验证提交。")
        integration = self._ensure_integration(actor.team_id)
        git = self.worktree_manager.git
        timeout = self.worktree_manager.limits.lifecycle_timeout_seconds
        pre_merge_sha = git.run(
            integration.execution_workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        try:
            git.run(
                integration.execution_workspace,
                ("merge", "--no-ff", "--no-edit", task.worktree_branch),
                timeout=timeout,
            )
        except GitCommandError:
            git.run(
                integration.execution_workspace,
                ("merge", "--abort"),
                timeout=timeout,
                check=False,
            )
            failed = task.revise(status=TaskStatus.INTEGRATION_FAILED)
            self.service.store.save_task(actor.team_id, failed)
            return {
                "ok": False,
                "status": TaskStatus.INTEGRATION_FAILED.value,
                "pre_merge_sha": pre_merge_sha,
                "task_id": task.task_id,
            }
        integration_sha = git.run(
            integration.execution_workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        integrated = task.revise(
            status=TaskStatus.COMPLETED,
            integration_sha=integration_sha,
        )
        self.service.store.save_task(actor.team_id, integrated)
        return {
            "ok": True,
            "status": TaskStatus.COMPLETED.value,
            "task_id": task.task_id,
            "pre_merge_sha": pre_merge_sha,
            "integration_sha": integration_sha,
        }

    def prepare_finalization(self, actor: ActorContext) -> dict[str, str]:
        """Prepare a short-lived, host-generated confirmation for final delivery."""

        self._require_lead(actor)
        tasks = self.service.store.list_tasks(actor.team_id)
        if not tasks or any(
            task.status
            not in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.DELETED}
            for task in tasks
        ):
            raise ValueError("所有 Team 任务结束后才能准备最终交付。")
        team = self.service.store.get_team(actor.team_id)
        integration = self._ensure_integration(actor.team_id)
        git = self.worktree_manager.git
        timeout = self.worktree_manager.limits.metadata_timeout_seconds
        target_sha = git.run(
            self.worktree_manager.repo_root,
            ("rev-parse", "--verify", f"refs/heads/{team.target_branch}^{{commit}}"),
            timeout=timeout,
        ).stdout
        integration_sha = git.run(
            integration.execution_workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        if target_sha != team.target_sha:
            raise ValueError("目标分支已从 Team 创建时的提交发生变化，请先重新协调。")
        ancestor = git.run(
            integration.execution_workspace,
            ("merge-base", "--is-ancestor", target_sha, integration_sha),
            timeout=timeout,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError("集成分支无法安全快进到目标分支。")
        nonce = uuid4().hex
        token = sha256(
            f"{team.team_id}:{target_sha}:{integration_sha}:{nonce}".encode("utf-8")
        ).hexdigest()
        prepared = {
            "confirmation_token": token,
            "target_branch": team.target_branch,
            "target_sha": target_sha,
            "integration_sha": integration_sha,
        }
        with self._lock:
            self._finalizations[token] = prepared
        return prepared

    def finalize(self, actor: ActorContext, confirmation_token: str) -> dict[str, str]:
        """Fast-forward the clean target branch after exact token validation."""

        self._require_lead(actor)
        with self._lock:
            prepared = self._finalizations.get(confirmation_token)
        if prepared is None:
            raise ValueError("最终交付确认令牌无效或已经使用。")
        team = self.service.store.get_team(actor.team_id)
        if prepared["target_branch"] != team.target_branch:
            raise ValueError("确认令牌不属于当前 Team 的目标分支。")
        root = self.worktree_manager.repo_root
        git = self.worktree_manager.git
        timeout = self.worktree_manager.limits.lifecycle_timeout_seconds
        current_branch = git.run(
            root,
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            timeout=timeout,
            check=False,
        )
        if (
            current_branch.returncode != 0
            or current_branch.stdout != prepared["target_branch"]
        ):
            raise ValueError("必须在 Team 创建时锁定的目标分支上完成最终交付。")
        status = git.run(
            root,
            ("status", "--porcelain", "--untracked-files=all"),
            timeout=timeout,
        ).stdout
        if status:
            raise ValueError("目标工作树不干净，拒绝最终交付。")
        target_sha = git.run(
            root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        if target_sha != prepared["target_sha"]:
            raise ValueError("目标分支已变化，原确认令牌失效。")
        integration = self._ensure_integration(actor.team_id)
        integration_sha = git.run(
            integration.execution_workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        if integration_sha != prepared["integration_sha"]:
            raise ValueError("集成分支已变化，原确认令牌失效。")
        ancestor = git.run(
            root,
            ("merge-base", "--is-ancestor", target_sha, integration_sha),
            timeout=timeout,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError("集成分支无法安全快进到目标分支。")
        git.run(
            root,
            ("merge", "--ff-only", integration.branch),
            timeout=timeout,
        )
        delivered_sha = git.run(
            root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        if delivered_sha != integration_sha:
            raise ValueError("最终交付后的提交与已确认集成提交不一致。")
        with self._lock:
            self._finalizations.pop(confirmation_token, None)
        return {
            "ok": "true",
            "status": "finalized",
            "target_branch": team.target_branch,
            "integration_sha": delivered_sha,
        }

    def _ensure_integration(self, team_id: str) -> WorktreeLease:
        with self._lock:
            existing = self._integration.get(team_id)
            if existing is not None:
                return existing
        team = self.service.store.get_team(team_id)
        identity = WorktreeIdentity.for_role(UUID(team.team_id), "team-integration")
        lease = self.worktree_manager.create(identity, base_ref=team.target_sha)
        with self._lock:
            self._integration[team_id] = lease
        return lease

    def _task_lease(self, task_id: str) -> WorktreeLease:
        with self._lock:
            existing = self._tasks.get(task_id)
        if existing is not None:
            return existing
        identity = WorktreeIdentity.for_role(UUID(task_id), "team-task")
        lease = self.worktree_manager.create(identity)
        with self._lock:
            self._tasks[task_id] = lease
        return lease

    def _require_lead(self, actor: ActorContext) -> None:
        if actor != self.service.actor():
            raise ValueError("只有固定 Lead 可以管理 Team Git 集成。")

    def _authorize_member(self, actor: ActorContext) -> TeamMember:
        self.service.actor_for_member(actor.member_id)
        member = self._member(actor.team_id, actor.member_id)
        if member.name != actor.member_name:
            raise ValueError("成员身份与注册表不一致。")
        return member

    def _member(self, team_id: str, member_id: str) -> TeamMember:
        for member in self.service.store.list_members(team_id):
            if member.member_id == member_id:
                return member
        raise ValueError("成员不属于当前团队。")
