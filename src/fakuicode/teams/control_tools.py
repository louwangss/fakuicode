"""Lead-only Team runtime, approval, and Git control tools."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Protocol
from typing import Callable

from fakuicode.errors import ToolExecutionError
from fakuicode.models import ToolDefinition
from fakuicode.teams.models import ActorContext, BackendType
from fakuicode.teams.service import TeamService
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments


class TeamRuntimeProtocol(Protocol):
    def start_member(self, actor: ActorContext, **kwargs: object) -> str: ...

    def resume_member(self, actor: ActorContext, **kwargs: object) -> str: ...

    def stop_member(self, actor: ActorContext, member_name: str) -> bool: ...

    def assign_member(self, actor: ActorContext, **kwargs: object) -> str: ...


class TeamGitProtocol(Protocol):
    def integrate_task(self, actor: ActorContext, task_id: str) -> dict[str, object]: ...

    def prepare_finalization(self, actor: ActorContext) -> dict[str, str]: ...

    def finalize(
        self, actor: ActorContext, confirmation_token: str
    ) -> dict[str, str]: ...

    def record_completion(
        self,
        actor: ActorContext,
        task_id: str,
        completion_sha: str,
        *,
        verification_summary: str,
    ) -> None: ...


def _success(payload: Mapping[str, object], summary: str) -> ToolExecution:
    return ToolExecution(
        True,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        summary,
    )


def _error(code: str, error: Exception) -> ToolExecution:
    message = str(error)
    return ToolExecution(
        False,
        json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        message,
    )


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError(f"{key} 必须是非空字符串。")
    return value.strip()


def _workflow_preparation(
    actor: ActorContext,
    arguments: Mapping[str, object],
    target: str,
    *,
    enabled: bool = True,
) -> ToolPreparation:
    return ToolPreparation(
        freeze_arguments(arguments),
        target,
        permission_capability=actor.workflow_capability if enabled else None,
    )


class TeamMemberStartTool:
    _FIELDS = {
        "task_id",
        "name",
        "agent_type",
        "role",
        "prompt",
        "description",
        "profile",
        "requires_plan_approval",
        "backend",
    }

    def __init__(self, actor: ActorContext, runtime: TeamRuntimeProtocol) -> None:
        self.actor = actor
        self.runtime = runtime

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_member_start",
            "创建长期 Team 成员、原子分配共享任务并启动其运行后端。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "name": {"type": "string"},
                    "agent_type": {"type": "string"},
                    "role": {"type": "string"},
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                    "profile": {"type": "string"},
                    "requires_plan_approval": {"type": "boolean"},
                    "backend": {
                        "type": "string",
                        "enum": [item.value for item in BackendType],
                    },
                },
                "required": [
                    "task_id",
                    "name",
                    "agent_type",
                    "role",
                    "prompt",
                    "description",
                    "requires_plan_approval",
                ],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        unknown = set(arguments) - self._FIELDS
        if unknown:
            raise ToolExecutionError(
                f"team_member_start 收到未知参数：{', '.join(sorted(unknown))}"
            )
        normalized = {
            key: _text(arguments, key)
            for key in (
                "task_id",
                "name",
                "agent_type",
                "role",
                "prompt",
                "description",
            )
        }
        profile = arguments.get("profile")
        if profile is not None and (
            not isinstance(profile, str) or not profile.strip()
        ):
            raise ToolExecutionError("profile 必须是非空字符串。")
        approval = arguments.get("requires_plan_approval")
        if not isinstance(approval, bool):
            raise ToolExecutionError("requires_plan_approval 必须是布尔值。")
        try:
            backend = BackendType(
                str(arguments.get("backend", BackendType.AUTO.value))
            )
        except ValueError as error:
            raise ToolExecutionError("backend 无效。") from error
        normalized.update(
            {
                "profile": None if profile is None else profile.strip(),
                "requires_plan_approval": approval,
                "backend": backend.value,
            }
        )
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:members",
            enabled=backend is not BackendType.SUBPROCESS,
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError, KeyError) as error:
            return _error("member_start_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            run_id = self.runtime.start_member(
                self.actor,
                task_id=str(arguments["task_id"]),
                name=str(arguments["name"]),
                agent_type=str(arguments["agent_type"]),
                role=str(arguments["role"]),
                prompt=str(arguments["prompt"]),
                description=str(arguments["description"]),
                profile=(
                    None
                    if arguments["profile"] is None
                    else str(arguments["profile"])
                ),
                requires_plan_approval=bool(arguments["requires_plan_approval"]),
                backend=BackendType(str(arguments["backend"])),
            )
        except (ValueError, RuntimeError, KeyError) as error:
            return _error("member_start_failed", error)
        selection_getter = getattr(self.runtime, "selection_for_run", None)
        selection = selection_getter(run_id) if callable(selection_getter) else None
        return _success(
            {"ok": True, "run_id": run_id, **(selection or {})},
            "Team 成员已启动",
        )


class TeamPlanReviewTool:
    _FIELDS = {"task_id", "request_id", "revision", "approved", "feedback"}

    def __init__(
        self,
        service: TeamService,
        actor: ActorContext,
        *,
        on_reviewed: Callable[[object], None] | None = None,
    ) -> None:
        self.service = service
        self.actor = actor
        self.on_reviewed = on_reviewed

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_plan_review",
            "Lead 使用精确 request_id 和 revision 批准或驳回成员计划。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "request_id": {"type": "string"},
                    "revision": {"type": "integer", "minimum": 1},
                    "approved": {"type": "boolean"},
                    "feedback": {"type": "string"},
                },
                "required": sorted(self._FIELDS),
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != self._FIELDS:
            raise ToolExecutionError("team_plan_review 参数不完整或包含未知字段。")
        revision = arguments.get("revision")
        approved = arguments.get("approved")
        feedback = arguments.get("feedback")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ToolExecutionError("revision 必须是正整数。")
        if not isinstance(approved, bool):
            raise ToolExecutionError("approved 必须是布尔值。")
        if not isinstance(feedback, str):
            raise ToolExecutionError("feedback 必须是字符串。")
        normalized = {
            "task_id": _text(arguments, "task_id"),
            "request_id": _text(arguments, "request_id"),
            "revision": revision,
            "approved": approved,
            "feedback": feedback.strip(),
        }
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:plan:{normalized['task_id']}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _error("plan_review_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            task = self.service.review_plan(
                self.actor,
                task_id=str(arguments["task_id"]),
                request_id=str(arguments["request_id"]),
                revision=int(arguments["revision"]),
                approved=bool(arguments["approved"]),
                feedback=str(arguments["feedback"]),
            )
        except (ValueError, RuntimeError) as error:
            return _error("plan_review_failed", error)
        wake_warning = None
        if self.on_reviewed is not None:
            try:
                self.on_reviewed(task)
            except Exception:
                wake_warning = "审批已持久保存，但成员自动恢复失败。"
        return _success(
            {
                "ok": True,
                "task": task.to_dict(),
                "wake_warning": wake_warning,
            },
            "成员计划已审批",
        )


class TeamMemberResumeTool:
    def __init__(self, actor: ActorContext, runtime: TeamRuntimeProtocol) -> None:
        self.actor = actor
        self.runtime = runtime

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_member_resume",
            "从持久会话恢复空闲 Team 成员并继续派活。",
            {
                "type": "object",
                "properties": {
                    "member_name": {"type": "string"},
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["member_name", "prompt", "description"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        expected = {"member_name", "prompt", "description"}
        if set(arguments) != expected:
            raise ToolExecutionError("team_member_resume 参数不完整或包含未知字段。")
        normalized = {key: _text(arguments, key) for key in expected}
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:members",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError, KeyError) as error:
            return _error("member_resume_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            run_id = self.runtime.resume_member(
                self.actor,
                member_name=str(arguments["member_name"]),
                prompt=str(arguments["prompt"]),
                description=str(arguments["description"]),
            )
        except (ValueError, RuntimeError, KeyError) as error:
            return _error("member_resume_failed", error)
        return _success({"ok": True, "run_id": run_id}, "Team 成员已恢复")


class TeamMemberAssignTool:
    _FIELDS = {"member_name", "task_id", "prompt", "description"}

    def __init__(self, actor: ActorContext, runtime: TeamRuntimeProtocol) -> None:
        self.actor = actor
        self.runtime = runtime

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_member_assign",
            "把新的 ready 任务分配给已有空闲成员，并从其持久会话继续工作。",
            {
                "type": "object",
                "properties": {
                    "member_name": {"type": "string"},
                    "task_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": sorted(self._FIELDS),
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != self._FIELDS:
            raise ToolExecutionError("team_member_assign 参数不完整或包含未知字段。")
        normalized = {key: _text(arguments, key) for key in self._FIELDS}
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:members",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError, KeyError) as error:
            return _error("member_assign_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            run_id = self.runtime.assign_member(
                self.actor,
                member_name=str(arguments["member_name"]),
                task_id=str(arguments["task_id"]),
                prompt=str(arguments["prompt"]),
                description=str(arguments["description"]),
            )
        except (ValueError, RuntimeError, KeyError) as error:
            return _error("member_assign_failed", error)
        return _success({"ok": True, "run_id": run_id}, "Team 成员已重新分配")


class TeamMemberStopTool:
    def __init__(self, actor: ActorContext, runtime: TeamRuntimeProtocol) -> None:
        self.actor = actor
        self.runtime = runtime

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_member_stop",
            "停止指定 Team 成员的当前运行；持久会话和成员登记仍保留。",
            {
                "type": "object",
                "properties": {"member_name": {"type": "string"}},
                "required": ["member_name"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"member_name"}:
            raise ToolExecutionError("team_member_stop 只接受 member_name。")
        member_name = _text(arguments, "member_name")
        return _workflow_preparation(
            self.actor,
            {"member_name": member_name},
            f"team:{self.actor.team_id}:members",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _error("member_stop_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            stopped = self.runtime.stop_member(
                self.actor,
                str(arguments["member_name"]),
            )
        except (ValueError, RuntimeError) as error:
            return _error("member_stop_failed", error)
        return _success({"ok": True, "stopped": stopped}, "Team 成员停止请求已处理")


class TeamTaskCompleteTool:
    _FIELDS = {"task_id", "completion_sha", "verification_summary"}

    def __init__(self, actor: ActorContext, git: TeamGitProtocol) -> None:
        self.actor = actor
        self.git = git

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_task_complete",
            "成员提交干净任务分支的完成提交和验证摘要，等待 Lead 集成。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "completion_sha": {"type": "string"},
                    "verification_summary": {"type": "string"},
                },
                "required": sorted(self._FIELDS),
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != self._FIELDS:
            raise ToolExecutionError("team_task_complete 参数不完整或包含未知字段。")
        normalized = {key: _text(arguments, key) for key in self._FIELDS}
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:task:{normalized['task_id']}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _error("task_completion_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            self.git.record_completion(
                self.actor,
                str(arguments["task_id"]),
                str(arguments["completion_sha"]),
                verification_summary=str(arguments["verification_summary"]),
            )
        except (ValueError, RuntimeError) as error:
            return _error("task_completion_failed", error)
        return _success(
            {"ok": True, "status": "integrating"},
            "任务完成提交已登记",
        )


class TeamIntegrateTaskTool:
    def __init__(self, actor: ActorContext, git: TeamGitProtocol) -> None:
        self.actor = actor
        self.git = git

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_integrate_task",
            "将已验证的任务分支合并到 Team 集成分支；冲突时自动中止并报告。",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"task_id"}:
            raise ToolExecutionError("team_integrate_task 只接受 task_id。")
        task_id = _text(arguments, "task_id")
        return _workflow_preparation(
            self.actor,
            {"task_id": task_id},
            f"team:{self.actor.team_id}:integration",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _error("task_integration_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            result = self.git.integrate_task(self.actor, str(arguments["task_id"]))
        except (ValueError, RuntimeError) as error:
            return _error("task_integration_failed", error)
        return _success(result, f"任务集成状态：{result['status']}")


class TeamFinalizePrepareTool:
    def __init__(self, actor: ActorContext, git: TeamGitProtocol) -> None:
        self.actor = actor
        self.git = git

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_finalize_prepare",
            "检查 Team 最终交付前置条件并生成一次性确认令牌。",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if arguments:
            raise ToolExecutionError("team_finalize_prepare 不接受参数。")
        return _workflow_preparation(
            self.actor,
            {},
            f"team:{self.actor.team_id}:finalize",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _error("finalize_prepare_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del arguments, cancel_event
        try:
            result = self.git.prepare_finalization(self.actor)
        except (ValueError, RuntimeError) as error:
            return _error("finalize_prepare_failed", error)
        return _success(result, "最终交付已准备，等待精确令牌确认")


class TeamFinalizeTool:
    def __init__(self, actor: ActorContext, git: TeamGitProtocol) -> None:
        self.actor = actor
        self.git = git

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_finalize",
            "使用 host 生成的一次性令牌，将集成分支安全快进到锁定目标分支。",
            {
                "type": "object",
                "properties": {"confirmation_token": {"type": "string"}},
                "required": ["confirmation_token"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"confirmation_token"}:
            raise ToolExecutionError("team_finalize 只接受 confirmation_token。")
        token = _text(arguments, "confirmation_token")
        return ToolPreparation(
            freeze_arguments({"confirmation_token": token}),
            f"team:{self.actor.team_id}:finalize",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _error("finalize_failed", error)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            result = self.git.finalize(
                self.actor,
                str(arguments["confirmation_token"]),
            )
        except (ValueError, RuntimeError) as error:
            return _error("finalize_failed", error)
        return _success(result, "Team 最终交付已完成")
