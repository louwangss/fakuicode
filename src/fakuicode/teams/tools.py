"""Provider-visible tools bound to a host-authenticated Team actor."""

from __future__ import annotations

from collections.abc import Mapping
import json
from threading import Event
from typing import Callable

from fakuicode.errors import ToolExecutionError
from fakuicode.models import ToolDefinition
from fakuicode.teams.models import ActorContext, MessageType, TaskKind
from fakuicode.teams.service import TeamService
from fakuicode.tools.base import (
    ToolExecution,
    ToolPreparation,
    freeze_arguments,
)


def _json_success(payload: Mapping[str, object], summary: str) -> ToolExecution:
    return ToolExecution(
        True,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        summary,
    )


def _json_error(code: str, message: str) -> ToolExecution:
    return ToolExecution(
        False,
        json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        message,
    )


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError(f"{key} 必须是非空字符串。")
    return value.strip()


def _workflow_preparation(
    actor: ActorContext,
    arguments: Mapping[str, object],
    target: str,
) -> ToolPreparation:
    return ToolPreparation(
        freeze_arguments(arguments),
        target,
        permission_capability=actor.workflow_capability,
    )


class TeamCreateTool:
    def __init__(
        self,
        service: TeamService,
        *,
        on_created: Callable[[object], None] | None = None,
    ) -> None:
        self.service = service
        self.on_created = on_created

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_create",
            "根据用户明确请求创建长期 Agent Team，并把当前主 Agent 注册为固定 Lead。",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"name"}:
            raise ToolExecutionError("team_create 只接受 name 参数。")
        name = _required_text(arguments, "name")
        return ToolPreparation(freeze_arguments({"name": name}), f"team:{name}")

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("team_create_failed", str(error))

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            team = self.service.create_team(str(arguments["name"]))
        except (ValueError, RuntimeError) as error:
            return _json_error("team_create_failed", str(error))
        activation_warning = None
        if self.on_created is not None:
            try:
                self.on_created(team)
            except Exception:
                activation_warning = "Team 已创建，但当前会话工具激活失败；请重启并恢复该会话。"
        return _json_success(
            {
                "ok": True,
                "team": team.to_dict(),
                "activation_warning": activation_warning,
            },
            f"团队 {team.name} 已创建",
        )


class TeamTaskCreateTool:
    _FIELDS = {"title", "description", "blocked_by", "kind"}

    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_task_create",
            "在当前 Team 的共享任务图中创建任务。blocked_by 只接受当前 Team 的任务 UUID。",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "blocked_by": {"type": "array", "items": {"type": "string"}},
                    "kind": {
                        "type": "string",
                        "enum": ["read_only", "task_worktree"],
                    },
                },
                "required": ["title"],
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
                f"team_task_create 收到未知参数：{', '.join(sorted(unknown))}"
            )
        title = _required_text(arguments, "title")
        description = arguments.get("description", "")
        if not isinstance(description, str):
            raise ToolExecutionError("description 必须是字符串。")
        blocked = arguments.get("blocked_by", ())
        if not isinstance(blocked, (list, tuple)) or not all(
            isinstance(item, str) for item in blocked
        ):
            raise ToolExecutionError("blocked_by 必须是字符串数组。")
        try:
            kind = TaskKind(str(arguments.get("kind", TaskKind.TASK_WORKTREE.value)))
        except ValueError as error:
            raise ToolExecutionError("kind 必须是 read_only 或 task_worktree。") from error
        normalized = {
            "title": title,
            "description": description,
            "blocked_by": tuple(blocked),
            "kind": kind.value,
        }
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:tasks",
        )

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("task_create_failed", str(error))

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            task = self.service.create_task(
                self.actor,
                title=str(arguments["title"]),
                description=str(arguments["description"]),
                blocked_by=tuple(arguments["blocked_by"]),
                kind=TaskKind(str(arguments["kind"])),
            )
        except (ValueError, RuntimeError) as error:
            return _json_error("task_create_failed", str(error))
        return _json_success(
            {"ok": True, "task": task.to_dict()},
            f"团队任务已创建：{task.title}",
        )


class TeamTaskListTool:
    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_task_list",
            "读取当前 Team 的共享任务列表及派生的依赖就绪状态。",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if arguments:
            raise ToolExecutionError("team_task_list 不接受参数。")
        return ToolPreparation(
            freeze_arguments({}),
            f"team:{self.actor.team_id}:tasks",
        )

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("task_list_failed", str(error))

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del arguments, cancel_event
        try:
            tasks = self.service.list_tasks(self.actor)
        except (ValueError, RuntimeError) as error:
            return _json_error("task_list_failed", str(error))
        return _json_success(
            {"ok": True, "tasks": list(tasks)},
            f"读取到 {len(tasks)} 个团队任务",
        )


class TeamTaskGetTool:
    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_task_get",
            "按 UUID 读取当前 Team 的一项共享任务及其依赖状态。",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"task_id"}:
            raise ToolExecutionError("team_task_get 只接受 task_id。")
        task_id = _required_text(arguments, "task_id")
        return ToolPreparation(
            freeze_arguments({"task_id": task_id}),
            f"team:{self.actor.team_id}:tasks:{task_id}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("task_get_failed", str(error))

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            task = self.service.get_task(self.actor, str(arguments["task_id"]))
        except (ValueError, RuntimeError) as error:
            return _json_error("task_get_failed", str(error))
        return _json_success({"ok": True, "task": task}, "团队任务已读取")


class TeamTaskUpdateTool:
    _FIELDS = {"task_id", "title", "description", "blocked_by"}

    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_task_update",
            "修改 pending 共享任务的标题、描述或完整 blocked_by 依赖集合。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        unknown = set(arguments) - self._FIELDS
        if unknown or "task_id" not in arguments or len(arguments) == 1:
            raise ToolExecutionError("team_task_update 参数无效或没有更新字段。")
        normalized: dict[str, object] = {
            "task_id": _required_text(arguments, "task_id")
        }
        for key in ("title", "description"):
            value = arguments.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ToolExecutionError(f"{key} 必须是字符串。")
                normalized[key] = value
        blocked = arguments.get("blocked_by")
        if blocked is not None:
            if not isinstance(blocked, (list, tuple)) or not all(
                isinstance(item, str) for item in blocked
            ):
                raise ToolExecutionError("blocked_by 必须是字符串数组。")
            normalized["blocked_by"] = tuple(blocked)
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:tasks:{normalized['task_id']}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("task_update_failed", str(error))

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            task = self.service.update_task(
                self.actor,
                str(arguments["task_id"]),
                title=(
                    None if "title" not in arguments else str(arguments["title"])
                ),
                description=(
                    None
                    if "description" not in arguments
                    else str(arguments["description"])
                ),
                blocked_by=(
                    None
                    if "blocked_by" not in arguments
                    else tuple(arguments["blocked_by"])
                ),
            )
        except (ValueError, RuntimeError) as error:
            return _json_error("task_update_failed", str(error))
        return _json_success(
            {"ok": True, "task": task.to_dict()},
            "团队任务已更新",
        )


class TeamTaskDeleteTool:
    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_task_delete",
            "软删除没有下游依赖的 pending 共享任务，并保留审计记录。",
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
            raise ToolExecutionError("team_task_delete 只接受 task_id。")
        task_id = _required_text(arguments, "task_id")
        return _workflow_preparation(
            self.actor,
            {"task_id": task_id},
            f"team:{self.actor.team_id}:tasks:{task_id}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("task_delete_failed", str(error))

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            task = self.service.delete_task(
                self.actor,
                str(arguments["task_id"]),
            )
        except (ValueError, RuntimeError) as error:
            return _json_error("task_delete_failed", str(error))
        return _json_success(
            {"ok": True, "task": task.to_dict()},
            "团队任务已软删除",
        )


class TeamMessageSendTool:
    _FIELDS = {"to", "body", "summary", "message_type", "payload"}
    _PUBLIC_TYPES = {
        MessageType.TEXT,
        MessageType.TASK_EVENT,
        MessageType.SHUTDOWN_REQUEST,
        MessageType.SHUTDOWN_RESPONSE,
    }

    def __init__(
        self,
        service: TeamService,
        actor: ActorContext,
        *,
        delivery_notifier: Callable[[tuple[object, ...]], None] | None = None,
    ) -> None:
        self.service = service
        self.actor = actor
        self.delivery_notifier = delivery_notifier

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_message_send",
            "向当前 Team 的成员发送持久消息；to='*' 时广播给除自己外的成员。",
            {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "body": {"type": "string"},
                    "summary": {"type": "string"},
                    "message_type": {
                        "type": "string",
                        "enum": sorted(item.value for item in self._PUBLIC_TYPES),
                    },
                    "payload": {"type": "object"},
                },
                "required": ["to", "body", "summary"],
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
                f"team_message_send 收到未知参数：{', '.join(sorted(unknown))}"
            )
        to = _required_text(arguments, "to")
        body = _required_text(arguments, "body")
        summary = _required_text(arguments, "summary")
        try:
            message_type = MessageType(
                str(arguments.get("message_type", MessageType.TEXT.value))
            )
        except ValueError as error:
            raise ToolExecutionError("message_type 无效。") from error
        if message_type not in self._PUBLIC_TYPES:
            raise ToolExecutionError("该结构化消息类型只能由 host 工作流生成。")
        payload = arguments.get("payload")
        if payload is not None and not isinstance(payload, Mapping):
            raise ToolExecutionError("payload 必须是对象。")
        normalized = {
            "to": to,
            "body": body,
            "summary": summary,
            "message_type": message_type.value,
            "payload": None if payload is None else dict(payload),
        }
        return _workflow_preparation(
            self.actor,
            normalized,
            f"team:{self.actor.team_id}:mailbox",
        )

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("message_send_failed", str(error))

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        try:
            messages = self.service.send_message(
                self.actor,
                to=str(arguments["to"]),
                body=str(arguments["body"]),
                summary=str(arguments["summary"]),
                message_type=MessageType(str(arguments["message_type"])),
                payload=arguments["payload"],
            )
        except (ValueError, RuntimeError) as error:
            return _json_error("message_send_failed", str(error))
        wake_warning = None
        if self.delivery_notifier is not None:
            try:
                self.delivery_notifier(messages)
            except Exception:
                wake_warning = "消息已持久投递，但目标成员自动唤醒失败。"
        return _json_success(
            {
                "ok": True,
                "delivered_to": [message.recipient_id for message in messages],
                "message_ids": [message.message_id for message in messages],
                "wake_warning": wake_warning,
            },
            f"消息已投递给 {len(messages)} 名成员",
        )


class TeamInboxListTool:
    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_inbox_list",
            "读取当前成员的持久邮箱。默认只返回未读消息，不会自动改变已读状态。",
            {
                "type": "object",
                "properties": {"unread_only": {"type": "boolean"}},
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) - {"unread_only"}:
            raise ToolExecutionError("team_inbox_list 只接受 unread_only 参数。")
        unread_only = arguments.get("unread_only", True)
        if not isinstance(unread_only, bool):
            raise ToolExecutionError("unread_only 必须是布尔值。")
        return ToolPreparation(
            freeze_arguments({"unread_only": unread_only}),
            f"team:{self.actor.team_id}:mailbox:{self.actor.member_id}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        try:
            prepared = self.prepare(arguments)
            return self.execute_prepared(prepared.arguments)
        except (ToolExecutionError, ValueError, RuntimeError) as error:
            return _json_error("inbox_list_failed", str(error))

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            messages = self.service.list_inbox(
                self.actor,
                unread_only=bool(arguments["unread_only"]),
            )
        except (ValueError, RuntimeError) as error:
            return _json_error("inbox_list_failed", str(error))
        return _json_success(
            {"ok": True, "messages": [message.to_dict() for message in messages]},
            f"读取到 {len(messages)} 条团队消息",
        )


class TeamPlanSubmitTool:
    _FIELDS = {"task_id", "plan", "summary"}

    def __init__(self, service: TeamService, actor: ActorContext) -> None:
        self.service = service
        self.actor = actor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "team_plan_submit",
            "为分配给自己的任务提交结构化计划审批请求。提交后停止实施并等待 Lead 回复。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "plan": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["task_id", "plan", "summary"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != self._FIELDS:
            raise ToolExecutionError(
                "team_plan_submit 必须且只能包含 task_id、plan、summary。"
            )
        normalized = {key: _required_text(arguments, key) for key in self._FIELDS}
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
            return _json_error("plan_submit_failed", str(error))

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event=None
    ) -> ToolExecution:
        del cancel_event
        try:
            request = self.service.submit_plan(
                self.actor,
                task_id=str(arguments["task_id"]),
                plan=str(arguments["plan"]),
                summary=str(arguments["summary"]),
            )
        except (ValueError, RuntimeError) as error:
            return _json_error("plan_submit_failed", str(error))
        return _json_success(
            {"ok": True, "plan_request": request},
            "计划已提交，等待 Lead 审批",
        )
