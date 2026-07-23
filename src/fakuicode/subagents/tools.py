"""Stable provider tools for launching and controlling child agents."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from threading import Event
from threading import RLock
from time import monotonic

from fakuicode.errors import ToolExecutionError
from fakuicode.models import ToolDefinition
from fakuicode.subagents.catalog import AgentCatalog
from fakuicode.subagents.runtime import ChildRuntimeError, ChildRuntimeFactory
from fakuicode.subagents.tasks import TaskManager, TaskManagerError, TaskSnapshot
from fakuicode.tools.base import (
    ToolExecution,
    ToolPreparation,
    freeze_arguments,
)


_INSTANCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_AGENT_FIELDS = {
    "prompt",
    "description",
    "subagent_type",
    "profile",
    "run_in_background",
    "name",
}


class AgentTool:
    def __init__(
        self,
        catalog: AgentCatalog,
        runtime_factory: ChildRuntimeFactory,
        manager: TaskManager,
        *,
        inline_timeout_seconds: float = 60.0,
        background_enabled: bool = True,
    ) -> None:
        if inline_timeout_seconds <= 0:
            raise ValueError("inline_timeout_seconds must be positive")
        self.catalog = catalog
        self.runtime_factory = runtime_factory
        self.manager = manager
        self.inline_timeout_seconds = inline_timeout_seconds
        self.background_enabled = background_enabled
        self._inline_lock = RLock()
        self._inline_task_id: str | None = None
        self._detach_event: Event | None = None

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "agent",
            "把一个边界明确的任务委派给独立子 Agent。subagent_type 留空表示 Fork 当前上下文。",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "交给子 Agent 的任务"},
                    "description": {"type": "string", "description": "供界面展示的一句话说明"},
                    "subagent_type": {"type": "string", "description": "预定义角色名；留空走 Fork"},
                    "profile": {"type": "string", "description": "Profile 覆盖或 inherit"},
                    "run_in_background": {"type": "boolean", "description": "是否立即进入后台"},
                    "name": {"type": "string", "description": "本次子 Agent 的唯一可读名称"},
                },
                "required": ["prompt", "description"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        unknown = set(arguments) - _AGENT_FIELDS
        if unknown:
            raise ToolExecutionError(f"agent 收到未知参数：{', '.join(sorted(unknown))}")
        normalized: dict[str, object] = {
            "prompt": _required_text(arguments, "prompt"),
            "description": _required_text(arguments, "description"),
        }
        for key in ("subagent_type", "profile", "name"):
            value = arguments.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ToolExecutionError(f"{key} 必须是字符串")
                if value.strip():
                    normalized[key] = value.strip()
        background = arguments.get("run_in_background", False)
        if not isinstance(background, bool):
            raise ToolExecutionError("run_in_background 必须是布尔值")
        normalized["run_in_background"] = background
        name = normalized.get("name")
        if isinstance(name, str) and _INSTANCE_NAME.fullmatch(name) is None:
            raise ToolExecutionError("name 必须是 1-64 位字母、数字、下划线或连字符")
        target = str(normalized.get("subagent_type") or "fork")
        return ToolPreparation(freeze_arguments(normalized), f"subagent:{target}")

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        try:
            prepared = self.prepare(arguments)
        except ToolExecutionError as error:
            return _error("invalid_arguments", str(error))
        return self.execute_prepared(prepared.arguments, cancel_event=cancel_event)

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        role = arguments.get("subagent_type")
        is_fork = not isinstance(role, str)
        if is_fork and arguments.get("profile") not in {None, "inherit"}:
            return _error("fork_profile_override", "Fork 必须继承父 Agent 的 Profile")
        definition = None
        if not is_fork:
            try:
                definition = self.catalog.resolve(role)
            except KeyError:
                return _error("unknown_subagent_type", f"未知 subagent_type: {role}")
        if is_fork and not self.background_enabled:
            return _error("background_disabled", "后台任务已禁用，无法 Fork")
        try:
            if is_fork:
                session = self.runtime_factory.create_fork(
                    name=str(arguments["name"]) if "name" in arguments else None,
                )
            else:
                assert definition is not None
                session = self.runtime_factory.create_defined(
                    definition,
                    profile_override=(
                        str(arguments["profile"])
                        if arguments.get("profile") not in {None, "inherit"}
                        else None
                    ),
                    name=str(arguments["name"]) if "name" in arguments else None,
                )
            requested_background = is_fork or bool(arguments.get("run_in_background")) or (
                definition.background if definition is not None else False
            )
            background = requested_background and self.background_enabled
            task_id = self.manager.launch(
                session,
                str(arguments["prompt"]),
                str(arguments["description"]),
                notify_on_done=background,
            )
        except (ChildRuntimeError, TaskManagerError) as error:
            return _error("launch_failed", str(error))
        if background:
            return _success(
                {
                    "ok": True,
                    "mode": "background",
                    "status": "async_launched",
                    "task_id": task_id,
                },
                "子 Agent 已在后台启动",
            )
        detach_event = Event()
        with self._inline_lock:
            self._inline_task_id = task_id
            self._detach_event = detach_event
        try:
            snapshot = self._wait_inline(
                task_id,
                detach_event=detach_event,
                cancel_event=cancel_event,
            )
        finally:
            with self._inline_lock:
                self._inline_task_id = None
                self._detach_event = None
        if snapshot is None:
            if not self.background_enabled:
                self.manager.stop(task_id)
                return _error("inline_timeout", "子 Agent 前台执行超时且后台任务已禁用")
            self.manager.mark_background(task_id)
            manual = detach_event.is_set()
            return _success(
                {
                    "ok": True,
                    "mode": "background",
                    "status": "manually_backgrounded" if manual else "timed_out_to_background",
                    "task_id": task_id,
                },
                "子 Agent 已转入后台",
            )
        payload = {
            "ok": snapshot.status == "completed",
            "mode": "inline",
            "status": snapshot.status,
            "task_id": task_id,
            "result": snapshot.result,
        }
        if snapshot.error:
            payload["error"] = snapshot.error
        return ToolExecution(
            snapshot.status == "completed",
            _json(payload),
            "子 Agent 已完成" if snapshot.status == "completed" else "子 Agent 未成功完成",
        )

    def background_current(self) -> bool:
        if not self.background_enabled:
            return False
        with self._inline_lock:
            if self._inline_task_id is None or self._detach_event is None:
                return False
            self._detach_event.set()
            return True

    def _wait_inline(
        self,
        task_id: str,
        *,
        detach_event: Event,
        cancel_event: Event | None,
    ):
        deadline = monotonic() + self.inline_timeout_seconds
        while True:
            snapshot = self.manager.get(task_id)
            if snapshot is None:
                raise TaskManagerError(f"未知 task_id：{task_id}")
            if snapshot.status in {"completed", "failed", "cancelled"}:
                return snapshot
            if detach_event.is_set():
                return None
            if cancel_event is not None and cancel_event.is_set():
                self.manager.stop(task_id)
            remaining = deadline - monotonic()
            if remaining <= 0:
                return None
            self.manager.wait(
                task_id,
                timeout=min(0.05, remaining),
                cancel_event=cancel_event,
            )


class TaskListTool:
    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "task_list",
            "列出当前会话中的子 Agent 任务。",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if arguments:
            raise ToolExecutionError("task_list 不接受参数")
        return ToolPreparation(freeze_arguments({}), "subagent_tasks")

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del arguments, cancel_event
        tasks = [_task_payload(item, include_result=False) for item in self.manager.list()]
        return _success({"ok": True, "tasks": tasks}, "已列出子 Agent 任务")


class TaskGetTool:
    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return _task_id_definition("task_get", "读取一个子 Agent 任务的完整状态。")

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        task_id = _only_task_id(arguments)
        return ToolPreparation(freeze_arguments({"task_id": task_id}), f"subagent_task:{task_id}")

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        try:
            return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)
        except ToolExecutionError as error:
            return _error("invalid_arguments", str(error))

    def execute_prepared(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        snapshot = self.manager.get(str(arguments["task_id"]))
        if snapshot is None:
            return _error("unknown_task", f"未知 task_id：{arguments['task_id']}")
        return _success(
            {"ok": True, "task": _task_payload(snapshot, include_result=True)},
            "已读取子 Agent 任务",
        )


class TaskStopTool:
    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return _task_id_definition("task_stop", "请求取消一个正在运行的子 Agent 任务。")

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        task_id = _only_task_id(arguments)
        return ToolPreparation(freeze_arguments({"task_id": task_id}), f"subagent_task:{task_id}")

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        try:
            return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)
        except (ToolExecutionError, TaskManagerError) as error:
            return _error("task_stop_failed", str(error))

    def execute_prepared(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        requested = self.manager.stop(str(arguments["task_id"]))
        status = "cancellation_requested" if requested else "already_finished"
        return _success({"ok": True, "status": status}, "已处理子 Agent 取消请求")


class SendMessageTool:
    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "send_message",
            "向一个仍存活且空闲的已命名子 Agent 续派任务。",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["name", "message"],
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"name", "message"}:
            raise ToolExecutionError("send_message 只接受 name 和 message")
        name = _required_text(arguments, "name")
        message = _required_text(arguments, "message")
        return ToolPreparation(
            freeze_arguments({"name": name, "message": message}),
            f"subagent:{name}",
        )

    def execute(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        try:
            return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)
        except (ToolExecutionError, TaskManagerError) as error:
            return _error("send_message_failed", str(error))

    def execute_prepared(self, arguments: Mapping[str, object], *, cancel_event=None) -> ToolExecution:
        del cancel_event
        task_id = self.manager.send_message(str(arguments["name"]), str(arguments["message"]))
        return _success(
            {
                "ok": True,
                "mode": "background",
                "status": "async_launched",
                "task_id": task_id,
            },
            "已向子 Agent 续派任务",
        )


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError(f"{key} 必须是非空字符串")
    return value.strip()


def _only_task_id(arguments: Mapping[str, object]) -> str:
    if set(arguments) != {"task_id"}:
        raise ToolExecutionError("工具只接受 task_id")
    return _required_text(arguments, "task_id")


def _task_id_definition(name: str, description: str) -> ToolDefinition:
    return ToolDefinition(
        name,
        description,
        {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    )


def _task_payload(snapshot: TaskSnapshot, *, include_result: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": snapshot.id,
        "name": snapshot.name,
        "role": snapshot.role,
        "description": snapshot.description,
        "status": snapshot.status,
        "tool_count": snapshot.tool_count,
        "last_activity": snapshot.last_activity,
        "conversation_id": snapshot.conversation_id,
        "profile": snapshot.profile_name,
    }
    if include_result:
        payload["result"] = snapshot.result
        payload["error"] = snapshot.error
        payload["usage"] = (
            {
                "input_tokens": snapshot.usage.input_tokens,
                "output_tokens": snapshot.usage.output_tokens,
                "cache_read_tokens": snapshot.usage.cache_read_tokens,
                "cache_write_tokens": snapshot.usage.cache_write_tokens,
            }
            if snapshot.usage is not None
            else None
        )
    return payload


def _success(payload: Mapping[str, object], summary: str) -> ToolExecution:
    return ToolExecution(True, _json(payload), summary)


def _error(code: str, message: str) -> ToolExecution:
    return ToolExecution(
        False,
        _json({"ok": False, "error": {"code": code, "message": message}}),
        "子 Agent 工具调用失败",
    )


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
