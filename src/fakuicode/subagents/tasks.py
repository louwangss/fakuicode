"""In-memory ownership and lifecycle tracking for child-agent runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Condition, Event, RLock
from time import monotonic, time_ns
from typing import Literal, Mapping, Protocol
from uuid import uuid4

from fakuicode.models import TokenUsage
from fakuicode.subagents.runtime import ChildRunResult


TaskStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
]
_TERMINAL = {"completed", "failed", "cancelled"}


class ManagedChildSession(Protocol):
    id: str
    name: str
    role: str
    profile_name: str
    conversation_id: str
    execution: Mapping[str, object]

    def run_to_completion(self, prompt: str, *, event_sink=None) -> ChildRunResult: ...

    def cancel(self) -> None: ...

    def touch(self) -> None: ...

    def close(self, *, status: str = "completed") -> None: ...


class TaskManagerError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSnapshot:
    id: str
    session_id: str
    name: str
    role: str
    description: str
    status: TaskStatus
    result: str
    error: str | None
    start_time_ns: int
    end_time_ns: int | None
    usage: TokenUsage | None
    tool_count: int
    last_activity: str
    conversation_id: str
    profile_name: str
    notify_on_done: bool
    execution: Mapping[str, object]


@dataclass
class _TaskRun:
    id: str
    session: ManagedChildSession
    prompt: str
    description: str
    status: TaskStatus = "queued"
    result: str = ""
    error: str | None = None
    start_time_ns: int = 0
    end_time_ns: int | None = None
    usage: TokenUsage | None = None
    tool_count: int = 0
    last_activity: str = ""
    notify_on_done: bool = False


class TaskManager:
    def __init__(self, *, max_concurrent: int = 2) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="fakuicode-subagent",
        )
        self._tasks: dict[str, _TaskRun] = {}
        self._sessions: dict[str, ManagedChildSession] = {}
        self._names: dict[str, str] = {}
        self._done: SimpleQueue[str] = SimpleQueue()
        self._notifications: SimpleQueue[str] = SimpleQueue()
        self._lock = RLock()
        self._changed = Condition(self._lock)
        self._closed = False

    def launch(
        self,
        session: ManagedChildSession,
        prompt: str,
        description: str,
        *,
        notify_on_done: bool = False,
    ) -> str:
        if not prompt.strip() or not description.strip():
            raise TaskManagerError("任务 prompt 和 description 不能为空")
        superseded: ManagedChildSession | None = None
        with self._changed:
            if self._closed:
                raise TaskManagerError("后台任务管理器已经关闭")
            existing_session_id = self._names.get(session.name)
            if existing_session_id is not None and existing_session_id != session.id:
                if any(
                    task.session.id == existing_session_id
                    and task.status not in _TERMINAL
                    for task in self._tasks.values()
                ):
                    raise TaskManagerError(f"Agent 名称 '{session.name}' 已被占用")
                superseded = self._sessions.pop(existing_session_id, None)
            if any(
                task.session.id == session.id and task.status not in _TERMINAL
                for task in self._tasks.values()
            ):
                raise TaskManagerError(f"Agent '{session.name}' 已有任务正在运行")
            self._sessions[session.id] = session
            # The name is intentionally a weak routing reference: a later idle
            # session with the same name becomes the SendMessage target.
            self._names[session.name] = session.id
            task_id = f"task-{uuid4()}"
            task = _TaskRun(
                task_id,
                session,
                prompt.strip(),
                description.strip(),
                notify_on_done=notify_on_done,
            )
            self._tasks[task_id] = task
            self._executor.submit(self._run, task_id)
            self._changed.notify_all()
        if superseded is not None:
            try:
                superseded.close(status="completed")
            except Exception:
                pass
        return task_id

    def send_message(self, name: str, message: str) -> str:
        with self._lock:
            session_id = self._names.get(name)
            session = self._sessions.get(session_id) if session_id is not None else None
            if session is None:
                raise TaskManagerError(f"找不到仍存活的 Agent：{name}")
            touch = getattr(session, "touch", None)
            if callable(touch):
                touch()
            return self.launch(
                session,
                message,
                f"续派给 {name}",
                notify_on_done=True,
            )

    def get(self, task_id: str) -> TaskSnapshot | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return _snapshot(task) if task is not None else None

    def list(self) -> tuple[TaskSnapshot, ...]:
        with self._lock:
            return tuple(
                _snapshot(task)
                for task in sorted(
                    self._tasks.values(),
                    key=lambda item: (item.start_time_ns, item.id),
                    reverse=True,
                )
            )

    def wait(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
        cancel_event: Event | None = None,
    ) -> TaskSnapshot | None:
        deadline = None if timeout is None else monotonic() + max(0.0, timeout)
        with self._changed:
            while True:
                task = self._tasks.get(task_id)
                if task is None:
                    raise TaskManagerError(f"未知 task_id：{task_id}")
                if task.status in _TERMINAL:
                    return _snapshot(task)
                if cancel_event is not None and cancel_event.is_set():
                    self.stop(task_id)
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._changed.wait(
                    0.05 if remaining is None else min(0.05, remaining)
                )

    def stop(self, task_id: str) -> bool:
        with self._changed:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskManagerError(f"未知 task_id：{task_id}")
            if task.status in _TERMINAL:
                return False
            task.status = "cancelling"
            task.session.cancel()
            self._changed.notify_all()
            return True

    def stop_by_name(self, name: str) -> bool:
        with self._lock:
            task = next(
                (
                    item
                    for item in reversed(tuple(self._tasks.values()))
                    if item.session.name == name and item.status not in _TERMINAL
                ),
                None,
            )
        return self.stop(task.id) if task is not None else False

    def mark_waiting_approval(self, name: str, waiting: bool) -> bool:
        with self._changed:
            task = next(
                (
                    item
                    for item in reversed(tuple(self._tasks.values()))
                    if item.session.name == name and item.status not in _TERMINAL
                ),
                None,
            )
            if task is None:
                return False
            if waiting and task.status == "running":
                task.status = "waiting_approval"
            elif not waiting and task.status == "waiting_approval":
                task.status = "running"
            self._changed.notify_all()
            return True

    def mark_background(self, task_id: str) -> None:
        with self._changed:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskManagerError(f"未知 task_id：{task_id}")
            if task.notify_on_done:
                return
            task.notify_on_done = True
            if task.status in _TERMINAL:
                self._notifications.put(task.id)
            self._changed.notify_all()

    def drain_done(self) -> tuple[str, ...]:
        items: list[str] = []
        while True:
            try:
                items.append(self._done.get_nowait())
            except Empty:
                return tuple(items)

    def drain_notifications(self) -> tuple[str, ...]:
        items: list[str] = []
        while True:
            try:
                items.append(self._notifications.get_nowait())
            except Empty:
                return tuple(items)

    def close(self) -> None:
        with self._changed:
            if self._closed:
                return
            self._closed = True
            sessions = tuple(self._sessions.values())
            for task in self._tasks.values():
                if task.status not in _TERMINAL:
                    task.status = "cancelling"
                    task.session.cancel()
            self._changed.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=False)
        for session in sessions:
            latest = next(
                (
                    task.status
                    for task in reversed(tuple(self._tasks.values()))
                    if task.session.id == session.id
                ),
                "completed",
            )
            try:
                session.close(
                    status=(
                        "cancelled"
                        if latest == "cancelled"
                        else "error"
                        if latest == "failed"
                        else "completed"
                    )
                )
            except Exception:
                continue

    def _run(self, task_id: str) -> None:
        with self._changed:
            task = self._tasks[task_id]
            if task.status == "cancelling":
                task.status = "cancelled"
                task.end_time_ns = time_ns()
                self._finish(task)
                return
            task.status = "running"
            task.start_time_ns = time_ns()
            self._changed.notify_all()
        try:
            outcome = task.session.run_to_completion(task.prompt)
        except Exception:
            outcome = ChildRunResult(
                "",
                "failed",
                "子 Agent 运行时发生内部错误",
            )
        with self._changed:
            task.result = outcome.text
            task.error = outcome.error
            task.usage = outcome.usage
            task.tool_count = outcome.tool_count
            task.last_activity = outcome.last_activity
            task.status = outcome.status
            task.end_time_ns = time_ns()
            self._finish(task)

    def _finish(self, task: _TaskRun) -> None:
        self._done.put(task.id)
        if task.notify_on_done:
            self._notifications.put(task.id)
        self._changed.notify_all()


def _snapshot(task: _TaskRun) -> TaskSnapshot:
    return TaskSnapshot(
        task.id,
        task.session.id,
        task.session.name,
        task.session.role,
        task.description,
        task.status,
        task.result,
        task.error,
        task.start_time_ns,
        task.end_time_ns,
        task.usage,
        task.tool_count,
        task.last_activity,
        task.session.conversation_id,
        task.session.profile_name,
        task.notify_on_done,
        dict(getattr(task.session, "execution", {"isolation": "shared"})),
    )
