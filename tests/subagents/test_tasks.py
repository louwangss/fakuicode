from __future__ import annotations

from threading import Event

import pytest

from fakuicode.models import TokenUsage
from fakuicode.subagents.runtime import ChildRunResult


class FakeSession:
    def __init__(self, name: str, *, gate: Event | None = None) -> None:
        self.id = f"session-{name}"
        self.name = name
        self.role = "general-purpose"
        self.profile_name = "default"
        self.conversation_id = f"conversation-{name}"
        self.gate = gate
        self.prompts: list[str] = []
        self.cancelled = False
        self.closed = False

    def run_to_completion(self, prompt: str, *, event_sink=None) -> ChildRunResult:
        del event_sink
        self.prompts.append(prompt)
        if self.gate is not None:
            self.gate.wait(timeout=2)
        if self.cancelled:
            return ChildRunResult("", "cancelled", "cancelled")
        return ChildRunResult(
            f"result:{prompt}",
            "completed",
            usage=TokenUsage(10, 5),
            tool_count=2,
            last_activity="read_file",
        )

    def cancel(self) -> None:
        self.cancelled = True
        if self.gate is not None:
            self.gate.set()

    def close(self, *, status: str = "completed") -> None:
        del status
        self.closed = True
        self.cancel()


def test_task_manager_tracks_completion_and_keeps_session_for_followup() -> None:
    from fakuicode.subagents.tasks import TaskManager

    manager = TaskManager(max_concurrent=2)
    session = FakeSession("review")
    first_id = manager.launch(session, "first", "first task")

    first = manager.wait(first_id, timeout=1)
    assert first is not None
    assert first.status == "completed"
    assert first.result == "result:first"
    assert first.tool_count == 2
    assert first.last_activity == "read_file"

    second_id = manager.send_message("review", "second")
    second = manager.wait(second_id, timeout=1)
    assert second is not None and second.status == "completed"
    assert session.prompts == ["first", "second"]
    assert manager.drain_done() == (first_id, second_id)
    manager.close()


def test_task_manager_requests_cancellation_and_rejects_duplicate_live_names() -> None:
    from fakuicode.subagents.tasks import TaskManager, TaskManagerError

    manager = TaskManager(max_concurrent=1)
    gate = Event()
    session = FakeSession("worker", gate=gate)
    task_id = manager.launch(session, "wait", "blocking task")

    duplicate = FakeSession("worker")
    duplicate.id = "session-worker-duplicate"
    with pytest.raises(TaskManagerError, match="名称"):
        manager.launch(duplicate, "duplicate", "duplicate")

    assert manager.stop(task_id) is True
    snapshot = manager.wait(task_id, timeout=1)
    assert snapshot is not None
    assert snapshot.status == "cancelled"
    manager.close()


def test_task_manager_wait_timeout_detaches_without_restarting_task() -> None:
    from fakuicode.subagents.tasks import TaskManager

    manager = TaskManager(max_concurrent=1)
    gate = Event()
    session = FakeSession("slow", gate=gate)
    task_id = manager.launch(session, "slow work", "slow task")

    assert manager.wait(task_id, timeout=0.01) is None
    assert session.prompts == ["slow work"]
    gate.set()
    finished = manager.wait(task_id, timeout=1)

    assert finished is not None and finished.status == "completed"
    assert session.prompts == ["slow work"]
    manager.close()
