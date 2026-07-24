from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread

from fakuicode.subagents.models import AgentDefinition, AgentSource
from fakuicode.subagents.runtime import ChildRunResult


class ImmediateSession:
    def __init__(self, name: str) -> None:
        self.id = f"session-{name}"
        self.name = name
        self.role = "explore"
        self.profile_name = "default"
        self.conversation_id = f"conversation-{name}"
        self.execution = {"isolation": "shared"}

    def run_to_completion(self, prompt: str, *, event_sink=None) -> ChildRunResult:
        del event_sink
        return ChildRunResult(f"answer:{prompt}", "completed")

    def cancel(self) -> None:
        pass

    def close(self, *, status: str = "completed") -> None:
        del status


class RuntimeFactory:
    def __init__(self) -> None:
        self.created = []

    def create_defined(self, definition, *, profile_override=None, name=None, isolation=None):
        self.created.append((definition.name, profile_override, name, isolation))
        session = ImmediateSession(name or definition.name)
        if isolation == "worktree":
            session.execution = {
                "isolation": "worktree",
                "branch": "worktree/role-explore/session",
                "workspace": "C:/child",
                "base_sha": "a" * 40,
                "status": "active",
            }
        return session

    def create_fork(self, *, name=None, isolation=None):
        self.created.append(("fork", None, name, isolation))
        session = ImmediateSession(name or "fork")
        if isolation == "worktree":
            session.execution = {
                "isolation": "worktree",
                "branch": "worktree/fork/session",
                "workspace": "C:/child",
                "base_sha": "a" * 40,
                "status": "active",
            }
        return session


class BlockingSession(ImmediateSession):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started = Event()
        self.release = Event()

    def run_to_completion(self, prompt: str, *, event_sink=None) -> ChildRunResult:
        del event_sink
        self.started.set()
        self.release.wait(timeout=2)
        return ChildRunResult(f"answer:{prompt}", "completed")

    def cancel(self) -> None:
        self.release.set()


class BlockingRuntimeFactory(RuntimeFactory):
    def __init__(self) -> None:
        super().__init__()
        self.session = BlockingSession("blocking")

    def create_defined(self, definition, *, profile_override=None, name=None, isolation=None):
        self.created.append((definition.name, profile_override, name, isolation))
        return self.session


def _catalog(tmp_path: Path):
    from fakuicode.subagents.catalog import AgentCatalog

    definition = AgentDefinition(
        "explore",
        "探索代码",
        "role",
        AgentSource.BUILTIN,
        tmp_path / "explore.md",
    )
    return AgentCatalog({"explore": definition})


def test_agent_tool_has_stable_schema_and_returns_inline_result(tmp_path: Path) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    runtime = RuntimeFactory()
    tool = AgentTool(_catalog(tmp_path), runtime, manager, inline_timeout_seconds=1)

    prepared = tool.prepare(
        {
            "prompt": "inspect",
            "description": "inspect repository",
            "subagent_type": "explore",
            "profile": "cheap",
            "name": "research",
        }
    )
    result = tool.execute_prepared(prepared.arguments)
    payload = json.loads(result.output)

    assert tool.definition.name == "agent"
    assert set(tool.definition.input_schema["properties"]) == {
        "prompt",
        "description",
        "subagent_type",
        "profile",
        "run_in_background",
            "name",
            "isolation",
    }
    assert payload == {
        "ok": True,
        "mode": "inline",
        "status": "completed",
        "task_id": payload["task_id"],
        "result": "answer:inspect",
        "execution": {"isolation": "shared"},
    }
    assert runtime.created == [("explore", "cheap", "research", None)]
    manager.close()


def test_agent_tool_returns_structured_error_for_unknown_role(tmp_path: Path) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    tool = AgentTool(_catalog(tmp_path), RuntimeFactory(), manager)

    result = tool.execute(
        {
            "prompt": "inspect",
            "description": "inspect repository",
            "subagent_type": "missing",
        }
    )

    assert result.success is False
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "unknown_subagent_type"
    assert "missing" in payload["error"]["message"]
    manager.close()


def test_agent_tool_background_launch_returns_before_result(tmp_path: Path) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    tool = AgentTool(_catalog(tmp_path), RuntimeFactory(), manager)

    result = tool.execute(
        {
            "prompt": "inspect",
            "description": "inspect repository",
            "subagent_type": "explore",
            "run_in_background": True,
        }
    )
    payload = json.loads(result.output)

    assert payload["ok"] is True
    assert payload["mode"] == "background"
    assert payload["status"] == "async_launched"
    assert result.metadata is not None
    assert result.metadata["finish_agent_turn"] is True
    assert payload["task_id"] in str(result.metadata["finish_agent_turn_message"])
    assert manager.wait(payload["task_id"], timeout=1) is not None
    assert manager.drain_notifications() == (payload["task_id"],)
    manager.close()


def test_task_get_running_result_finishes_turn_instead_of_inviting_polling(
    tmp_path: Path,
) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import TaskGetTool

    manager = TaskManager(max_concurrent=1)
    session = BlockingSession("planner")
    task_id = manager.launch(
        session,
        "make a plan",
        "planning",
        notify_on_done=True,
    )
    assert session.started.wait(timeout=1)

    result = TaskGetTool(manager).execute({"task_id": task_id})
    payload = json.loads(result.output)

    assert payload["task"]["status"] == "running"
    assert payload["task"]["poll_again"] is False
    assert result.metadata is not None
    assert result.metadata["finish_agent_turn"] is True
    assert task_id in str(result.metadata["finish_agent_turn_message"])

    session.release.set()
    assert manager.wait(task_id, timeout=1) is not None
    manager.close()


def test_agent_tool_fork_is_always_background_and_rejects_profile_override(
    tmp_path: Path,
) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    runtime = RuntimeFactory()
    tool = AgentTool(_catalog(tmp_path), runtime, manager)

    rejected = tool.execute(
        {
            "prompt": "inspect",
            "description": "fork inspection",
            "profile": "cheap",
        }
    )
    assert json.loads(rejected.output)["error"]["code"] == "fork_profile_override"

    launched = tool.execute(
        {
            "prompt": "inspect",
            "description": "fork inspection",
            "run_in_background": False,
            "name": "fork-one",
        }
    )
    payload = json.loads(launched.output)
    assert payload["mode"] == "background"
    assert payload["status"] == "async_launched"
    assert runtime.created == [("fork", None, "fork-one", None)]
    assert manager.wait(payload["task_id"], timeout=1) is not None
    manager.close()


def test_agent_tool_can_strengthen_a_defined_or_fork_session_to_worktree(
    tmp_path: Path,
) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    runtime = RuntimeFactory()
    tool = AgentTool(_catalog(tmp_path), runtime, manager, inline_timeout_seconds=1)

    defined = tool.execute(
        {
            "prompt": "inspect",
            "description": "isolated role",
            "subagent_type": "explore",
            "isolation": "worktree",
        }
    )
    defined_payload = json.loads(defined.output)
    fork = tool.execute(
        {
            "prompt": "inspect",
            "description": "isolated fork",
            "isolation": "worktree",
        }
    )
    fork_payload = json.loads(fork.output)

    assert defined_payload["execution"] == {
        "isolation": "worktree",
        "branch": "worktree/role-explore/session",
    }
    assert fork_payload["execution"] == {
        "isolation": "worktree",
        "branch": "worktree/fork/session",
    }
    assert runtime.created == [
        ("explore", None, None, "worktree"),
        ("fork", None, None, "worktree"),
    ]
    manager.close()


def test_background_switch_forces_defined_agents_inline_and_rejects_fork(
    tmp_path: Path,
) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    runtime = RuntimeFactory()
    tool = AgentTool(
        _catalog(tmp_path),
        runtime,
        manager,
        inline_timeout_seconds=1,
        background_enabled=False,
    )

    defined = tool.execute(
        {
            "prompt": "inspect",
            "description": "defined inspection",
            "subagent_type": "explore",
            "run_in_background": True,
        }
    )
    assert json.loads(defined.output)["mode"] == "inline"

    fork = tool.execute(
        {
            "prompt": "inspect",
            "description": "fork inspection",
        }
    )
    assert json.loads(fork.output)["error"]["code"] == "background_disabled"
    manager.close()


def test_inline_agent_can_detach_to_background_without_restarting(tmp_path: Path) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    runtime = BlockingRuntimeFactory()
    tool = AgentTool(
        _catalog(tmp_path),
        runtime,
        manager,
        inline_timeout_seconds=1,
    )
    result_holder = []
    worker = Thread(
        target=lambda: result_holder.append(
            tool.execute(
                {
                    "prompt": "inspect",
                    "description": "manual background",
                    "subagent_type": "explore",
                }
            )
        )
    )
    worker.start()
    assert runtime.session.started.wait(timeout=1)

    assert tool.background_current() is True
    worker.join(timeout=1)
    payload = json.loads(result_holder[0].output)
    assert payload["status"] == "manually_backgrounded"

    runtime.session.release.set()
    assert manager.wait(payload["task_id"], timeout=1) is not None
    assert manager.drain_notifications() == (payload["task_id"],)
    manager.close()


def test_inline_timeout_adopts_the_same_running_task(tmp_path: Path) -> None:
    from fakuicode.subagents.tasks import TaskManager
    from fakuicode.subagents.tools import AgentTool

    manager = TaskManager(max_concurrent=1)
    runtime = BlockingRuntimeFactory()
    tool = AgentTool(
        _catalog(tmp_path),
        runtime,
        manager,
        inline_timeout_seconds=0.01,
    )

    result = tool.execute(
        {
            "prompt": "inspect",
            "description": "automatic background",
            "subagent_type": "explore",
        }
    )
    payload = json.loads(result.output)
    assert payload["status"] == "timed_out_to_background"

    runtime.session.release.set()
    snapshot = manager.wait(payload["task_id"], timeout=1)
    assert snapshot is not None and snapshot.status == "completed"
    assert manager.drain_notifications() == (payload["task_id"],)
    manager.close()
