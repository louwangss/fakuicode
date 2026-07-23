from __future__ import annotations

import json
from pathlib import Path

from fakuicode.subagents.models import AgentDefinition, AgentSource
from fakuicode.subagents.runtime import ChildRunResult


class ImmediateSession:
    def __init__(self, name: str) -> None:
        self.id = f"session-{name}"
        self.name = name
        self.role = "explore"
        self.profile_name = "default"
        self.conversation_id = f"conversation-{name}"

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

    def create_defined(self, definition, *, profile_override=None, name=None):
        self.created.append((definition.name, profile_override, name))
        return ImmediateSession(name or definition.name)


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
    }
    assert payload == {
        "ok": True,
        "mode": "inline",
        "status": "completed",
        "task_id": payload["task_id"],
        "result": "answer:inspect",
    }
    assert runtime.created == [("explore", "cheap", "research")]
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
    assert manager.wait(payload["task_id"], timeout=1) is not None
    manager.close()

