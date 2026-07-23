from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from fakuicode.models import AgentMessage, AgentStreamEvent, ProfileSet, ProviderConfig
from fakuicode.session import AgentSessionController
from fakuicode.skills import SkillDiscovery, SkillManager
from fakuicode.skills.isolated import IsolatedSkillExecutor, select_recent_user_turns
from fakuicode.storage import ConversationStore
from fakuicode.tools.base import ToolExecution
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


def test_select_recent_user_turns_preserves_assistant_tool_groups() -> None:
    from fakuicode.models import ToolCall, ToolResult

    messages = [
        AgentMessage("user", "old"),
        AgentMessage("assistant", tool_calls=(ToolCall("1", "read_file", {"path": "x"}),)),
        AgentMessage("user", tool_results=(ToolResult("1", "read_file", True, "x", "read"),)),
        AgentMessage("assistant", "old answer"),
        AgentMessage("user", "new"),
        AgentMessage("assistant", "new answer"),
    ]

    selected = select_recent_user_turns(messages, 1)

    assert [message.content for message in selected] == ["new", "new answer"]
    assert select_recent_user_turns(messages, 0) == ()


def _isolated_skill(tmp_path: Path):
    root = tmp_path / "skills"
    package = root / "test"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\n"
        "name: test\n"
        "description: test workflow\n"
        "fakuicode:\n"
        "  execution: isolated\n"
        "  history-turns: 1\n"
        "  profile: inherit\n"
        "  visible-tools: [read_file]\n"
        "---\n"
        "Run tests for $ARGUMENTS.\n",
        encoding="utf-8",
    )
    return SkillDiscovery(root, tmp_path / "user", tmp_path / "builtin").refresh({"read_file"}).skills["test"]


def test_isolated_executor_persists_hidden_child_and_returns_final_response(tmp_path: Path) -> None:
    requests = []

    class Provider:
        def __init__(self, config):
            self.config = config

        def stream_agent(self, messages: Sequence[AgentMessage], tools, *, request=None, cancel_event=None) -> Iterator[AgentStreamEvent]:
            requests.append(request)
            yield AgentStreamEvent("text_delta", "2 passed")
            yield AgentStreamEvent("completed")

        def cancel(self) -> None:
            pass

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")
    profiles = ProfileSet(
        {"default": ProviderConfig("openai", "mock", "https://example.test", "secret", context_window=128_000)},
        "default",
    )
    executor = IsolatedSkillExecutor(
        store=store,
        parent_conversation_id=parent.id,
        workspace=tmp_path,
        profiles=profiles,
        active_profile_name="default",
        parent_messages=lambda: (
            AgentMessage("user", "old"),
            AgentMessage("assistant", "old answer"),
            AgentMessage("user", "current"),
            AgentMessage("assistant", "current answer"),
        ),
        provider_factory=Provider,
        tool_registry_factory=lambda: ToolRegistry(WorkspacePolicy(tmp_path)),
    )

    result = executor.run(_isolated_skill(tmp_path), "changed files")

    assert result.success is True
    assert result.output == "2 passed"
    assert len(store.list_conversations()) == 1
    children = store.child_conversation_ids(parent.id)
    assert len(children) == 1
    child = store.get_conversation(children[0])
    assert child.conversation_type == "skill"
    assert child.status == "completed"
    assert [message.content for message in requests[0].messages[:2]] == ["current", "current answer"]
    assert "Run tests for changed files." in requests[0].system_supplement
    assert f"包根目录：{(tmp_path / 'skills' / 'test').resolve()}" in requests[0].system_supplement
    assert all(tool.name not in {"load_skill", "install_skill"} for tool in requests[0].tools)


def test_isolated_executor_marks_unexpected_provider_failure_as_error(tmp_path: Path) -> None:
    class BrokenProvider:
        def __init__(self, config):
            raise RuntimeError("provider setup failed")

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")
    profiles = ProfileSet(
        {"default": ProviderConfig("openai", "mock", "https://example.test", "secret")},
        "default",
    )
    executor = IsolatedSkillExecutor(
        store=store,
        parent_conversation_id=parent.id,
        workspace=tmp_path,
        profiles=profiles,
        active_profile_name="default",
        parent_messages=tuple,
        provider_factory=BrokenProvider,
        tool_registry_factory=lambda: ToolRegistry(WorkspacePolicy(tmp_path)),
    )

    result = executor.run(_isolated_skill(tmp_path), "")

    child = store.get_conversation(store.child_conversation_ids(parent.id)[0])
    assert result.success is False
    assert result.metadata is not None and result.metadata["status"] == "error"
    assert child.status == "error"


def test_isolated_executor_propagates_provider_cancellation(tmp_path: Path) -> None:
    from fakuicode.errors import RequestCancelled

    class CancelledProvider:
        def __init__(self, config):
            self.config = config

        def stream_agent(self, messages, tools, *, request=None, cancel_event=None):
            raise RequestCancelled()
            yield

        def cancel(self) -> None:
            pass

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")
    profiles = ProfileSet(
        {"default": ProviderConfig("openai", "mock", "https://example.test", "secret")},
        "default",
    )
    executor = IsolatedSkillExecutor(
        store=store,
        parent_conversation_id=parent.id,
        workspace=tmp_path,
        profiles=profiles,
        active_profile_name="default",
        parent_messages=tuple,
        provider_factory=CancelledProvider,
        tool_registry_factory=lambda: ToolRegistry(WorkspacePolicy(tmp_path)),
    )

    result = executor.run(_isolated_skill(tmp_path), "")

    child = store.get_conversation(store.child_conversation_ids(parent.id)[0])
    assert result.metadata is not None and result.metadata["status"] == "cancelled"
    assert child.status == "cancelled"


def test_direct_isolated_cancellation_propagates_to_parent_stream(tmp_path: Path) -> None:
    class Provider:
        config = ProviderConfig("openai", "mock", "https://example.test", "secret")

        def cancel(self) -> None:
            pass

    skill = _isolated_skill(tmp_path)
    registry = ToolRegistry(WorkspacePolicy(tmp_path))
    manager = SkillManager(
        SkillDiscovery(skill.package_path.parent, tmp_path / "user", tmp_path / "builtin"),
        registry,
        context_window=128_000,
        isolated_runner=lambda name, arguments, cancel: ToolExecution(
            False,
            "独立 Skill 已取消。",
            "cancelled",
            metadata={"child_conversation_id": "child-1", "skill": name, "profile": "default", "status": "cancelled"},
        ),
    )
    manager.refresh()
    store = ConversationStore(tmp_path / "parent.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")
    session = AgentSessionController(
        Provider(),
        registry,
        store=store,
        conversation_id=parent.id,
        skill_manager=manager,
    )

    events = list(session.send("/test", skill_invocation=("test", None)))

    assert events[-1].kind == "cancelled"
    assistant = [event for event in store.load_events(parent.id) if event.kind == "assistant"][-1]
    assert assistant.metadata is not None and assistant.metadata["status"] == "cancelled"


def test_direct_isolated_skill_is_blocked_by_current_plan_mode(tmp_path: Path) -> None:
    class Provider:
        config = ProviderConfig("openai", "mock", "https://example.test", "secret")

        def cancel(self) -> None:
            pass

    skill = _isolated_skill(tmp_path)
    registry = ToolRegistry(WorkspacePolicy(tmp_path))
    calls: list[str] = []
    manager = SkillManager(
        SkillDiscovery(skill.package_path.parent, tmp_path / "user", tmp_path / "builtin"),
        registry,
        context_window=128_000,
        isolated_runner=lambda name, arguments, cancel: (
            calls.append(name) or ToolExecution(True, "unexpected", "unexpected")
        ),
    )
    manager.refresh()
    session = AgentSessionController(Provider(), registry, skill_manager=manager)
    session.enable_plan_mode()

    events = list(session.send("/test", skill_invocation=("test", None)))

    assert calls == []
    assert events[-1].kind == "error"
    assert "计划模式" in events[0].text
