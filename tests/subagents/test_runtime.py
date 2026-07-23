from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from fakuicode.models import (
    AgentMessage,
    AgentStreamEvent,
    ProfileSet,
    ProviderConfig,
    ToolCall,
    ToolDefinition,
)
from fakuicode.providers.base import AgentRequest
from fakuicode.permissions.config import PermissionConfigSnapshot
from fakuicode.permissions.manager import PermissionManager
from fakuicode.permissions.models import ApprovalChoice, PermissionMode
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.storage import ConversationStore
from fakuicode.subagents.models import (
    AgentDefinition,
    AgentSource,
    PermissionBehavior,
)
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


class TextProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.requests = []

    def stream_agent(
        self,
        messages: Sequence[object],
        tools: Sequence[object],
        *,
        request: object,
        cancel_event=None,
    ) -> Iterator[AgentStreamEvent]:
        del cancel_event
        self.requests.append(request)
        yield AgentStreamEvent("text_delta", "child result")
        yield AgentStreamEvent("completed")

    def cancel(self) -> None:
        pass


class RecordingApproval:
    def __init__(self) -> None:
        self.requests = []

    def request(self, request, *, cancel_event=None):
        del cancel_event
        self.requests.append(request)
        return ApprovalChoice.ONCE


def _config(model: str = "test-model") -> ProviderConfig:
    return ProviderConfig("anthropic", model, "https://example.test", "secret")


def _definition(
    tmp_path: Path,
    *,
    permission_mode: PermissionBehavior = PermissionBehavior.INHERIT,
) -> AgentDefinition:
    return AgentDefinition(
        "explore",
        "探索代码",
        "role sentinel",
        AgentSource.PROJECT,
        tmp_path / "explore.md",
        tools=("read_file", "find_files", "search_code"),
        permission_mode=permission_mode,
    )


def test_defined_runtime_starts_clean_with_role_prompt_and_filtered_tools(tmp_path: Path) -> None:
    from fakuicode.subagents.runtime import ChildRuntimeFactory

    store = ConversationStore(tmp_path / "store.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")
    profiles = ProfileSet({"default": _config()}, "default")
    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
        approval_handler=RecordingApproval(),
    )
    providers: list[TextProvider] = []

    def provider_factory(config: ProviderConfig) -> TextProvider:
        provider = TextProvider(config)
        providers.append(provider)
        return provider

    factory = ChildRuntimeFactory(
        store=store,
        parent_conversation_id=parent.id,
        workspace=tmp_path,
        profiles=profiles,
        active_profile_name="default",
        provider_factory=provider_factory,
        tool_registry_factory=lambda child_permissions: ToolRegistry(
            WorkspacePolicy(tmp_path),
            permission_manager=child_permissions,
        ),
        parent_permissions=permissions,
        approval_handler=RecordingApproval(),
        project_instructions="project sentinel",
    )

    child = factory.create_defined(_definition(tmp_path))
    outcome = child.run_to_completion("inspect the repository")

    assert outcome.status == "completed"
    assert outcome.text == "child result"
    assert [item.name for item in child.registry.definitions()] == [
        "read_file",
        "find_files",
        "search_code",
    ]
    assert providers[0].requests[0].messages[-1].content == "inspect the repository"
    supplement = providers[0].requests[0].system_supplement
    assert "project sentinel" in supplement
    assert "role sentinel" in supplement
    record = store.get_conversation(child.conversation_id)
    assert record.conversation_type == "agent"
    assert record.agent_name == "explore"


def test_dont_ask_child_denies_new_write_without_using_shared_approval(tmp_path: Path) -> None:
    from fakuicode.subagents.runtime import ChildRuntimeFactory

    shared_approval = RecordingApproval()
    permissions = PermissionManager(
        PermissionConfigSnapshot(mode=PermissionMode.DEFAULT),
        DangerousCommandGuard(tmp_path),
        approval_handler=shared_approval,
    )
    factory = ChildRuntimeFactory(
        store=None,
        parent_conversation_id=None,
        workspace=tmp_path,
        profiles=ProfileSet({"default": _config()}, "default"),
        active_profile_name="default",
        provider_factory=TextProvider,
        tool_registry_factory=lambda child_permissions: ToolRegistry(
            WorkspacePolicy(tmp_path),
            permission_manager=child_permissions,
        ),
        parent_permissions=permissions,
        approval_handler=shared_approval,
    )
    definition = AgentDefinition(
        "writer",
        "测试权限",
        "role",
        AgentSource.PROJECT,
        tmp_path / "writer.md",
        permission_mode=PermissionBehavior.DONT_ASK,
    )

    child = factory.create_defined(definition)
    result = child.registry.execute(
        ToolCall("write-1", "write_file", {"path": "blocked.txt", "content": "no"})
    )

    assert result.success is False
    assert shared_approval.requests == []
    assert not (tmp_path / "blocked.txt").exists()


def test_runtime_rejects_unknown_profile_override(tmp_path: Path) -> None:
    from fakuicode.subagents.runtime import ChildRuntimeFactory, ChildRuntimeError

    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
    )
    factory = ChildRuntimeFactory(
        store=None,
        parent_conversation_id=None,
        workspace=tmp_path,
        profiles=ProfileSet({"default": _config()}, "default"),
        active_profile_name="default",
        provider_factory=TextProvider,
        tool_registry_factory=lambda child_permissions: ToolRegistry(
            WorkspacePolicy(tmp_path),
            permission_manager=child_permissions,
        ),
        parent_permissions=permissions,
    )

    import pytest

    with pytest.raises(ChildRuntimeError, match="Profile"):
        factory.create_defined(_definition(tmp_path), profile_override="missing")


def test_fork_runtime_reuses_parent_prompt_and_message_prefix_but_removes_control_tools(
    tmp_path: Path,
) -> None:
    from fakuicode.subagents.runtime import ChildRuntimeFactory

    providers: list[TextProvider] = []

    def provider_factory(config: ProviderConfig) -> TextProvider:
        provider = TextProvider(config)
        providers.append(provider)
        return provider

    seed = AgentRequest(
        (
            AgentMessage("user", "parent question"),
            AgentMessage("assistant", "parent answer"),
        ),
        (
            ToolDefinition("read_file", "read", {"type": "object"}),
            ToolDefinition("agent", "delegate", {"type": "object"}),
            ToolDefinition("task_list", "tasks", {"type": "object"}),
        ),
        system_prompt="stable parent prompt",
        system_supplement="dynamic parent supplement",
        output_token_limit=777,
    )
    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
    )
    store = ConversationStore(tmp_path / "fork.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")
    factory = ChildRuntimeFactory(
        store=store,
        parent_conversation_id=parent.id,
        workspace=tmp_path,
        profiles=ProfileSet({"default": _config()}, "default"),
        active_profile_name="default",
        provider_factory=provider_factory,
        tool_registry_factory=lambda child_permissions: ToolRegistry(
            WorkspacePolicy(tmp_path),
            permission_manager=child_permissions,
        ),
        parent_permissions=permissions,
        parent_request_provider=lambda: seed,
    )

    child = factory.create_fork(name="fork-one")
    outcome = child.run_to_completion("inspect another module")

    assert outcome.status == "completed"
    request = providers[0].requests[0]
    assert request.system_prompt == seed.system_prompt
    assert request.system_supplement == seed.system_supplement
    assert request.output_token_limit == seed.output_token_limit
    assert request.messages[: len(seed.messages)] == seed.messages
    assert "inspect another module" in request.messages[-1].content
    assert "不要启动其他子 Agent" in request.messages[-1].content
    assert [tool.name for tool in request.tools] == ["read_file"]
    assert child.role == "fork"
    child_events = store.load_events(child.conversation_id)
    assert any(event.kind == "user" for event in child_events)
    assert all(event.content != "parent question" for event in child_events)


def test_fork_runtime_requires_a_successful_parent_request(tmp_path: Path) -> None:
    from fakuicode.subagents.runtime import ChildRuntimeFactory, ChildRuntimeError

    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
    )
    factory = ChildRuntimeFactory(
        store=None,
        parent_conversation_id=None,
        workspace=tmp_path,
        profiles=ProfileSet({"default": _config()}, "default"),
        active_profile_name="default",
        provider_factory=TextProvider,
        tool_registry_factory=lambda child_permissions: ToolRegistry(
            WorkspacePolicy(tmp_path),
            permission_manager=child_permissions,
        ),
        parent_permissions=permissions,
        parent_request_provider=lambda: None,
    )

    import pytest

    with pytest.raises(ChildRuntimeError, match="成功请求"):
        factory.create_fork()
