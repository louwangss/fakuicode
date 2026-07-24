from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import subprocess
import sys

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
from fakuicode.worktrees.manager import WorktreeManager


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


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Runtime Tests")
    _git(repo, "config", "user.email", "runtime@example.test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


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


def test_worktree_runtime_rebinds_tools_and_instructions_but_keeps_parent_storage(
    tmp_path: Path,
) -> None:
    from fakuicode.subagents.runtime import ChildRuntimeFactory

    repo = _repository(tmp_path)
    store = ConversationStore(tmp_path / "store.sqlite3")
    parent = store.create_conversation("Main", repo, "default")
    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(repo),
    )
    seen_contexts = []
    providers: list[TextProvider] = []

    def provider_factory(config: ProviderConfig) -> TextProvider:
        provider = TextProvider(config)
        providers.append(provider)
        return provider

    def registry_factory(child_permissions, execution_context):
        seen_contexts.append(execution_context)
        assert execution_context is not None
        return ToolRegistry(
            WorkspacePolicy(
                execution_context.execution_workspace,
                mappings=execution_context.mappings,
            ),
            permission_manager=child_permissions,
        )

    factory = ChildRuntimeFactory(
        store=store,
        parent_conversation_id=parent.id,
        workspace=repo,
        profiles=ProfileSet({"default": _config()}, "default"),
        active_profile_name="default",
        provider_factory=provider_factory,
        tool_registry_factory=registry_factory,
        parent_permissions=permissions,
        worktree_manager=WorktreeManager(repo),
        project_instruction_provider=lambda workspace: f"instructions:{workspace}",
    )
    definition = AgentDefinition(
        "explore",
        "探索代码",
        "role sentinel",
        AgentSource.PROJECT,
        repo / "explore.md",
        tools=("read_file",),
        isolation="worktree",
    )

    child = factory.create_defined(definition)
    outcome = child.run_to_completion("inspect")

    context = seen_contexts[0]
    assert outcome.status == "completed"
    assert child.registry.policy.workspace == context.execution_workspace
    assert context.execution_workspace != repo
    assert store.get_conversation(child.conversation_id).workspace == repo.resolve()
    request = providers[0].requests[0]
    assert f"instructions:{context.execution_workspace}" in request.system_supplement
    assert str(repo.resolve()) in request.system_supplement
    assert str(context.execution_workspace) in request.system_supplement
    assert child.execution["isolation"] == "worktree"
    assert child.execution["status"] == "active"

    worktree_root = context.worktree_root
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.filesystem import WriteFileTool

    WriteFileTool(child.registry.policy).execute(
        {"path": "README.md", "content": "child-only\n"}
    )
    command = RunCommandTool(child.registry.policy).execute(
        {
            "command": [
                sys.executable,
                "-c",
                "from pathlib import Path; print(Path.cwd())",
            ]
        }
    )
    assert "child-only" not in (repo / "README.md").read_text(encoding="utf-8")
    assert (worktree_root / "README.md").read_text(encoding="utf-8") == "child-only\n"
    assert str(context.execution_workspace) in command.output

    child.close()

    assert child.execution["status"] == "retained"
    assert worktree_root.exists()
