from __future__ import annotations

from pathlib import Path

from fakuicode.models import ToolCall
from fakuicode.skills import SkillDiscovery, SkillManager
from fakuicode.skills.install import SkillInstallResult
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


def _skill(root: Path, name: str, *, invocation: str = "auto", tools: str = "", body: str = "Do $ARGUMENTS") -> None:
    package = root / name
    package.mkdir(parents=True)
    visible = f"\n  visible-tools: [{tools}]" if tools else ""
    (package / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {name} workflow\n"
        "fakuicode:\n"
        f"  invocation: {invocation}{visible}\n"
        "  execution: shared\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def _manager(tmp_path: Path) -> tuple[SkillManager, ToolRegistry, Path]:
    skill_root = tmp_path / ".fakuicode" / "skills"
    _skill(skill_root, "one", tools="read_file")
    _skill(skill_root, "two", tools="find_files")
    _skill(skill_root, "manual", invocation="manual")
    registry = ToolRegistry(WorkspacePolicy(tmp_path))
    manager = SkillManager(
        SkillDiscovery(skill_root, tmp_path / "user", tmp_path / "builtin"),
        registry,
        context_window=128_000,
    )
    manager.refresh()
    return manager, registry, skill_root


def test_load_skill_is_system_visible_and_refuses_manual_model_activation(tmp_path: Path) -> None:
    manager, registry, _ = _manager(tmp_path)

    names = {definition.name for definition in registry.definitions()}
    denied = registry.execute(ToolCall("1", "load_skill", {"name": "manual", "arguments": ""}))

    assert "load_skill" in names
    assert denied.success is False
    assert "manual" in denied.output
    assert not manager.active


def test_install_skill_is_system_visible_and_routes_a_structured_request(tmp_path: Path) -> None:
    manager, registry, _ = _manager(tmp_path)
    requests = []

    class Installer:
        def install(self, request, *, confirm, cancel_event=None):
            del confirm, cancel_event
            requests.append(request)
            return SkillInstallResult(True, "frontend-design", "installed", tmp_path / "target")

    manager.installer = Installer()
    manager.install_confirmation = lambda preview, cancel: None

    result = registry.execute(
        ToolCall(
            "install",
            "install_skill",
            {
                "source": "https://www.skills.sh/anthropics/skills/frontend-design",
                "scope": "project",
                "preset": "coding",
                "replace": False,
            },
        )
    )

    assert result.success is True
    assert result.metadata == {
        "skill": "frontend-design",
        "path": str(tmp_path / "target"),
    }
    assert requests[0].preset.value == "coding"


def test_install_skill_is_hidden_and_execution_blocked_in_plan_mode(tmp_path: Path) -> None:
    manager, registry, _ = _manager(tmp_path)
    manager.set_mode("plan")

    visible = {definition.name for definition in registry.definitions(read_only_only=True)}
    result = registry.execute(
        ToolCall(
            "install",
            "install_skill",
            {"source": "https://github.com/acme/skills", "skill": "demo"},
        ),
        read_only_only=True,
    )

    assert "install_skill" not in visible
    assert result.success is False
    assert "计划模式" in result.output


def test_shared_activation_pins_rendered_sop_and_unions_visible_tools(tmp_path: Path) -> None:
    manager, registry, _ = _manager(tmp_path)

    first = registry.execute(ToolCall("1", "load_skill", {"name": "one", "arguments": "alpha"}))
    second = registry.execute(ToolCall("2", "load_skill", {"name": "two", "arguments": "beta"}))

    assert first.success and second.success
    assert [skill.name for skill in manager.active] == ["one", "two"]
    assert "Do alpha" in manager.active_prompt
    assert "Do beta" in manager.active_prompt
    assert f"包根目录：{(tmp_path / '.fakuicode' / 'skills' / 'one').resolve()}" in manager.active_prompt
    assert {item.name for item in registry.definitions()} == {
        "load_skill",
        "install_skill",
        "read_file",
        "find_files",
    }
    hidden = registry.execute(ToolCall("3", "write_file", {"path": "x", "content": "x"}))
    assert hidden.success is False
    assert "not visible" in hidden.output


def test_reinvocation_atomically_replaces_snapshot_and_validator_can_roll_back(tmp_path: Path) -> None:
    manager, registry, _ = _manager(tmp_path)
    assert manager.invoke("one", "old").success
    manager.activation_validator = lambda candidate: (_ for _ in ()).throw(ValueError("too large"))

    rejected = manager.invoke("one", "new")

    assert rejected.success is False
    assert [skill.arguments for skill in manager.active] == ["old"]
    assert "Do old" in manager.active_prompt


def test_activation_rolls_back_when_snapshot_persistence_fails(tmp_path: Path) -> None:
    manager, registry, _ = _manager(tmp_path)
    manager.on_activation = lambda active: (_ for _ in ()).throw(OSError("database unavailable"))

    rejected = manager.invoke("one", "new")

    assert rejected.success is False
    assert manager.active == ()
    assert {item.name for item in registry.definitions()} >= {"read_file", "write_file", "load_skill"}


def test_refresh_marks_changed_active_skill_stale_and_clear_restores_full_tools(tmp_path: Path) -> None:
    manager, registry, skill_root = _manager(tmp_path)
    assert manager.invoke("one", "x").success
    entry = skill_root / "one" / "SKILL.md"
    entry.write_text(entry.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    manager.refresh()

    assert manager.active[0].stale is True
    manager.clear()
    assert not manager.active
    assert {item.name for item in registry.definitions()} >= {"read_file", "write_file", "load_skill"}


def test_shared_activation_snapshot_restores_after_reopen_and_clear_removes_it(tmp_path: Path) -> None:
    from fakuicode.models import ProviderConfig
    from fakuicode.session import AgentSessionController
    from fakuicode.storage import ConversationStore

    class Provider:
        config = ProviderConfig("openai", "mock", "https://example.test", "secret")

        def cancel(self) -> None:
            pass

    skill_root = tmp_path / ".fakuicode" / "skills"
    _skill(skill_root, "one", tools="read_file")
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("Main", tmp_path, "default")

    first_registry = ToolRegistry(WorkspacePolicy(tmp_path))
    first_manager = SkillManager(
        SkillDiscovery(skill_root, tmp_path / "user", tmp_path / "builtin"),
        first_registry,
        context_window=128_000,
    )
    first_manager.refresh()
    first_session = AgentSessionController(
        Provider(),
        first_registry,
        store=store,
        conversation_id=conversation.id,
        skill_manager=first_manager,
    )
    assert first_manager.invoke("one", "persisted").success
    first_session.close()

    second_registry = ToolRegistry(WorkspacePolicy(tmp_path))
    second_manager = SkillManager(
        SkillDiscovery(skill_root, tmp_path / "user", tmp_path / "builtin"),
        second_registry,
        context_window=128_000,
    )
    second_manager.refresh()
    second_session = AgentSessionController(
        Provider(),
        second_registry,
        store=store,
        conversation_id=conversation.id,
        skill_manager=second_manager,
    )

    assert [item.arguments for item in second_manager.active] == ["persisted"]
    second_session.clear_context()
    assert second_manager.active == ()


def test_direct_activation_hard_limit_includes_the_pending_user_invocation(tmp_path: Path) -> None:
    from fakuicode.models import ProviderConfig
    from fakuicode.session import AgentSessionController

    class Provider:
        config = ProviderConfig(
            "openai",
            "mock",
            "https://example.test",
            "secret",
            context_window=128_000,
        )

        def cancel(self) -> None:
            pass

    manager, registry, _ = _manager(tmp_path)
    session = AgentSessionController(Provider(), registry, skill_manager=manager)
    oversized = "/one " + ("x" * 500_000)

    events = list(session.send(oversized, skill_invocation=("one", "")))

    assert manager.active == ()
    assert "硬输入上限" in events[0].text
