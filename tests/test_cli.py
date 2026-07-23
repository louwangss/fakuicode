from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console


def test_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    from fakuicode.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "Fakuicode" in capsys.readouterr().out


def make_renderer() -> tuple[object, StringIO]:
    from fakuicode.renderer import Renderer

    output = StringIO()
    return Renderer(Console(file=output, force_terminal=False, color_system=None)), output


def test_invalid_configuration_returns_nonzero_without_echoing_key() -> None:
    from fakuicode.cli import main
    from fakuicode.errors import ConfigurationError

    renderer, output = make_renderer()
    api_key = "test-key-must-not-leak"

    def failing_loader(path: object) -> object:
        del path
        raise ConfigurationError("Invalid configuration")

    result = main(["--config", "invalid.yaml"], renderer=renderer, config_loader=failing_loader)

    assert result == 2
    assert api_key not in output.getvalue()


def test_valid_configuration_starts_tui_without_running_a_provider_request() -> None:
    from fakuicode.cli import main
    from fakuicode.models import ProviderConfig

    class FakeApp:
        def __init__(self) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

    app = FakeApp()
    config = ProviderConfig("openai", "test-model", "https://api.example.test/v1", "test-key")

    result = main(
        [],
        config_loader=lambda path: config,
        provider_factory=lambda loaded_config: object(),
        app_factory=lambda loaded_config, provider: app,
    )

    assert result == 0
    assert app.ran is True


def test_default_cli_creates_its_private_conversation_store(tmp_path, monkeypatch) -> None:
    from fakuicode.cli import main
    from fakuicode.models import ProfileSet, ProviderConfig

    created_paths = []

    class FakeStore:
        def __init__(self, path) -> None:
            created_paths.append(path)

    captured = {}

    class FakeApp:
        def __init__(
            self,
            config,
            *,
            provider,
            store,
            profile_name,
            profiles,
            workspace,
            permission_snapshot,
            permission_repository,
            mcp_snapshot,
            mcp_trust_repository,
            hook_snapshot,
            hook_repository,
            hook_trust_repository,
            instruction_loader,
            memory_service,
            provider_factory,
            skill_user_root,
            agent_user_root,
            skill_trust_repository,
            ) -> None:
            self.ran = False
            assert store is not None
            assert profile_name == "fast"
            assert profiles.active is config
            captured["workspace"] = workspace
            captured["snapshot"] = permission_snapshot
            captured["repository"] = permission_repository
            captured["mcp_snapshot"] = mcp_snapshot
            captured["mcp_trust_repository"] = mcp_trust_repository
            captured["hook_snapshot"] = hook_snapshot
            captured["hook_repository"] = hook_repository
            captured["hook_trust_repository"] = hook_trust_repository
            captured["instruction_loader"] = instruction_loader
            captured["memory_service"] = memory_service
            captured["provider_factory"] = provider_factory
            captured["skill_user_root"] = skill_user_root
            captured["agent_user_root"] = agent_user_root
            captured["skill_trust_repository"] = skill_trust_repository

        def run(self) -> None:
            self.ran = True

    config = ProviderConfig("openai", "test-model", "https://api.example.test/v1", "test-key")
    monkeypatch.setattr("fakuicode.cli.FakuicodeApp", FakeApp)
    result = main(
        [],
        config_loader=lambda path: ProfileSet({"fast": config}, "fast"),
        provider_factory=lambda loaded_config: object(),
        store_factory=FakeStore,
        store_path=tmp_path / "private.sqlite3",
        workspace=tmp_path / "workspace",
        permission_home=tmp_path / "home",
        app_factory=None,
    )

    assert result == 0
    assert created_paths == [tmp_path / "private.sqlite3"]
    assert captured["workspace"] == (tmp_path / "workspace").resolve()
    assert captured["instruction_loader"].workspace == (tmp_path / "workspace").resolve()
    assert captured["snapshot"].project_trusted is False
    assert captured["repository"].paths.user == tmp_path / "home" / ".fakuicode" / "permissions.yaml"
    assert captured["memory_service"].repository.paths.root == tmp_path / "home" / ".fakuicode" / "memory"
    assert captured["memory_service"].maintenance_runner.provider_factory is captured["provider_factory"]
    assert not (tmp_path / "workspace" / ".fakuicode" / "memory").exists()
