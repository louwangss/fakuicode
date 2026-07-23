from __future__ import annotations

from pathlib import Path

import pytest


API_KEY = "test-api-key-must-not-leak"


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fakuicode.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def config_text(
    *, protocol: str = "anthropic", base_url: str = "https://api.example.test/v1", thinking: str = ""
) -> str:
    return f"""protocol: {protocol}
model: test-model
base_url: {base_url}
api_key: {API_KEY}
{thinking}"""


def test_loads_anthropic_configuration_with_enabled_thinking(tmp_path: Path) -> None:
    from fakuicode.config import load_config

    config = load_config(write_config(tmp_path, config_text(thinking="thinking:\n  enabled: true\n")))

    assert config.protocol == "anthropic"
    assert config.thinking is not None
    assert config.thinking.enabled is True


def test_load_profiles_accepts_named_profiles_and_a_selected_default(tmp_path: Path) -> None:
    from fakuicode.config import load_profiles

    profiles = load_profiles(
        write_config(
            tmp_path,
            """default_profile: fast
profiles:
  fast:
    protocol: openai
    model: gpt-fast
    base_url: https://api.example.test/v1
    api_key: test-api-key-must-not-leak
    context_window: 64000
  reasoned:
    protocol: anthropic
    model: claude-reasoned
    base_url: https://api.example.test/v1
    api_key: another-test-key
    thinking:
      enabled: true
""",
        )
    )

    assert profiles.active_name == "fast"
    assert profiles.active.model == "gpt-fast"
    assert profiles.get("fast").context_window == 64_000
    assert profiles.get("reasoned").thinking is not None


def test_example_configuration_stays_loadable_and_exposes_the_model_picker_profiles() -> None:
    from fakuicode.config import load_profiles

    example = Path(__file__).parent.parent / "fakuicode.example.yaml"
    profiles = load_profiles(example)

    assert profiles.active_name == "glm-coding"
    assert list(profiles.profiles) == ["openai-fast", "claude-thinking", "glm-coding"]
    assert profiles.get("glm-coding").protocol == "openai"


def test_old_single_profile_configuration_becomes_default_profile(tmp_path: Path) -> None:
    from fakuicode.config import load_profiles

    profiles = load_profiles(write_config(tmp_path, config_text()))

    assert profiles.active_name == "default"
    assert profiles.active.model == "test-model"


def test_accepts_remote_https_endpoint(tmp_path: Path) -> None:
    from fakuicode.config import load_config

    config = load_config(write_config(tmp_path, config_text(base_url="https://provider.example.test/v1")))

    assert config.base_url == "https://provider.example.test/v1"


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8080/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"],
)
def test_accepts_loopback_http_endpoints(tmp_path: Path, base_url: str) -> None:
    from fakuicode.config import load_config

    config = load_config(write_config(tmp_path, config_text(base_url=base_url)))

    assert config.base_url == base_url


@pytest.mark.parametrize(
    "base_url",
    ["http://provider.example.test/v1", "ftp://localhost/v1", "https:///v1", "https://user:pass@example.test/v1"],
)
def test_rejects_unsafe_or_invalid_endpoint_without_echoing_api_key(tmp_path: Path, base_url: str) -> None:
    from fakuicode.config import load_config
    from fakuicode.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as error:
        load_config(write_config(tmp_path, config_text(base_url=base_url)))

    assert API_KEY not in str(error.value)


def test_rejects_missing_field_without_echoing_api_key(tmp_path: Path) -> None:
    from fakuicode.config import load_config
    from fakuicode.errors import ConfigurationError

    path = write_config(tmp_path, f"protocol: openai\napi_key: {API_KEY}\n")

    with pytest.raises(ConfigurationError) as error:
        load_config(path)

    assert "model" in str(error.value)
    assert API_KEY not in str(error.value)


@pytest.mark.parametrize(
    "thinking, expected_message",
    [
        ("thinking:\n  enabled: true\n  budget_tokens: 2048\n", "budget_tokens"),
        ("thinking:\n  enabled: true\n  unexpected: value\n", "only"),
    ],
)
def test_rejects_removed_or_unknown_thinking_fields(
    tmp_path: Path, thinking: str, expected_message: str
) -> None:
    from fakuicode.config import load_config
    from fakuicode.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match=expected_message) as error:
        load_config(write_config(tmp_path, config_text(thinking=thinking)))

    assert API_KEY not in str(error.value)


def test_rejects_thinking_for_openai_with_migration_safe_error(tmp_path: Path) -> None:
    from fakuicode.config import load_config
    from fakuicode.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="thinking") as error:
        load_config(
            write_config(
                tmp_path,
                config_text(protocol="openai", thinking="thinking:\n  enabled: true\n"),
            )
        )

    assert API_KEY not in str(error.value)
