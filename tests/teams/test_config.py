from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.config import load_team_config
from fakuicode.errors import ConfigurationError
from fakuicode.teams.config import TeamFeatureConfig, coordinator_is_enabled


def test_team_config_defaults_to_safe_disabled(tmp_path: Path) -> None:
    path = tmp_path / "fakuicode.yaml"
    path.write_text(
        "protocol: openai\nmodel: test\nbase_url: https://example.test\napi_key: key\n",
        encoding="utf-8",
    )

    assert load_team_config(path) == TeamFeatureConfig()


def test_coordinator_requires_config_and_exact_environment_gate(tmp_path: Path) -> None:
    path = tmp_path / "fakuicode.yaml"
    path.write_text(
        """
profiles:
  default:
    protocol: openai
    model: test
    base_url: https://example.test
    api_key: key
teams:
  coordinator:
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    config = load_team_config(path)

    assert coordinator_is_enabled(config, {"FAKUICODE_COORDINATOR": "1"}) is True
    assert coordinator_is_enabled(config, {"FAKUICODE_COORDINATOR": "true"}) is False
    assert coordinator_is_enabled(TeamFeatureConfig(), {"FAKUICODE_COORDINATOR": "1"}) is False


def test_team_config_rejects_unknown_or_non_boolean_fields(tmp_path: Path) -> None:
    path = tmp_path / "fakuicode.yaml"
    path.write_text(
        """
profiles:
  default:
    protocol: openai
    model: test
    base_url: https://example.test
    api_key: key
teams:
  coordinator:
    enabled: yes
    shell: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_team_config(path)
