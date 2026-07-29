"""Feature gates for Agent Team behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fakuicode.errors import ConfigurationError


@dataclass(frozen=True)
class TeamFeatureConfig:
    coordinator_enabled: bool = False


def parse_team_config(root: Mapping[str, object]) -> TeamFeatureConfig:
    raw_teams = root.get("teams")
    if raw_teams is None:
        return TeamFeatureConfig()
    if not isinstance(raw_teams, Mapping):
        raise ConfigurationError("Configuration field 'teams' must be a mapping.")
    if set(raw_teams) - {"coordinator"}:
        raise ConfigurationError("Configuration field 'teams' contains unknown fields.")
    raw_coordinator = raw_teams.get("coordinator")
    if raw_coordinator is None:
        return TeamFeatureConfig()
    if not isinstance(raw_coordinator, Mapping):
        raise ConfigurationError("Configuration field 'teams.coordinator' must be a mapping.")
    if set(raw_coordinator) - {"enabled"}:
        raise ConfigurationError(
            "Configuration field 'teams.coordinator' supports only 'enabled'."
        )
    enabled = raw_coordinator.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigurationError(
            "Configuration field 'teams.coordinator.enabled' must be boolean."
        )
    return TeamFeatureConfig(coordinator_enabled=enabled)


def coordinator_is_enabled(
    config: TeamFeatureConfig,
    environment: Mapping[str, str],
) -> bool:
    """Require both the static capability gate and an exact startup opt-in."""

    return (
        config.coordinator_enabled
        and environment.get("FAKUICODE_COORDINATOR", "") == "1"
    )
