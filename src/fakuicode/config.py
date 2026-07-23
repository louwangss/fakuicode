"""YAML configuration loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import yaml

from fakuicode.errors import ConfigurationError
from fakuicode.models import ProfileSet, ProviderConfig, ThinkingConfig


REQUIRED_FIELDS = ("protocol", "model", "base_url", "api_key")


def load_config(path: Path) -> ProviderConfig:
    """Load the active provider configuration for backward compatibility."""
    return load_profiles(path).active


def load_profiles(path: Path) -> ProfileSet:
    """Load one legacy configuration or a named set of provider profiles."""
    loaded = _load_yaml(path)
    if "profiles" not in loaded:
        return ProfileSet({"default": _parse_provider_config(loaded)}, "default")

    raw_profiles = loaded.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ConfigurationError("Configuration field 'profiles' must be a non-empty mapping.")
    profiles: dict[str, ProviderConfig] = {}
    for name, value in raw_profiles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, Mapping):
            raise ConfigurationError("Each profile must have a non-empty name and a mapping value.")
        profiles[name.strip()] = _parse_provider_config(value)

    active_name = loaded.get("default_profile")
    if active_name is None:
        active_name = next(iter(profiles))
    if not isinstance(active_name, str) or active_name not in profiles:
        raise ConfigurationError("Configuration field 'default_profile' must name an existing profile.")
    return ProfileSet(profiles, active_name)


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from error

    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Configuration must be a YAML mapping.")
    return loaded


def _parse_provider_config(loaded: Mapping[str, object]) -> ProviderConfig:
    missing = [field for field in REQUIRED_FIELDS if not isinstance(loaded.get(field), str) or not loaded[field].strip()]
    if missing:
        raise ConfigurationError(f"Configuration is missing required field(s): {', '.join(missing)}.")

    protocol = loaded["protocol"].strip()
    if protocol not in {"anthropic", "openai"}:
        raise ConfigurationError("Configuration field 'protocol' must be 'anthropic' or 'openai'.")

    base_url = loaded["base_url"].strip().rstrip("/")
    _validate_base_url(base_url)

    thinking = _parse_thinking(loaded.get("thinking"), protocol)
    context_window = loaded.get("context_window", 128_000)
    if not isinstance(context_window, int) or isinstance(context_window, bool) or context_window < 1_024:
        raise ConfigurationError("Configuration field 'context_window' must be an integer of at least 1024.")
    return ProviderConfig(protocol, loaded["model"].strip(), base_url, loaded["api_key"].strip(), thinking, context_window)


def _parse_thinking(value: object, protocol: str) -> ThinkingConfig | None:
    if value is None:
        return None
    if protocol != "anthropic":
        raise ConfigurationError("Configuration field 'thinking' is supported only for anthropic.")
    if not isinstance(value, Mapping):
        raise ConfigurationError("Configuration field 'thinking' must be a mapping.")
    extra_fields = set(value) - {"enabled"}
    if "budget_tokens" in extra_fields:
        raise ConfigurationError("Remove 'thinking.budget_tokens': Claude uses adaptive thinking.")
    if extra_fields:
        raise ConfigurationError("Configuration field 'thinking' supports only 'enabled'.")
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigurationError("Thinking requires boolean 'enabled'.")
    return ThinkingConfig(enabled)


def _validate_base_url(base_url: str) -> None:
    """Allow HTTPS endpoints and HTTP only for explicit local development."""
    try:
        parsed_url = urlparse(base_url)
        hostname = parsed_url.hostname
        _ = parsed_url.port
    except ValueError as error:
        raise ConfigurationError("Configuration field 'base_url' must be a valid HTTP(S) URL.") from error

    if parsed_url.scheme not in {"http", "https"} or not hostname:
        raise ConfigurationError("Configuration field 'base_url' must be a valid HTTP(S) URL.")
    if parsed_url.username is not None or parsed_url.password is not None:
        raise ConfigurationError("Configuration field 'base_url' must not include URL user information.")
    if parsed_url.scheme == "http" and not _is_loopback_host(hostname):
        raise ConfigurationError("Configuration field 'base_url' may use HTTP only for a loopback host.")


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
