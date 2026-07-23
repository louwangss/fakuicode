"""Strict YAML loading and centralized validation for lifecycle Hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

import yaml

from fakuicode.errors import HookConfigurationError
from fakuicode.hooks.models import (
    AgentAction,
    CommandAction,
    HookCondition,
    HookConfigSnapshot,
    HookEvent,
    HookPredicate,
    HookRule,
    HookSource,
    HttpAction,
    PromptAction,
)
from fakuicode.hooks.pointers import JsonPointerError, parse_pointer
from fakuicode.hooks.trust import HookTrustIdentity, HookTrustRepository
from fakuicode.mcp.trust import workspace_id
from fakuicode.matching import GlobSyntaxError, compile_glob


_VERSION = 1
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ENV_REFERENCE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_RULE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


@dataclass(frozen=True)
class HookPaths:
    user: Path
    project: Path
    trust: Path

    @classmethod
    def for_workspace(cls, workspace: Path, *, home: Path | None = None) -> HookPaths:
        user_root = (home or Path.home()) / ".fakuicode"
        return cls(
            user_root / "hooks.yaml",
            workspace / ".fakuicode" / "hooks.yaml",
            user_root / "trusted-hooks.yaml",
        )


class HookConfigRepository:
    def __init__(
        self,
        paths: HookPaths,
        workspace: Path,
        *,
        project_trusted: bool = False,
        trust_repository: HookTrustRepository | None = None,
    ) -> None:
        self.paths = paths
        self.workspace = workspace.resolve()
        self.project_trusted = project_trusted
        self.trust_repository = trust_repository

    def load(self) -> HookConfigSnapshot:
        diagnostics: list[str] = []
        user_rules = self._load_source("user", self.paths.user, HookSource.USER, diagnostics)
        try:
            _validate_project_path(self.paths.project, self.workspace)
            project_fingerprint = _fingerprint(self.paths.project)
            project_trusted = self.project_trusted
            if project_fingerprint is not None and self.trust_repository is not None:
                project_trusted = self.trust_repository.is_trusted(
                    HookTrustIdentity(workspace_id(self.workspace), project_fingerprint)
                )
            project_rules = self._load_source(
                "project", self.paths.project, HookSource.PROJECT, diagnostics
            )
        except HookConfigurationError as error:
            diagnostics.append(f"project Hook config is invalid: {error}")
            project_rules = ()
            project_fingerprint = None
            project_trusted = False
        active = user_rules + (project_rules if project_trusted else ())
        return HookConfigSnapshot(
            rules=active,
            project_rules=project_rules,
            diagnostics=tuple(diagnostics),
            project_fingerprint=project_fingerprint,
            project_trusted=project_trusted,
        )

    @staticmethod
    def _load_source(
        label: str,
        path: Path,
        source: HookSource,
        diagnostics: list[str],
    ) -> tuple[HookRule, ...]:
        try:
            return _load_hook_file(path, source)
        except HookConfigurationError as error:
            diagnostics.append(f"{label} Hook config is invalid: {error}")
            return ()


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise HookConfigurationError("mapping keys must be scalar") from error
        if duplicate:
            raise HookConfigurationError("duplicate YAML mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _load_hook_file(path: Path, source: HookSource) -> tuple[HookRule, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except OSError as error:
        raise HookConfigurationError("file cannot be read") from error
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except HookConfigurationError:
        raise
    except (yaml.YAMLError, TypeError, ValueError) as error:
        raise HookConfigurationError("invalid YAML") from error
    mapping = _mapping(raw, "top level")
    _fields(mapping, {"version", "hooks"}, required={"version", "hooks"})
    if mapping["version"] != _VERSION or isinstance(mapping["version"], bool):
        raise HookConfigurationError("version must be 1")
    hooks = mapping["hooks"]
    if not isinstance(hooks, list):
        raise HookConfigurationError("hooks must be a list")
    rules = tuple(_parse_rule(value, source, index) for index, value in enumerate(hooks, 1))
    if len({rule.name for rule in rules}) != len(rules):
        raise HookConfigurationError("hook names must be unique within one source")
    return rules


def _parse_rule(raw: object, source: HookSource, index: int) -> HookRule:
    mapping = _mapping(raw, "hook")
    _fields(mapping, {"name", "event", "if", "action"}, required={"event", "action"})
    name = mapping.get("name", f"{source.value}-{index}")
    if not isinstance(name, str) or _RULE_NAME.fullmatch(name) is None:
        raise HookConfigurationError("hook name is invalid")
    try:
        event = HookEvent(mapping["event"])
    except (TypeError, ValueError) as error:
        raise HookConfigurationError("event is unknown") from error
    condition = _parse_condition(mapping["if"]) if "if" in mapping else None
    action = _parse_action(mapping["action"])
    if event is HookEvent.PRE_TOOL_USE and getattr(action, "async_", False):
        raise HookConfigurationError("pre_tool_use actions cannot be asynchronous")
    return HookRule(name, event, action, source, condition)


def _parse_condition(raw: object) -> HookCondition:
    mapping = _mapping(raw, "if")
    if set(mapping) not in ({"all"}, {"any"}):
        raise HookConfigurationError("if must contain exactly one of all or any")
    mode = next(iter(mapping))
    values = mapping[mode]
    if not isinstance(values, list) or not values:
        raise HookConfigurationError(f"if.{mode} must be a non-empty list")
    return HookCondition(mode, tuple(_parse_predicate(value) for value in values))


def _parse_predicate(raw: object) -> HookPredicate:
    mapping = _mapping(raw, "condition predicate")
    _fields(mapping, {"field", "exact", "glob", "regex", "not"}, required={"field"})
    matcher_fields = set(mapping) & {"exact", "glob", "regex"}
    if len(matcher_fields) != 1:
        raise HookConfigurationError("condition predicate needs exactly one matcher")
    field = mapping["field"]
    if not isinstance(field, str):
        raise HookConfigurationError("condition field must be a JSON Pointer")
    path = _parse_pointer(field)
    negated = mapping.get("not", False)
    if not isinstance(negated, bool):
        raise HookConfigurationError("condition not must be boolean")
    kind = next(iter(matcher_fields))
    expected = mapping[kind]
    compiled = None
    if kind == "exact":
        if isinstance(expected, (dict, list)) or expected is None:
            raise HookConfigurationError("exact matcher must be a scalar")
    else:
        if not isinstance(expected, str) or not expected:
            raise HookConfigurationError(f"{kind} matcher must be a non-empty string")
        try:
            compiled = compile_glob(expected)[0] if kind == "glob" else re.compile(expected)
        except (re.error, GlobSyntaxError) as error:
            raise HookConfigurationError(f"{kind} matcher is invalid") from error
    return HookPredicate(field, path, kind, expected, negated, compiled)


def _parse_action(raw: object):
    mapping = _mapping(raw, "action")
    action_type = mapping.get("type")
    if action_type == "prompt":
        _fields(mapping, {"type", "content", "once"}, required={"type", "content"})
        return PromptAction(_text(mapping["content"], "prompt content"), _boolean(mapping, "once"))
    if action_type == "command":
        _fields(
            mapping,
            {"type", "command", "command_windows", "timeout_seconds", "async", "once"},
            required={"type", "command"},
        )
        windows = mapping.get("command_windows")
        if windows is not None:
            windows = _text(windows, "command_windows")
        timeout = mapping.get("timeout_seconds", 60)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise HookConfigurationError("timeout_seconds must be positive")
        return CommandAction(
            _text(mapping["command"], "command"), windows, float(timeout),
            _boolean(mapping, "async"), _boolean(mapping, "once")
        )
    if action_type == "http":
        _fields(
            mapping,
            {"type", "url", "headers", "allowed_env_vars", "include", "async", "once"},
            required={"type", "url"},
        )
        url = _text(mapping["url"], "url")
        _validate_url(url)
        headers = _string_mapping(mapping.get("headers", {}), "headers")
        if any(
            _HEADER_NAME.fullmatch(key) is None
            or "\r" in value
            or "\n" in value
            or "${" in _ENV_REFERENCE.sub("", value)
            for key, value in headers
        ):
            raise HookConfigurationError("headers contain an invalid name, value, or environment reference")
        allowed = _string_list(mapping.get("allowed_env_vars", []), "allowed_env_vars")
        if any(_ENV_NAME.fullmatch(name) is None for name in allowed):
            raise HookConfigurationError("allowed_env_vars contains an invalid name")
        referenced = {name for _, value in headers for name in _ENV_REFERENCE.findall(value)}
        if not referenced.issubset(set(allowed)):
            raise HookConfigurationError("header environment reference is not allowlisted")
        include = _string_list(mapping.get("include", []), "include")
        for pointer in include:
            _parse_pointer(pointer)
        return HttpAction(
            url, headers, allowed, include, _boolean(mapping, "async"), _boolean(mapping, "once")
        )
    if action_type == "agent":
        _fields(mapping, {"type", "prompt", "once"}, required={"type", "prompt"})
        return AgentAction(_text(mapping["prompt"], "agent prompt"), _boolean(mapping, "once"))
    raise HookConfigurationError("action type is unknown")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise HookConfigurationError(f"{label} must be a string-keyed mapping")
    return value


def _fields(mapping: Mapping[str, object], allowed: set[str], *, required: set[str]) -> None:
    if set(mapping) - allowed:
        raise HookConfigurationError("mapping contains an unknown field")
    if required - set(mapping):
        raise HookConfigurationError("mapping is missing a required field")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HookConfigurationError(f"{label} must be a non-empty string")
    return value


def _boolean(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key, False)
    if not isinstance(value, bool):
        raise HookConfigurationError(f"{key} must be boolean")
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise HookConfigurationError(f"{label} must be a string list")
    return tuple(value)


def _string_mapping(value: object, label: str) -> tuple[tuple[str, str], ...]:
    mapping = _mapping(value, label)
    if not all(key and isinstance(item, str) for key, item in mapping.items()):
        raise HookConfigurationError(f"{label} values must be strings")
    return tuple(mapping.items())


def _parse_pointer(value: str) -> tuple[str, ...]:
    try:
        return parse_pointer(value)
    except JsonPointerError as error:
        raise HookConfigurationError(str(error)) from error


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise HookConfigurationError("HTTP URL is unsafe")
    loopback = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise HookConfigurationError("HTTP URL must use HTTPS or loopback HTTP")


def _fingerprint(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _validate_project_path(path: Path, workspace: Path) -> None:
    lexical = Path(os.path.abspath(path))
    try:
        lexical.relative_to(workspace)
    except ValueError as error:
        raise HookConfigurationError("path is outside the workspace") from error
    missing: list[str] = []
    ancestor = lexical
    while not ancestor.exists() and ancestor != ancestor.parent:
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    try:
        resolved = ancestor.resolve(strict=True).joinpath(*reversed(missing)).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise HookConfigurationError("path cannot be resolved safely") from error
    if resolved != lexical:
        raise HookConfigurationError("path must not use symbolic-link redirection")
