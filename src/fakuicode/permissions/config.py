"""Strict permission configuration loading and controlled persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from fakuicode.errors import PermissionConfigurationError, PermissionPersistenceError
from fakuicode.permissions.models import PermissionMode, Rule, RuleEffect, RuleSource
from fakuicode.permissions.rules import RuleSyntaxError, parse_rule


_VERSION = 1


@dataclass(frozen=True)
class PermissionPaths:
    user: Path
    project_shared: Path
    project_local: Path
    trust: Path

    @classmethod
    def for_workspace(cls, workspace: Path, *, home: Path | None = None) -> PermissionPaths:
        user_root = (home or Path.home()) / ".fakuicode"
        project_root = workspace / ".fakuicode"
        return cls(
            user_root / "permissions.yaml",
            project_root / "permissions.yaml",
            project_root / "permissions.local.yaml",
            user_root / "trusted-workspaces.yaml",
        )


@dataclass(frozen=True)
class PermissionConfigSnapshot:
    mode: PermissionMode = PermissionMode.DEFAULT
    user_rules: tuple[Rule, ...] = ()
    project_shared_rules: tuple[Rule, ...] = ()
    project_local_rules: tuple[Rule, ...] = ()
    project_trusted: bool = False
    locked: bool = False
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class PermissionConfigRepository:
    """Load startup snapshots and perform the two trusted internal writes."""

    def __init__(self, paths: PermissionPaths, workspace: Path) -> None:
        self.paths = paths
        self.workspace = workspace.resolve()
        self.workspace_identity = _workspace_identity(self.workspace)

    def load(self) -> PermissionConfigSnapshot:
        diagnostics: list[str] = []
        mode = PermissionMode.DEFAULT
        rules_by_source: dict[RuleSource, tuple[Rule, ...]] = {}
        sources = (
            ("user", self.paths.user, RuleSource.USER, True),
            ("project shared", self.paths.project_shared, RuleSource.PROJECT_SHARED, False),
            ("project local", self.paths.project_local, RuleSource.PROJECT_LOCAL, False),
        )
        for label, path, source, allow_mode in sources:
            try:
                if source is not RuleSource.USER:
                    _validate_project_config_path(path, self.workspace)
                loaded_mode, rules = _load_rule_file(path, source, allow_mode=allow_mode)
            except PermissionConfigurationError as error:
                diagnostics.append(f"{label} permission config is invalid: {error}")
                rules = ()
                loaded_mode = None
            rules_by_source[source] = rules
            if loaded_mode is not None:
                mode = loaded_mode

        trusted = False
        warnings: list[str] = []
        try:
            trusted = self.workspace_identity in _load_trusted_workspaces(self.paths.trust)
        except PermissionConfigurationError as error:
            warnings.append(f"trusted workspace config is invalid: {error}")

        locked = bool(diagnostics)
        return PermissionConfigSnapshot(
            mode=PermissionMode.STRICT if locked else mode,
            user_rules=rules_by_source[RuleSource.USER],
            project_shared_rules=rules_by_source[RuleSource.PROJECT_SHARED],
            project_local_rules=rules_by_source[RuleSource.PROJECT_LOCAL],
            project_trusted=trusted,
            locked=locked,
            diagnostics=tuple(diagnostics),
            warnings=tuple(warnings),
        )

    def set_project_trusted(self, trusted: bool) -> PermissionConfigSnapshot:
        try:
            identities = _load_trusted_workspaces(self.paths.trust)
        except PermissionConfigurationError:
            identities = set()
        if trusted:
            identities.add(self.workspace_identity)
        else:
            identities.discard(self.workspace_identity)
        document = {"version": _VERSION, "trusted_workspaces": sorted(identities)}
        _atomic_write_yaml(self.paths.trust, document)
        return self.load()

    def save_project_local_allow(
        self, snapshot: PermissionConfigSnapshot, expression: str
    ) -> PermissionConfigSnapshot:
        try:
            _validate_project_config_path(self.paths.project_local, self.workspace)
        except PermissionConfigurationError as error:
            raise PermissionPersistenceError(
                "The project-local permission path is not safe to write."
            ) from error
        try:
            new_rule = parse_rule(expression, RuleEffect.ALLOW, RuleSource.PROJECT_LOCAL)
        except (RuleSyntaxError, ValueError) as error:
            raise PermissionPersistenceError("The exact permission rule is invalid.") from error
        if not new_rule.exact:
            raise PermissionPersistenceError("Permanent automatic permissions must be exact rules.")

        existing = list(snapshot.project_local_rules)
        if not any(
            rule.expression == new_rule.expression and rule.effect is RuleEffect.ALLOW for rule in existing
        ):
            existing.append(new_rule)
        document = _rules_document(existing)
        _atomic_write_yaml(self.paths.project_local, document)
        return replace(snapshot, project_local_rules=tuple(_sorted_rules(existing)))


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise PermissionConfigurationError("duplicate YAML mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(path: Path) -> Mapping[str, object] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise PermissionConfigurationError("file cannot be read") from error
    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except PermissionConfigurationError:
        raise
    except (yaml.YAMLError, TypeError, ValueError) as error:
        raise PermissionConfigurationError("invalid YAML") from error
    if not isinstance(loaded, Mapping):
        raise PermissionConfigurationError("top level must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise PermissionConfigurationError("mapping keys must be strings")
    return loaded


def _load_rule_file(
    path: Path, source: RuleSource, *, allow_mode: bool
) -> tuple[PermissionMode | None, tuple[Rule, ...]]:
    loaded = _load_yaml_mapping(path)
    if loaded is None:
        return None, ()
    allowed_fields = {"version", "rules"} | ({"mode"} if allow_mode else set())
    extra = set(loaded) - allowed_fields
    if extra:
        raise PermissionConfigurationError("unknown top-level field")
    if loaded.get("version") != _VERSION or isinstance(loaded.get("version"), bool):
        raise PermissionConfigurationError("version must be 1")

    mode: PermissionMode | None = None
    if allow_mode and "mode" in loaded:
        try:
            mode = PermissionMode(loaded["mode"])
        except (TypeError, ValueError) as error:
            raise PermissionConfigurationError("mode must be strict, default, or trusted") from error

    raw_rules = loaded.get("rules", {})
    if not isinstance(raw_rules, Mapping):
        raise PermissionConfigurationError("rules must be a mapping")
    if not all(isinstance(key, str) for key in raw_rules):
        raise PermissionConfigurationError("rule group names must be strings")
    if set(raw_rules) - {"allow", "deny"}:
        raise PermissionConfigurationError("rules contain an unknown result group")
    rules: list[Rule] = []
    for effect in (RuleEffect.ALLOW, RuleEffect.DENY):
        expressions = raw_rules.get(effect.value, [])
        if not isinstance(expressions, list) or not all(isinstance(item, str) for item in expressions):
            raise PermissionConfigurationError(f"rules.{effect.value} must be a string list")
        for expression in expressions:
            try:
                rules.append(parse_rule(expression, effect, source))
            except RuleSyntaxError as error:
                raise PermissionConfigurationError("contains an invalid permission rule") from error
    return mode, tuple(rules)


def _load_trusted_workspaces(path: Path) -> set[str]:
    loaded = _load_yaml_mapping(path)
    if loaded is None:
        return set()
    if set(loaded) != {"version", "trusted_workspaces"}:
        raise PermissionConfigurationError("trust file fields are invalid")
    if loaded.get("version") != _VERSION or isinstance(loaded.get("version"), bool):
        raise PermissionConfigurationError("trust file version must be 1")
    values = loaded.get("trusted_workspaces")
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise PermissionConfigurationError("trusted_workspaces must be a string list")
    return {_normalize_stored_identity(value) for value in values}


def _workspace_identity(workspace: Path) -> str:
    return _normalize_stored_identity(str(workspace.resolve()))


def _normalize_stored_identity(value: str) -> str:
    return os.path.normcase(os.path.normpath(value)).replace("\\", "/")


def _validate_project_config_path(path: Path, workspace: Path) -> None:
    """Reject project configuration paths redirected through links or outside the workspace."""

    lexical = Path(os.path.abspath(path))
    try:
        lexical.relative_to(workspace)
    except ValueError as error:
        raise PermissionConfigurationError("path is outside the workspace") from error

    missing: list[str] = []
    ancestor = lexical
    while not ancestor.exists() and ancestor != ancestor.parent:
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    try:
        resolved = ancestor.resolve(strict=True).joinpath(*reversed(missing)).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PermissionConfigurationError("path cannot be resolved safely") from error
    if resolved != lexical:
        raise PermissionConfigurationError("path must not use symbolic-link redirection")


def _rules_document(rules: list[Rule]) -> dict[str, object]:
    allow = sorted({rule.expression for rule in rules if rule.effect is RuleEffect.ALLOW})
    deny = sorted({rule.expression for rule in rules if rule.effect is RuleEffect.DENY})
    groups: dict[str, object] = {}
    if allow:
        groups["allow"] = allow
    if deny:
        groups["deny"] = deny
    return {"version": _VERSION, "rules": groups}


def _sorted_rules(rules: list[Rule]) -> list[Rule]:
    return sorted(rules, key=lambda rule: (rule.effect.value, rule.expression))


def _atomic_write_yaml(path: Path, document: Mapping[str, object]) -> None:
    content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
    except OSError as error:
        raise PermissionPersistenceError("Unable to prepare the permission configuration save.") from error
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PermissionPersistenceError("Unable to save the permission configuration.") from error
