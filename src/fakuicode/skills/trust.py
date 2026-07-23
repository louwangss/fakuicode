"""Versioned, fingerprint-based trust for project Skill packages containing code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile

import yaml

from fakuicode.mcp.trust import workspace_id
from fakuicode.skills.models import SkillDefinition, SkillSource


_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillTrustStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillTrustIdentity:
    workspace_id: str
    skill_name: str
    fingerprint: str


@dataclass(frozen=True)
class SkillTrustRequest:
    name: str
    source: SkillSource
    package_path: Path
    fingerprint: str
    capabilities: tuple[str, ...]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SkillTrustStorageError("trust mapping contains a duplicate key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def skill_identity(workspace: Path, skill: SkillDefinition) -> SkillTrustIdentity:
    return SkillTrustIdentity(workspace_id(workspace), skill.name, skill.fingerprint)


def build_skill_trust_request(skill: SkillDefinition) -> SkillTrustRequest:
    return SkillTrustRequest(
        skill.name,
        skill.source,
        skill.package_path,
        skill.fingerprint,
        tuple(f"{tool.name}: {tool.description}" for tool in skill.tools),
    )


class SkillTrustRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.diagnostic: str | None = None

    def is_trusted(self, identity: SkillTrustIdentity) -> bool:
        try:
            records = self._load()
        except SkillTrustStorageError:
            self.diagnostic = "Skill 信任存储不可用，项目 Skill 已按未信任处理。"
            return False
        self.diagnostic = None
        return records.get(identity.workspace_id, {}).get(identity.skill_name) == identity.fingerprint

    def approve(self, identity: SkillTrustIdentity) -> None:
        try:
            records = self._load()
        except SkillTrustStorageError:
            self.diagnostic = "Skill 信任存储不可用，项目 Skill 已按未信任处理。"
            raise
        records.setdefault(identity.workspace_id, {})[identity.skill_name] = identity.fingerprint
        document = {
            "version": _VERSION,
            "trusted_skills": {
                workspace: dict(sorted(skills.items())) for workspace, skills in sorted(records.items())
            },
        }
        _atomic_write(self.path, document)
        self.diagnostic = None

    def _load(self) -> dict[str, dict[str, str]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise SkillTrustStorageError("trust file cannot be read") from error
        try:
            raw = yaml.load(text, Loader=_UniqueKeyLoader)
        except (yaml.YAMLError, SkillTrustStorageError) as error:
            raise SkillTrustStorageError("trust file is invalid") from error
        if not isinstance(raw, Mapping) or set(raw) != {"version", "trusted_skills"}:
            raise SkillTrustStorageError("trust file fields are invalid")
        if raw.get("version") != _VERSION or isinstance(raw.get("version"), bool):
            raise SkillTrustStorageError("trust file version is invalid")
        trusted = raw.get("trusted_skills")
        if not isinstance(trusted, Mapping):
            raise SkillTrustStorageError("trusted_skills is invalid")
        result: dict[str, dict[str, str]] = {}
        for workspace, skills in trusted.items():
            if not isinstance(workspace, str) or _HASH.fullmatch(workspace) is None or not isinstance(skills, Mapping):
                raise SkillTrustStorageError("trust record is invalid")
            parsed = {}
            for name, fingerprint in skills.items():
                if (
                    not isinstance(name, str)
                    or _NAME.fullmatch(name) is None
                    or not isinstance(fingerprint, str)
                    or _HASH.fullmatch(fingerprint) is None
                ):
                    raise SkillTrustStorageError("skill trust record is invalid")
                parsed[name] = fingerprint
            result[workspace] = parsed
        return result


def _atomic_write(path: Path, document: Mapping[str, object]) -> None:
    content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    except OSError as error:
        raise SkillTrustStorageError("unable to prepare trust save") from error
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SkillTrustStorageError("unable to save trust") from error
