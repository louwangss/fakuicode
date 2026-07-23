"""Independent content-fingerprint trust for project Hook configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile

import yaml


_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")


class HookTrustStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class HookTrustIdentity:
    workspace_id: str
    fingerprint: str


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise HookTrustStorageError("trust mapping key is invalid") from error
        if duplicate:
            raise HookTrustStorageError("trust mapping contains a duplicate key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


class HookTrustRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.diagnostic: str | None = None

    def is_trusted(self, identity: HookTrustIdentity) -> bool:
        try:
            records = self._load()
        except HookTrustStorageError:
            self.diagnostic = "Hook 信任存储不可用，项目 Hook 已按未信任处理。"
            return False
        self.diagnostic = None
        return records.get(identity.workspace_id) == identity.fingerprint

    def approve(self, identity: HookTrustIdentity) -> None:
        records = self._load()
        records[identity.workspace_id] = identity.fingerprint
        _atomic_write(
            self.path,
            {"version": _VERSION, "trusted_projects": dict(sorted(records.items()))},
        )
        self.diagnostic = None

    def _load(self) -> dict[str, str]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise HookTrustStorageError("trust file cannot be read") from error
        try:
            raw = yaml.load(text, Loader=_UniqueKeyLoader)
        except (yaml.YAMLError, HookTrustStorageError) as error:
            raise HookTrustStorageError("trust file is invalid") from error
        if not isinstance(raw, Mapping) or set(raw) != {"version", "trusted_projects"}:
            raise HookTrustStorageError("trust file fields are invalid")
        if raw.get("version") != _VERSION or isinstance(raw.get("version"), bool):
            raise HookTrustStorageError("trust file version is invalid")
        values = raw.get("trusted_projects")
        if not isinstance(values, Mapping):
            raise HookTrustStorageError("trusted_projects is invalid")
        result: dict[str, str] = {}
        for workspace, fingerprint in values.items():
            if (
                not isinstance(workspace, str)
                or _HASH.fullmatch(workspace) is None
                or not isinstance(fingerprint, str)
                or _HASH.fullmatch(fingerprint) is None
            ):
                raise HookTrustStorageError("trust record is invalid")
            result[workspace] = fingerprint
        return result


def _atomic_write(path: Path, document: Mapping[str, object]) -> None:
    content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    except OSError as error:
        raise HookTrustStorageError("unable to prepare trust save") from error
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
        raise HookTrustStorageError("unable to save trust") from error
