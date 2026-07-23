"""Fingerprint-based trust for project-defined MCP servers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit, urlunsplit

import yaml

from fakuicode.mcp.config import referenced_variables
from fakuicode.mcp.models import (
    DisabledServerConfig,
    HttpServerConfig,
    McpConfigSource,
    McpDiagnostic,
    McpFailureCode,
    McpServerConfig,
    McpServerIdentity,
    McpTransportType,
    McpTrustRequest,
    StdioServerConfig,
)


_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class McpTrustStorageError(RuntimeError):
    """Raised when trust cannot be read or changed safely."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise McpTrustStorageError("trust mapping key is invalid") from error
        if duplicate:
            raise McpTrustStorageError("trust mapping contains a duplicate key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def workspace_id(workspace: Path) -> str:
    normalized = os.path.normcase(os.path.normpath(str(workspace.resolve()))).replace("\\", "/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def server_identity(workspace: Path, config: McpServerConfig) -> McpServerIdentity:
    canonical = _canonical_server(config)
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return McpServerIdentity(
        workspace_id(workspace),
        config.name,
        hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )


def build_trust_request(workspace: Path, config: McpServerConfig) -> McpTrustRequest | None:
    if config.source is not McpConfigSource.PROJECT or isinstance(config, DisabledServerConfig):
        return None
    identity = server_identity(workspace, config)
    if isinstance(config, StdioServerConfig):
        return McpTrustRequest(
            identity=identity,
            transport=McpTransportType.STDIO,
            command=_ellipsize(config.command),
            argument_count=len(config.args),
            environment_names=tuple(sorted(config.env_templates)),
            referenced_variable_names=referenced_variables(config),
        )
    return McpTrustRequest(
        identity=identity,
        transport=McpTransportType.HTTP,
        redacted_url=_ellipsize(_redact_url(config.url)),
        header_names=tuple(sorted(config.header_templates)),
        referenced_variable_names=referenced_variables(config),
    )


class McpTrustRepository:
    """Versioned fail-closed repository for approved project server fingerprints."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.diagnostic: McpDiagnostic | None = None

    def is_trusted(self, identity: McpServerIdentity) -> bool:
        try:
            records = self._load()
        except McpTrustStorageError:
            self.diagnostic = _storage_diagnostic()
            return False
        self.diagnostic = None
        return records.get(identity.workspace_id, {}).get(identity.server_name) == identity.fingerprint

    def approve(self, identity: McpServerIdentity) -> None:
        try:
            records = self._load()
        except McpTrustStorageError:
            self.diagnostic = _storage_diagnostic()
            raise
        records.setdefault(identity.workspace_id, {})[identity.server_name] = identity.fingerprint
        document = {
            "version": _VERSION,
            "trusted_servers": {
                workspace: dict(sorted(servers.items()))
                for workspace, servers in sorted(records.items())
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
            raise McpTrustStorageError("trust file cannot be read") from error
        try:
            loaded = yaml.load(text, Loader=_UniqueKeyLoader)
        except (yaml.YAMLError, McpTrustStorageError) as error:
            raise McpTrustStorageError("trust file is invalid") from error
        if not isinstance(loaded, Mapping) or set(loaded) != {"version", "trusted_servers"}:
            raise McpTrustStorageError("trust file fields are invalid")
        if loaded.get("version") != _VERSION or isinstance(loaded.get("version"), bool):
            raise McpTrustStorageError("trust file version is invalid")
        raw = loaded.get("trusted_servers")
        if not isinstance(raw, Mapping):
            raise McpTrustStorageError("trusted_servers is invalid")
        records: dict[str, dict[str, str]] = {}
        for workspace, servers in raw.items():
            if (
                not isinstance(workspace, str)
                or _HASH.fullmatch(workspace) is None
                or not isinstance(servers, Mapping)
            ):
                raise McpTrustStorageError("trust record is invalid")
            parsed: dict[str, str] = {}
            for name, fingerprint in servers.items():
                if (
                    not isinstance(name, str)
                    or _SERVER_NAME.fullmatch(name) is None
                    or not isinstance(fingerprint, str)
                    or _HASH.fullmatch(fingerprint) is None
                ):
                    raise McpTrustStorageError("server trust record is invalid")
                parsed[name] = fingerprint
            records[workspace] = parsed
        return records


def _canonical_server(config: McpServerConfig) -> dict[str, object]:
    if isinstance(config, DisabledServerConfig):
        return {"name": config.name, "source": config.source.value, "enabled": False}
    common: dict[str, object] = {
        "name": config.name,
        "source": config.source.value,
        "type": config.transport.value,
        "enabled": True,
        "enabled_tools": sorted(config.enabled_tools) if config.enabled_tools is not None else None,
        "disabled_tools": sorted(config.disabled_tools),
    }
    if isinstance(config, StdioServerConfig):
        common.update(
            command=config.command,
            args=list(config.args),
            env=dict(sorted(config.env_templates.items())),
        )
    else:
        common.update(
            url=config.url,
            headers=dict(sorted(config.header_templates.items())),
        )
    return common


def _ellipsize(value: str, limit: int = 160) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    query = "<redacted>" if parsed.query else ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def _storage_diagnostic() -> McpDiagnostic:
    return McpDiagnostic(
        "MCP 信任存储不可用，项目 Server 已按未信任处理。",
        failure_code=McpFailureCode.TRUST_STORAGE,
    )


def _atomic_write(path: Path, document: Mapping[str, object]) -> None:
    content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
    except OSError as error:
        raise McpTrustStorageError("unable to prepare trust save") from error
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
        raise McpTrustStorageError("unable to save trust") from error
