"""Immutable value objects shared by the MCP client subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class McpTransportType(_StringEnum):
    STDIO = "stdio"
    HTTP = "http"


class McpConfigSource(_StringEnum):
    USER = "user"
    PROJECT = "project"


class McpServerStatus(_StringEnum):
    DISABLED = "disabled"
    PENDING_TRUST = "pending_trust"
    TRUST_DENIED = "trust_denied"
    CONFIG_ERROR = "config_error"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    RESTART_REQUIRED = "restart_required"
    CLOSED = "closed"


class McpFailureCode(_StringEnum):
    INVALID_CONFIG = "invalid_config"
    MISSING_ENVIRONMENT = "missing_environment"
    INSECURE_URL = "insecure_url"
    TRUST_STORAGE = "trust_storage"
    STARTUP_TIMEOUT = "startup_timeout"
    CONNECT_FAILED = "connect_failed"
    INCOMPATIBLE_PROTOCOL = "incompatible_protocol"
    TOOLS_UNAVAILABLE = "tools_unavailable"
    PROTOCOL_ERROR = "protocol_error"
    CALL_TIMEOUT = "call_timeout"
    CALL_CANCELLED = "call_cancelled"
    CONNECTION_CLOSED = "connection_closed"


@dataclass(frozen=True)
class SecretText:
    value: str = field(repr=False)


@dataclass(frozen=True)
class StdioServerConfig:
    name: str
    source: McpConfigSource
    command: str
    args: tuple[str, ...] = ()
    env_templates: Mapping[str, str] = field(default_factory=dict, repr=False)
    enabled_tools: frozenset[str] | None = None
    disabled_tools: frozenset[str] = frozenset()

    @property
    def transport(self) -> McpTransportType:
        return McpTransportType.STDIO


@dataclass(frozen=True)
class HttpServerConfig:
    name: str
    source: McpConfigSource
    url: str = field(repr=False)
    header_templates: Mapping[str, str] = field(default_factory=dict, repr=False)
    enabled_tools: frozenset[str] | None = None
    disabled_tools: frozenset[str] = frozenset()

    @property
    def transport(self) -> McpTransportType:
        return McpTransportType.HTTP


@dataclass(frozen=True)
class DisabledServerConfig:
    name: str
    source: McpConfigSource

    @property
    def transport(self) -> None:
        return None


McpServerConfig = StdioServerConfig | HttpServerConfig | DisabledServerConfig


@dataclass(frozen=True)
class ResolvedStdioServerConfig:
    name: str
    command: str
    args: tuple[str, ...]
    environment: Mapping[str, SecretText] = field(repr=False)
    enabled_tools: frozenset[str] | None = None
    disabled_tools: frozenset[str] = frozenset()
    working_directory: Path | None = None

    @property
    def transport(self) -> McpTransportType:
        return McpTransportType.STDIO


@dataclass(frozen=True)
class ResolvedHttpServerConfig:
    name: str
    url: str = field(repr=False)
    headers: Mapping[str, SecretText] = field(repr=False)
    enabled_tools: frozenset[str] | None = None
    disabled_tools: frozenset[str] = frozenset()

    @property
    def transport(self) -> McpTransportType:
        return McpTransportType.HTTP


ResolvedServerConfig = ResolvedStdioServerConfig | ResolvedHttpServerConfig


@dataclass(frozen=True)
class McpDiagnostic:
    message: str
    server_name: str | None = None
    source: McpConfigSource | None = None
    failure_code: McpFailureCode = McpFailureCode.INVALID_CONFIG


@dataclass(frozen=True)
class McpConfigSnapshot:
    servers: tuple[McpServerConfig, ...] = ()
    diagnostics: tuple[McpDiagnostic, ...] = ()
    has_configuration: bool = False


@dataclass(frozen=True)
class McpServerIdentity:
    workspace_id: str
    server_name: str
    fingerprint: str


@dataclass(frozen=True)
class McpTrustRequest:
    identity: McpServerIdentity
    transport: McpTransportType
    command: str | None = None
    arguments: tuple[str, ...] = field(default=(), repr=False)
    working_directory: Path | None = None
    redacted_url: str | None = None
    environment_names: tuple[str, ...] = ()
    header_names: tuple[str, ...] = ()
    referenced_variable_names: tuple[str, ...] = ()

    @property
    def argument_count(self) -> int:
        return len(self.arguments)


@dataclass(frozen=True)
class McpInitializeInfo:
    protocol_version: str
    tools_supported: bool


@dataclass(frozen=True)
class McpRemoteTool:
    name: object
    description: object = None
    input_schema: object = None


@dataclass(frozen=True)
class McpToolPage:
    tools: tuple[McpRemoteTool, ...]
    next_cursor: str | None = None


@dataclass(frozen=True)
class McpContentBlock:
    kind: str
    text: str | None = None


@dataclass(frozen=True)
class McpCallResult:
    content: tuple[McpContentBlock, ...] = ()
    structured_content: Any = None
    is_error: bool = False
    failure_code: McpFailureCode | None = None
    public_summary: str | None = None


@dataclass(frozen=True)
class McpToolBinding:
    public_name: str
    server_name: str
    remote_name: str
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True)
class McpServerState:
    name: str
    transport: McpTransportType | None
    status: McpServerStatus
    tool_count: int = 0
    failure_code: McpFailureCode | None = None
    public_summary: str | None = None


@dataclass(frozen=True)
class McpStartupSnapshot:
    states: tuple[McpServerState, ...] = ()
    tools: tuple[McpToolBinding, ...] = ()
    diagnostics: tuple[McpDiagnostic, ...] = ()
