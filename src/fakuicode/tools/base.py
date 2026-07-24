"""Shared contract for local tools exposed to a provider."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from types import MappingProxyType
from typing import Mapping, Protocol

from fakuicode.models import ToolDefinition
from fakuicode.permissions.models import PermissionScope


FINISH_AGENT_TURN = "finish_agent_turn"
FINISH_AGENT_TURN_MESSAGE = "finish_agent_turn_message"


@dataclass(frozen=True)
class ToolExecution:
    """A bounded local outcome before it is linked to a provider call."""

    success: bool
    output: str
    summary: str
    duration_seconds: float | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ToolPreparation:
    """Validated immutable arguments plus the only target used for authorization."""

    arguments: Mapping[str, object]
    target: str
    permission_scope: PermissionScope = PermissionScope.TARGET


@dataclass(frozen=True)
class PreparedToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]
    target: str
    read_only: bool
    permission_scope: PermissionScope = PermissionScope.TARGET


class Tool(Protocol):
    """A single provider-visible local capability."""

    @property
    def definition(self) -> ToolDefinition: ...

    @property
    def read_only(self) -> bool: ...

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation: ...

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution: ...

    def execute(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution: ...


def freeze_arguments(arguments: Mapping[str, object]) -> Mapping[str, object]:
    """Freeze a mapping after each tool has converted nested containers."""

    return MappingProxyType(dict(arguments))
