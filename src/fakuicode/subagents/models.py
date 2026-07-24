"""Immutable contracts for subagent definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AgentSource(str, Enum):
    PLUGIN = "plugin"
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


class PermissionBehavior(str, Enum):
    INHERIT = "inherit"
    DEFAULT = "default"
    STRICT = "strict"
    TRUSTED = "trusted"
    DONT_ASK = "dontAsk"
    PLAN = "plan"


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    prompt: str
    source: AgentSource
    path: Path
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    profile: str = "inherit"
    max_turns: int | None = None
    permission_mode: PermissionBehavior = PermissionBehavior.INHERIT
    background: bool = False


@dataclass(frozen=True)
class CatalogDiagnostic:
    source: AgentSource
    path: Path
    message: str
    name: str | None = None

