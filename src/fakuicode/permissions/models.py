"""Immutable value objects shared by the permission subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class PermissionMode(_StringEnum):
    STRICT = "strict"
    DEFAULT = "default"
    TRUSTED = "trusted"


class DecisionKind(_StringEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class RuleEffect(_StringEnum):
    ALLOW = "allow"
    DENY = "deny"


class RuleSource(_StringEnum):
    USER = "user"
    PROJECT_SHARED = "project_shared"
    PROJECT_LOCAL = "project_local"
    SESSION = "session"


class ApprovalChoice(_StringEnum):
    DENY = "deny"
    ONCE = "once"
    SESSION = "session"
    PERMANENT = "permanent"


class PermissionScope(_StringEnum):
    TARGET = "target"
    TOOL = "tool"


@dataclass(frozen=True)
class Rule:
    expression: str
    tool_name: str
    pattern: str
    effect: RuleEffect
    source: RuleSource
    exact: bool
    _matcher: re.Pattern[str] = field(repr=False, compare=False)

    def matches(self, target: str) -> bool:
        return self._matcher.fullmatch(target) is not None


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str
    layer: str
    rule: Rule | None = None


@dataclass(frozen=True)
class PermissionRequest:
    request_id: str
    call_id: str
    tool_name: str
    target: str
    reason: str
    exact_rule: str
    scope: PermissionScope = PermissionScope.TARGET
    source: str | None = None


@dataclass(frozen=True)
class PermissionSubject:
    tool_name: str
    target: str
    read_only: bool
