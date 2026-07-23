"""Immutable contracts for lifecycle Hook configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

from fakuicode.hooks.pointers import resolve_pointer


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class HookEvent(_StringEnum):
    APP_START = "app_start"
    APP_STOP = "app_stop"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    PRE_MODEL_REQUEST = "pre_model_request"
    POST_MODEL_RESPONSE = "post_model_response"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    CONTEXT_CLEARED = "context_cleared"


class HookSource(_StringEnum):
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True)
class HookPredicate:
    field: str
    path: tuple[str, ...]
    kind: str
    expected: object
    negated: bool = False
    _regex: re.Pattern[str] | None = field(default=None, repr=False, compare=False)

    def matches(self, payload: object) -> bool:
        found, actual = resolve_pointer(payload, self.path)
        if not found:
            return False
        if self.kind == "exact":
            matched = type(actual) is type(self.expected) and actual == self.expected
        else:
            matched = isinstance(actual, str) and self._regex is not None and self._regex.fullmatch(actual) is not None
        return not matched if self.negated else matched


@dataclass(frozen=True)
class HookCondition:
    mode: str
    predicates: tuple[HookPredicate, ...]

    def matches(self, payload: object) -> bool:
        results = (predicate.matches(payload) for predicate in self.predicates)
        return all(results) if self.mode == "all" else any(results)


@dataclass(frozen=True)
class PromptAction:
    content: str
    once: bool = False


@dataclass(frozen=True)
class CommandAction:
    command: str
    command_windows: str | None = None
    timeout_seconds: float = 60.0
    async_: bool = False
    once: bool = False


@dataclass(frozen=True)
class HttpAction:
    url: str
    headers: tuple[tuple[str, str], ...] = ()
    allowed_env_vars: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    async_: bool = False
    once: bool = False


@dataclass(frozen=True)
class AgentAction:
    prompt: str
    once: bool = False


HookAction = PromptAction | CommandAction | HttpAction | AgentAction


@dataclass(frozen=True)
class HookRule:
    name: str
    event: HookEvent
    action: HookAction
    source: HookSource
    condition: HookCondition | None = None


@dataclass(frozen=True)
class HookConfigSnapshot:
    rules: tuple[HookRule, ...] = ()
    project_rules: tuple[HookRule, ...] = ()
    diagnostics: tuple[str, ...] = ()
    project_fingerprint: str | None = None
    project_trusted: bool = False
