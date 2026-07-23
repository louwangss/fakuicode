"""Core data models shared across Fakuicode modules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal, Mapping


ProtocolName = Literal["anthropic", "openai"]
AgentMode = Literal["execute", "plan"]
ToolArgumentError = Literal["invalid_json"]


@dataclass(frozen=True)
class ThinkingConfig:
    enabled: bool


@dataclass(frozen=True)
class ProviderConfig:
    protocol: ProtocolName
    model: str
    base_url: str
    api_key: str
    thinking: ThinkingConfig | None = None
    context_window: int = 128_000


@dataclass(frozen=True)
class ProfileSet:
    """Named provider configurations with one active profile."""

    profiles: Mapping[str, ProviderConfig]
    active_name: str

    def __post_init__(self) -> None:
        if not self.profiles or self.active_name not in self.profiles:
            raise ValueError("The active profile must exist.")

    @property
    def active(self) -> ProviderConfig:
        return self.profiles[self.active_name]

    def get(self, name: str) -> ProviderConfig:
        return self.profiles[name]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["thinking_start", "thinking_delta", "thinking_end", "text_delta", "completed"]
    text: str = ""


TimelineEventKind = Literal[
    "user",
    "assistant",
    "thinking",
    "tool_call",
    "tool_result",
    "progress",
    "usage",
    "summary",
    "context_diagnostic",
    "hook_diagnostic",
    "skill_activation",
    "agent_result",
    "system",
]


@dataclass(frozen=True)
class TimelineEvent:
    """A persistent, ordered item shown in a conversation timeline."""

    sequence: int
    kind: TimelineEventKind
    content: str
    call_id: str | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]
    argument_error: ToolArgumentError | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    output: str
    summary: str
    duration_seconds: float | None = None
    metadata: Mapping[str, object] | None = None

    def to_model_content(self) -> str:
        """Return the stable, provider-independent tool-result payload."""
        return json.dumps(
            {"success": self.success, "summary": self.summary, "output": self.output},
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class ProviderMessageState:
    """Opaque, bounded Provider blocks required to continue one assistant turn."""

    protocol: ProtocolName
    thinking_blocks: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class AgentMessage:
    """A provider-neutral turn, including native tool calls and their results."""

    role: Literal["user", "assistant"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    provider_state: ProviderMessageState | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Exact token counts reported by a provider for one streamed response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    context_input_tokens: int | None = None

    @property
    def available(self) -> bool:
        return self.input_tokens is not None or self.output_tokens is not None

    @property
    def cache_available(self) -> bool:
        return self.cache_read_tokens is not None or self.cache_write_tokens is not None


@dataclass(frozen=True)
class AgentProgress:
    """A state transition emitted by the bounded agent loop."""

    round_number: int
    phase: Literal["model", "tools"]


@dataclass(frozen=True)
class ContextStatus:
    """Provider-neutral, non-content status for context preparation."""

    trigger: Literal["automatic", "manual", "emergency"]
    result: Literal["compacted", "noop", "failed", "blocked", "breaker"]
    estimated_before: int | None = None
    estimated_after: int | None = None
    artifact_count: int = 0
    artifact_bytes: int = 0
    duration_seconds: float | None = None
    consecutive_failures: int = 0
    error_category: str | None = None
    recovery_hint: str = ""


@dataclass(frozen=True)
class AgentStreamEvent:
    """An incremental provider event for an agent turn."""

    kind: Literal[
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_delta",
        "tool_call",
        "tool_result",
        "usage",
        "context_status",
        "progress",
        "completed",
        "cancelled",
        "error",
    ]
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    usage: TokenUsage | None = None
    context_status: ContextStatus | None = None
    progress: AgentProgress | None = None
    provider_state: ProviderMessageState | None = None
