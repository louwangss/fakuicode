from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from threading import Event
from typing import Protocol

from fakuicode.models import AgentMessage, AgentStreamEvent, Message, StreamEvent, ToolDefinition
from fakuicode.prompting import build_stable_prompt


AGENT_SYSTEM_PROMPT = build_stable_prompt()


@dataclass(frozen=True)
class ProviderCapabilities:
    """Optional protocol features that callers may request without assuming support."""

    supports_output_token_limit: bool = False


@dataclass(frozen=True)
class AgentRequest:
    """Provider request with stable and dynamic system content kept distinct."""

    messages: tuple[AgentMessage, ...]
    tools: tuple[ToolDefinition, ...]
    system_prompt: str = AGENT_SYSTEM_PROMPT
    system_supplement: str = ""
    cancel_event: Event | None = None
    output_token_limit: int | None = None

    def __post_init__(self) -> None:
        if self.output_token_limit is not None and self.output_token_limit <= 0:
            raise ValueError("output_token_limit must be positive when provided.")


class ChatProvider(Protocol):
    def stream_chat(self, messages: Sequence[Message]) -> Iterator[StreamEvent]: ...


class AgentProvider(Protocol):
    """Canonical tool-calling provider contract used by the agent loop."""

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def stream_agent_request(self, request: AgentRequest) -> Iterator[AgentStreamEvent]: ...

    def cancel(self) -> None: ...


class Provider(ChatProvider, AgentProvider, Protocol):
    """Complete production Provider contract returned by the Provider factory."""
