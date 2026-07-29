"""Canonical Provider invocation with an isolated legacy compatibility adapter."""

from __future__ import annotations

import inspect
from typing import Iterator

from fakuicode.errors import ProviderCapabilityError
from fakuicode.models import AgentStreamEvent, ToolDefinition
from fakuicode.providers.base import AGENT_SYSTEM_PROMPT, AgentRequest


class LegacyAgentProviderAdapter:
    """Adapt pre-contract Provider objects without leaking reflection into core paths."""

    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.stream_agent = provider.stream_agent  # type: ignore[attr-defined]
        self.parameters = _parameters(self.stream_agent)
        self.accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in self.parameters.values()
        )

    @property
    def supports_structured_requests(self) -> bool:
        return "request" in self.parameters

    @property
    def supports_system_context(self) -> bool:
        return self.supports_structured_requests or "system_instruction" in self.parameters

    def stream(
        self,
        request: AgentRequest,
        *,
        legacy_system_instruction: str = "",
        preserve_tool_tuple: bool = False,
    ) -> Iterator[AgentStreamEvent]:
        kwargs: dict[str, object] = {}
        if "cancel_event" in self.parameters or self.accepts_kwargs:
            kwargs["cancel_event"] = request.cancel_event
        if self.supports_structured_requests:
            kwargs["request"] = request
            tools: tuple[ToolDefinition, ...] | list[ToolDefinition] = request.tools
        elif request.system_prompt != AGENT_SYSTEM_PROMPT or request.output_token_limit is not None:
            raise ProviderCapabilityError()
        elif "system_instruction" in self.parameters:
            kwargs["system_instruction"] = request.system_supplement or legacy_system_instruction
            tools = request.tools if preserve_tool_tuple else list(request.tools)
        else:
            if request.system_supplement or legacy_system_instruction:
                raise ProviderCapabilityError()
            tools = request.tools if preserve_tool_tuple else list(request.tools)
        return self.stream_agent(request.messages, tools, **kwargs)


def invoke_provider_stream(
    provider: object,
    request: AgentRequest,
    *,
    legacy_system_instruction: str = "",
    preserve_tool_tuple: bool = False,
) -> Iterator[AgentStreamEvent]:
    """Use the canonical request contract, falling back only through the adapter."""

    stream_request = getattr(provider, "stream_agent_request", None)
    if callable(stream_request):
        return stream_request(request)
    return LegacyAgentProviderAdapter(provider).stream(
        request,
        legacy_system_instruction=legacy_system_instruction,
        preserve_tool_tuple=preserve_tool_tuple,
    )


def provider_supports_structured_requests(provider: object) -> bool:
    """Report whether all AgentRequest fields have an explicit Provider channel."""

    if callable(getattr(provider, "stream_agent_request", None)):
        return True
    try:
        return LegacyAgentProviderAdapter(provider).supports_structured_requests
    except AttributeError:
        return False


def provider_supports_system_context(provider: object) -> bool:
    """Report whether dynamic system content has an explicit Provider channel."""

    if callable(getattr(provider, "stream_agent_request", None)):
        return True
    try:
        return LegacyAgentProviderAdapter(provider).supports_system_context
    except AttributeError:
        return False


def is_agent_provider(provider: object) -> bool:
    """Distinguish agent-capable Providers from chat-only compatibility objects."""

    return callable(getattr(provider, "stream_agent_request", None)) or callable(
        getattr(provider, "stream_agent", None)
    )


def _parameters(callable_object: object) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError):
        return {}
