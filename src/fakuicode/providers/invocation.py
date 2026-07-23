"""Explicit capability adaptation for provider agent-stream calls."""

from __future__ import annotations

import inspect
from typing import Iterator

from fakuicode.errors import ProviderCapabilityError
from fakuicode.models import AgentStreamEvent, ToolDefinition
from fakuicode.providers.base import AGENT_SYSTEM_PROMPT, AgentRequest


def invoke_provider_stream(
    provider: object,
    request: AgentRequest,
    *,
    legacy_system_instruction: str = "",
    preserve_tool_tuple: bool = False,
) -> Iterator[AgentStreamEvent]:
    """Invoke a Provider only through an explicitly declared system channel."""

    stream_agent = provider.stream_agent  # type: ignore[attr-defined]
    parameters = _parameters(stream_agent)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, object] = {}
    if "cancel_event" in parameters or accepts_kwargs:
        kwargs["cancel_event"] = request.cancel_event
    if "request" in parameters:
        kwargs["request"] = request
        tools: tuple[ToolDefinition, ...] | list[ToolDefinition] = request.tools
    elif request.system_prompt != AGENT_SYSTEM_PROMPT or request.output_token_limit is not None:
        raise ProviderCapabilityError()
    elif "system_instruction" in parameters:
        kwargs["system_instruction"] = request.system_supplement or legacy_system_instruction
        tools = request.tools if preserve_tool_tuple else list(request.tools)
    else:
        if request.system_supplement or legacy_system_instruction:
            raise ProviderCapabilityError()
        tools = request.tools if preserve_tool_tuple else list(request.tools)
    return stream_agent(request.messages, tools, **kwargs)


def _parameters(callable_object: object) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError):
        return {}
