"""Tests for explicit Provider system-channel capability detection."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Event

import pytest

from fakuicode.errors import ProviderCapabilityError
from fakuicode.models import AgentMessage, AgentStreamEvent
from fakuicode.providers.base import AgentRequest
from fakuicode.providers.invocation import invoke_provider_stream


class _StructuredProvider:
    def __init__(self) -> None:
        self.request: AgentRequest | None = None

    def stream_agent(self, messages: object, tools: object, *, request: AgentRequest) -> Iterator[AgentStreamEvent]:
        self.request = request
        yield AgentStreamEvent("completed")


class _SystemProvider:
    def __init__(self) -> None:
        self.system_instruction: str | None = None

    def stream_agent(
        self,
        messages: object,
        tools: object,
        *,
        cancel_event: object = None,
        system_instruction: str = "",
    ) -> Iterator[AgentStreamEvent]:
        self.system_instruction = system_instruction
        yield AgentStreamEvent("completed")


class _KwargsProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: dict[str, object] = {}

    def stream_agent(self, messages: object, tools: object, **kwargs: object) -> Iterator[AgentStreamEvent]:
        self.calls += 1
        self.kwargs = kwargs
        yield AgentStreamEvent("completed")


class _LegacyProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream_agent(self, messages: object, tools: object, *, cancel_event: object = None) -> Iterator[AgentStreamEvent]:
        self.calls += 1
        yield AgentStreamEvent("completed")


def _request(supplement: str = "project instructions") -> AgentRequest:
    return AgentRequest((AgentMessage("user", "hello"),), (), system_supplement=supplement)


def test_invocation_prefers_an_explicit_structured_request_parameter() -> None:
    provider = _StructuredProvider()
    request = _request()

    assert list(invoke_provider_stream(provider, request))[-1].kind == "completed"
    assert provider.request is request


def test_invocation_uses_only_an_explicit_system_instruction_parameter_as_fallback() -> None:
    provider = _SystemProvider()

    list(invoke_provider_stream(provider, _request()))

    assert provider.system_instruction == "project instructions"


@pytest.mark.parametrize("provider_type", [_KwargsProvider, _LegacyProvider])
def test_invocation_rejects_nonempty_system_content_without_a_proven_channel(
    provider_type: type[_KwargsProvider] | type[_LegacyProvider],
) -> None:
    provider = provider_type()

    with pytest.raises(ProviderCapabilityError):
        list(invoke_provider_stream(provider, _request()))

    assert provider.calls == 0


def test_invocation_keeps_the_legacy_path_for_an_empty_system_supplement() -> None:
    provider = _LegacyProvider()

    list(invoke_provider_stream(provider, _request("")))

    assert provider.calls == 1


def test_invocation_preserves_cancellation_for_a_kwargs_legacy_provider() -> None:
    provider = _KwargsProvider()
    cancel_event = Event()
    request = AgentRequest(
        (AgentMessage("user", "hello"),),
        (),
        system_supplement="",
        cancel_event=cancel_event,
    )

    list(invoke_provider_stream(provider, request))

    assert provider.kwargs == {"cancel_event": cancel_event}


def test_invocation_rejects_request_only_constraints_without_a_structured_channel() -> None:
    provider = _SystemProvider()
    request = AgentRequest(
        (AgentMessage("user", "hello"),),
        (),
        system_prompt="summary-only system prompt",
        output_token_limit=4_000,
    )

    with pytest.raises(ProviderCapabilityError):
        list(invoke_provider_stream(provider, request))

    assert provider.system_instruction is None


def test_invocation_rejects_legacy_system_instruction_without_a_declared_channel() -> None:
    provider = _LegacyProvider()

    with pytest.raises(ProviderCapabilityError):
        list(invoke_provider_stream(provider, _request(""), legacy_system_instruction="plan mode"))

    assert provider.calls == 0
