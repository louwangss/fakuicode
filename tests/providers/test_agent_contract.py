from __future__ import annotations

from pathlib import Path

import httpx


def test_provider_errors_expose_machine_readable_categories_without_response_bodies() -> None:
    from fakuicode.errors import ProviderError, RequestCancelled

    ordinary = ProviderError("safe failure")
    compatible_transient = ProviderError("try again", retryable=True)
    transient = ProviderError("try again", category="transient")
    overflow = ProviderError("context is too large", category="context_overflow")

    assert (ordinary.category, ordinary.retryable) == ("other", False)
    assert (compatible_transient.category, compatible_transient.retryable) == (
        "transient",
        True,
    )
    assert (transient.category, transient.retryable) == ("transient", True)
    assert (overflow.category, overflow.retryable) == ("context_overflow", False)
    assert "raw-provider-body" not in str(overflow)
    assert not isinstance(RequestCancelled(), ProviderError)


def test_internal_requests_can_ask_for_a_provider_neutral_output_limit() -> None:
    from fakuicode.models import AgentMessage
    from fakuicode.providers.base import AgentRequest, ProviderCapabilities

    ordinary = AgentRequest((AgentMessage("user", "hello"),), ())
    summary = AgentRequest(
        (AgentMessage("user", "summarize"),),
        (),
        output_token_limit=4_000,
    )

    assert ordinary.output_token_limit is None
    assert summary.output_token_limit == 4_000
    assert ProviderCapabilities().supports_output_token_limit is False
    assert ProviderCapabilities(supports_output_token_limit=True).supports_output_token_limit is True


def test_agent_message_keeps_tool_calls_and_results_in_provider_neutral_form() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolResult

    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    result = ToolResult("call-1", "read_file", True, "contents", "read README.md")

    assistant = AgentMessage("assistant", tool_calls=(call,))
    user = AgentMessage("user", tool_results=(result,))

    assert assistant.tool_calls == (call,)
    assert user.tool_results == (result,)


def test_agent_stream_event_carries_native_tool_call() -> None:
    from fakuicode.models import AgentStreamEvent, ToolCall

    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    event = AgentStreamEvent("tool_call", tool_call=call)

    assert event.kind == "tool_call"
    assert event.tool_call == call


def _native_write_call(protocol: str):
    from fakuicode.models import AgentMessage, ProviderConfig, ToolDefinition
    from fakuicode.providers.anthropic import AnthropicProvider
    from fakuicode.providers.base import AgentRequest
    from fakuicode.providers.openai import OpenAIProvider

    if protocol == "anthropic":
        stream = b'''event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"anthropic-call","name":"write_file","input":{}}}\n\nevent: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"notes.txt\\",\\"content\\":\\"provider secret\\"}"}}\n\nevent: content_block_stop\ndata: {"index":0}\n\nevent: message_stop\ndata: {}\n\n'''
        provider = AnthropicProvider(
            ProviderConfig("anthropic", "claude-test", "https://api.anthropic.com/v1", "test-key"),
            httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=stream))),
        )
    else:
        stream = b'''data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"openai-call","type":"function","function":{"name":"write_file","arguments":"{\\"path\\":\\"notes.txt\\",\\"content\\":\\"provider secret\\"}"}}]}}]}\n\ndata: [DONE]\n\n'''
        provider = OpenAIProvider(
            ProviderConfig("openai", "gpt-test", "https://api.openai.com/v1", "test-key"),
            httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=stream))),
        )
    events = provider.stream_agent_request(
        AgentRequest(
            (AgentMessage("user", "write a note"),),
            (ToolDefinition("write_file", "Write a file.", {"type": "object"}),),
        )
    )
    return next(event.tool_call for event in events if event.kind == "tool_call")


def test_anthropic_and_openai_calls_receive_equivalent_permission_results(tmp_path: Path) -> None:
    from fakuicode.permissions.config import PermissionConfigSnapshot
    from fakuicode.permissions.manager import PermissionManager
    from fakuicode.permissions.safety import DangerousCommandGuard
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    calls = (_native_write_call("anthropic"), _native_write_call("openai"))
    assert calls[0] is not None and calls[1] is not None
    assert (calls[0].name, calls[0].arguments) == (calls[1].name, calls[1].arguments)

    results = []
    for index, call in enumerate(calls):
        workspace = tmp_path / str(index)
        workspace.mkdir()
        permissions = PermissionManager(
            PermissionConfigSnapshot(),
            DangerousCommandGuard(workspace),
        )
        result = ToolRegistry(
            WorkspacePolicy(workspace), permission_manager=permissions
        ).execute(call)
        results.append(result)
        assert not (workspace / "notes.txt").exists()

    assert [(result.success, result.output, result.summary) for result in results] == [
        (False, "Permission denied: The user denied this action.", "permission denied"),
        (False, "Permission denied: The user denied this action.", "permission denied"),
    ]
    assert "provider secret" not in "".join(result.output for result in results)
