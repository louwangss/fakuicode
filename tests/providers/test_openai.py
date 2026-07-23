from __future__ import annotations

import json

import httpx
import pytest


API_KEY = "openai-test-key-must-not-leak"
SERVER_DETAIL = "openai-server-detail-must-not-leak"


def make_provider(handler: httpx.MockTransport) -> object:
    from fakuicode.models import ProviderConfig
    from fakuicode.providers.openai import OpenAIProvider

    return OpenAIProvider(
        ProviderConfig("openai", "gpt-test", "https://api.openai.com/v1", API_KEY),
        httpx.Client(transport=handler),
    )


def test_openai_provider_builds_request_and_maps_sse_text_to_unified_events() -> None:
    from fakuicode.models import Message

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        assert json.loads(request.content) == {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n')

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_chat([Message("user", "hello")]))

    assert [(event.kind, event.text) for event in events] == [("text_delta", "Hi"), ("completed", "")]


@pytest.mark.parametrize(
    "response, expected_message",
    [
        (httpx.Response(429, content=f'{{"error":"{SERVER_DETAIL}"}}'), "response failed"),
        (httpx.Response(200, content=b"data: not-json\n\n"), "stream format failed"),
        (
            httpx.Response(200, content=f'event: error\ndata: {{"error":"{SERVER_DETAIL}"}}\n\ndata: [DONE]\n\n'.encode()),
            "stream reported an error",
        ),
        (
            httpx.Response(200, content=f'data: {{"error":{{"message":"{SERVER_DETAIL}"}}}}\n\ndata: [DONE]\n\n'.encode()),
            "stream reported an error",
        ),
    ],
)
def test_openai_provider_exposes_safe_errors(response: httpx.Response, expected_message: str) -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    provider = make_provider(httpx.MockTransport(lambda request: response))

    with pytest.raises(ProviderError, match=expected_message) as error:
        list(provider.stream_chat([Message("user", "hello")]))

    assert API_KEY not in str(error.value)
    assert SERVER_DETAIL not in str(error.value)


def test_openai_provider_rejects_stream_that_ends_without_done() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, content=b'data: {"choices":[]}\n\n')))

    with pytest.raises(ProviderError, match="before completion"):
        list(provider.stream_chat([Message("user", "hello")]))


def test_openai_provider_marks_rate_limits_as_retryable() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(429, content=b"{}")))
    with pytest.raises(ProviderError) as error:
        list(provider.stream_chat([Message("user", "hello")]))
    assert error.value.retryable is True


def test_openai_provider_streams_native_tool_calls() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolDefinition
    from fakuicode.providers.base import AGENT_SYSTEM_PROMPT

    stream = b'''data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"REA"}},{"index":1,"id":"call-2","type":"function","function":{"name":"find_files","arguments":"{\\"pattern\\":\\"**/*."}}]}}]}\n\ndata: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"DME.md\\"}"}},{"index":1,"function":{"arguments":"py\\"}"}}]}}]}\n\ndata: [DONE]\n\n'''
    tool = ToolDefinition("read_file", "Read a UTF-8 file.", {"type": "object"})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] == [
            {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}}
        ]
        assert body["messages"] == [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "read README"},
        ]
        return httpx.Response(200, content=stream)

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_agent([AgentMessage("user", "read README")], [tool]))

    assert [(event.kind, event.tool_call) for event in events] == [
        ("tool_call", ToolCall("call-1", "read_file", {"path": "README.md"})),
        ("tool_call", ToolCall("call-2", "find_files", {"pattern": "**/*.py"})),
        ("completed", None),
    ]


def test_openai_provider_translates_streamed_dsml_tool_markup() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolDefinition

    markup = (
        '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="list_dir">'
        '<｜｜DSML｜｜parameter name="dirPath" string="true">.</｜｜DSML｜｜parameter>'
        '</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>'
    )
    stream = (
        f"data: {json.dumps({'choices': [{'delta': {'content': 'I will inspect. ' + markup[:30]}}]})}\n\n"
        f"data: {json.dumps({'choices': [{'delta': {'content': markup[30:]}}]})}\n\n"
        "data: [DONE]\n\n"
    ).encode()
    tool = ToolDefinition("find_files", "Find files.", {"type": "object"})

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, content=stream)))
    events = list(provider.stream_agent([AgentMessage("user", "list project files")], [tool]))

    assert [(event.kind, event.tool_call) for event in events] == [
        ("text_delta", None),
        ("tool_call", ToolCall("dsml-1", "find_files", {"pattern": "**/*", "path": "."})),
        ("completed", None),
    ]
    assert events[0].text == "I will inspect. "


def test_openai_provider_serializes_native_tool_history() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolResult

    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    result = ToolResult("call-1", "read_file", True, "contents", "read README")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][1:] == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": result.to_model_content()},
        ]
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n')

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_agent([AgentMessage("assistant", tool_calls=(call,)), AgentMessage("user", tool_results=(result,))], []))

    assert [(event.kind, event.text) for event in events] == [("text_delta", "answer"), ("completed", "")]


def test_openai_provider_omits_tools_for_a_tool_free_summary_turn() -> None:
    from fakuicode.models import AgentMessage

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" not in body
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"summary"}}]}\n\ndata: [DONE]\n\n')

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_agent([AgentMessage("user", "summarize")], []))
    assert [(event.kind, event.text) for event in events] == [("text_delta", "summary"), ("completed", "")]


def test_openai_agent_stream_uses_voluntary_usage_without_requesting_stream_options() -> None:
    from fakuicode.models import AgentMessage, TokenUsage
    from fakuicode.providers.base import AGENT_SYSTEM_PROMPT

    stream = b'''data: {"choices":[{"delta":{"content":"done"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}\n\ndata: [DONE]\n\n'''

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "stream_options" not in body
        assert body["messages"][0] == {
            "role": "system",
            "content": f"{AGENT_SYSTEM_PROMPT}\n\nPlan using read-only tools.",
        }
        return httpx.Response(200, content=stream)

    provider = make_provider(httpx.MockTransport(handler))
    events = list(
        provider.stream_agent(
            [AgentMessage("user", "inspect")],
            [],
            system_instruction="Plan using read-only tools.",
        )
    )

    assert [(event.kind, event.usage) for event in events] == [
        ("text_delta", None),
        ("usage", TokenUsage(input_tokens=11, output_tokens=5, context_input_tokens=11)),
        ("completed", None),
    ]


def test_openai_structured_request_keeps_stable_system_message_first_and_parses_cached_tokens() -> None:
    from fakuicode.models import AgentMessage, TokenUsage
    from fakuicode.providers.base import AgentRequest

    stream = b'''data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":9}}}\n\ndata: [DONE]\n\n'''

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "system", "content": "stable"},
            {"role": "system", "content": "<system-reminder>dynamic</system-reminder>"},
            {"role": "user", "content": "inspect"},
        ]
        assert "prompt_cache_key" not in body
        assert "ttl" not in body
        return httpx.Response(200, content=stream)

    provider = make_provider(httpx.MockTransport(handler))
    events = list(
        provider.stream_agent(
            [AgentMessage("user", "inspect")],
            [],
            request=AgentRequest(
                (AgentMessage("user", "inspect"),), (), "stable", "<system-reminder>dynamic</system-reminder>"
            ),
        )
    )

    assert [event.usage for event in events if event.kind == "usage"] == [
        TokenUsage(11, 5, 9, None, context_input_tokens=11)
    ]


def test_official_openai_maps_internal_output_limit_to_max_completion_tokens() -> None:
    from fakuicode.models import AgentMessage
    from fakuicode.providers.base import AgentRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["max_completion_tokens"] == 4_000
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    provider = make_provider(httpx.MockTransport(handler))
    assert provider.capabilities.supports_output_token_limit is True

    list(
        provider.stream_agent(
            [AgentMessage("user", "summarize")],
            [],
            request=AgentRequest(
                (AgentMessage("user", "summarize"),),
                (),
                output_token_limit=4_000,
            ),
        )
    )


def test_unknown_openai_compatible_endpoint_does_not_receive_output_limit() -> None:
    from fakuicode.models import AgentMessage, ProviderConfig
    from fakuicode.providers.base import AgentRequest
    from fakuicode.providers.openai import OpenAIProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert "max_completion_tokens" not in json.loads(request.content)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    provider = OpenAIProvider(
        ProviderConfig(
            "openai",
            "compatible-model",
            "https://compatible.example/v1",
            API_KEY,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.capabilities.supports_output_token_limit is False

    list(
        provider.stream_agent(
            [AgentMessage("user", "summarize")],
            [],
            request=AgentRequest(
                (AgentMessage("user", "summarize"),),
                (),
                output_token_limit=4_000,
            ),
        )
    )


@pytest.mark.parametrize(
    ("payload", "expected_category"),
    [
        (
            {
                "error": {
                    "code": "context_length_exceeded",
                    "message": f"maximum context length exceeded: {SERVER_DETAIL}",
                }
            },
            "context_overflow",
        ),
        (
            {
                "error": {
                    "code": "invalid_value",
                    "message": f"another invalid request: {SERVER_DETAIL}",
                }
            },
            "other",
        ),
    ],
)
def test_openai_classifies_only_known_context_overflow_errors(
    payload: dict[str, object], expected_category: str
) -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage

    provider = make_provider(
        httpx.MockTransport(lambda request: httpx.Response(400, json=payload))
    )

    with pytest.raises(ProviderError) as error:
        list(provider.stream_agent([AgentMessage("user", "hello")], []))

    assert error.value.category == expected_category
    assert SERVER_DETAIL not in str(error.value)
