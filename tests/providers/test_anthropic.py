from __future__ import annotations

import json

import httpx
import pytest


API_KEY = "anthropic-test-key-must-not-leak"
SERVER_DETAIL = "anthropic-server-detail-must-not-leak"


def make_provider(handler: httpx.MockTransport) -> object:
    from fakuicode.models import ProviderConfig, ThinkingConfig
    from fakuicode.providers.anthropic import AnthropicProvider

    config = ProviderConfig("anthropic", "claude-test", "https://api.anthropic.com/v1", API_KEY, ThinkingConfig(True))
    return AnthropicProvider(config, httpx.Client(transport=handler))


def make_deepseek_provider(handler: httpx.MockTransport) -> object:
    from fakuicode.models import ProviderConfig, ThinkingConfig
    from fakuicode.providers.anthropic import AnthropicProvider

    config = ProviderConfig(
        "anthropic",
        "deepseek-v4-flash",
        "https://api.deepseek.com/anthropic",
        API_KEY,
        ThinkingConfig(True),
    )
    return AnthropicProvider(config, httpx.Client(transport=handler))


def test_anthropic_provider_builds_adaptive_thinking_request_and_maps_events() -> None:
    from fakuicode.models import Message

    stream = b'''event: content_block_start\ndata: {"content_block":{"type":"thinking"}}\n\nevent: content_block_delta\ndata: {"delta":{"type":"thinking_delta","thinking":"reason"}}\n\nevent: content_block_stop\ndata: {}\n\nevent: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"answer"}}\n\nevent: message_stop\ndata: {}\n\n'''

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == API_KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert json.loads(request.content) == {
            "model": "claude-test",
            "max_tokens": 4096,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        return httpx.Response(200, content=stream)

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_chat([Message("user", "hello")]))

    assert [(event.kind, event.text) for event in events] == [
        ("thinking_start", ""),
        ("thinking_delta", "reason"),
        ("thinking_end", ""),
        ("text_delta", "answer"),
        ("completed", ""),
    ]


@pytest.mark.parametrize(
    "response, expected_message",
    [
        (httpx.Response(401, content=f'{{"error":"{SERVER_DETAIL}"}}'), "response failed"),
        (
            httpx.Response(
                200,
                content=f'event: error\ndata: {{"error":"{SERVER_DETAIL}"}}\n\nevent: message_stop\ndata: {{}}\n\n'.encode(),
            ),
            "stream reported an error",
        ),
    ],
)
def test_anthropic_provider_exposes_safe_errors(response: httpx.Response, expected_message: str) -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    provider = make_provider(httpx.MockTransport(lambda request: response))

    with pytest.raises(ProviderError, match=expected_message) as error:
        list(provider.stream_chat([Message("user", "hello")]))

    assert API_KEY not in str(error.value)
    assert SERVER_DETAIL not in str(error.value)


def test_anthropic_provider_rejects_stream_that_ends_without_message_stop() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, content=b"event: ping\ndata: {}\n\n")))

    with pytest.raises(ProviderError, match="before completion"):
        list(provider.stream_chat([Message("user", "hello")]))


def test_anthropic_provider_marks_authentication_failure_as_non_retryable() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(401, content=b"{}")))
    with pytest.raises(ProviderError) as error:
        list(provider.stream_chat([Message("user", "hello")]))
    assert error.value.retryable is False


def test_anthropic_provider_exposes_structured_http_diagnostics_without_server_detail() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    response = httpx.Response(
        400,
        headers={"request-id": "req_safe-123"},
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": SERVER_DETAIL,
            },
        },
    )
    provider = make_provider(httpx.MockTransport(lambda request: response))

    with pytest.raises(ProviderError) as captured:
        list(provider.stream_chat([Message("user", "hello")]))

    error = captured.value
    assert error.status_code == 400
    assert error.error_type == "invalid_request_error"
    assert error.failure_phase == "http_status"
    assert error.request_id == "req_safe-123"
    assert error.retryable is False
    assert SERVER_DETAIL not in str(error)


def test_anthropic_provider_classifies_retryable_stream_error_without_exposing_payload() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    stream = (
        "event: error\n"
        f'data: {{"type":"error","error":{{"type":"overloaded_error","message":"{SERVER_DETAIL}"}}}}\n\n'
    ).encode()
    provider = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"x-request-id": "req_stream-456"},
                content=stream,
            )
        )
    )

    with pytest.raises(ProviderError) as captured:
        list(provider.stream_chat([Message("user", "hello")]))

    error = captured.value
    assert error.status_code is None
    assert error.error_type == "overloaded_error"
    assert error.failure_phase == "stream_event"
    assert error.request_id == "req_stream-456"
    assert error.retryable is True
    assert SERVER_DETAIL not in str(error)


def test_anthropic_provider_drops_untrusted_diagnostic_identifiers() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import Message

    response = httpx.Response(
        400,
        headers={"request-id": "unsafe request id with spaces"},
        json={"error": {"type": SERVER_DETAIL, "message": SERVER_DETAIL}},
    )
    provider = make_provider(httpx.MockTransport(lambda request: response))

    with pytest.raises(ProviderError) as captured:
        list(provider.stream_chat([Message("user", "hello")]))

    error = captured.value
    assert error.error_type == "unknown_error"
    assert error.request_id is None
    assert SERVER_DETAIL not in str(error)


def test_anthropic_provider_streams_native_tool_calls() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolDefinition
    from fakuicode.providers.base import AGENT_SYSTEM_PROMPT

    stream = b'''event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tool-1","name":"read_file","input":{}}}\n\nevent: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"REA"}}\n\nevent: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"DME.md\\"}"}}\n\nevent: content_block_stop\ndata: {"index":0}\n\nevent: content_block_start\ndata: {"index":1,"content_block":{"type":"tool_use","id":"tool-2","name":"find_files","input":{}}}\n\nevent: content_block_delta\ndata: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"pattern\\":\\"**/*.py\\"}"}}\n\nevent: content_block_stop\ndata: {"index":1}\n\nevent: message_stop\ndata: {}\n\n'''

    tool = ToolDefinition(
        "read_file",
        "Read a UTF-8 file.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["system"] == AGENT_SYSTEM_PROMPT
        assert body["tools"] == [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        ]
        assert body["messages"] == [{"role": "user", "content": "read README"}]
        return httpx.Response(200, content=stream)

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_agent([AgentMessage("user", "read README")], [tool]))

    assert [(event.kind, event.tool_call) for event in events] == [
        ("tool_call", ToolCall("tool-1", "read_file", {"path": "README.md"})),
        ("tool_call", ToolCall("tool-2", "find_files", {"pattern": "**/*.py"})),
        ("completed", None),
    ]


def test_anthropic_provider_marks_incomplete_tool_json_for_agent_recovery() -> None:
    from fakuicode.models import AgentMessage

    stream = b'''event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tool-1","name":"write_file","input":{}}}\n\nevent: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"index.html\\",\\"content\\":\\"unfinished"}}\n\nevent: content_block_stop\ndata: {"index":0}\n\nevent: message_delta\ndata: {"delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4096}}\n\nevent: message_stop\ndata: {}\n\n'''
    provider = make_deepseek_provider(
        httpx.MockTransport(lambda request: httpx.Response(200, content=stream))
    )

    events = list(provider.stream_agent([AgentMessage("user", "build")], []))

    call = next(event.tool_call for event in events if event.tool_call is not None)
    assert call.name == "write_file"
    assert call.arguments == {}
    assert call.argument_error == "invalid_json"
    assert events[-1].kind == "completed"


def test_anthropic_provider_translates_dsml_tool_markup_after_visible_text() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolDefinition

    markup = (
        '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="read_file">'
        '<｜｜DSML｜｜parameter name="filepath" string="true">README.md</｜｜DSML｜｜parameter>'
        '</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>'
    )
    stream = (
        f"event: content_block_delta\ndata: {json.dumps({'delta': {'type': 'text_delta', 'text': 'I will inspect. ' + markup[:40]}})}\n\n"
        f"event: content_block_delta\ndata: {json.dumps({'delta': {'type': 'text_delta', 'text': markup[40:]}})}\n\n"
        "event: message_stop\ndata: {}\n\n"
    ).encode()
    tool = ToolDefinition("read_file", "Read a file.", {"type": "object"})

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, content=stream)))
    events = list(provider.stream_agent([AgentMessage("user", "inspect README")], [tool]))

    assert [(event.kind, event.tool_call) for event in events] == [
        ("text_delta", None),
        ("tool_call", ToolCall("dsml-1", "read_file", {"path": "README.md"})),
        ("completed", None),
    ]
    assert events[0].text == "I will inspect. "


def test_anthropic_provider_translates_a_dsml_root_file_listing_to_recursive_discovery() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolDefinition

    markup = (
        '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="find_files">'
        '<｜｜DSML｜｜parameter name="pattern" string="true">*</｜｜DSML｜｜parameter>'
        '</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>'
    )
    stream = (
        f"event: content_block_delta\ndata: {json.dumps({'delta': {'type': 'text_delta', 'text': markup}})}\n\n"
        "event: message_stop\ndata: {}\n\n"
    ).encode()
    tool = ToolDefinition("find_files", "Find files.", {"type": "object"})

    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, content=stream)))
    events = list(provider.stream_agent([AgentMessage("user", "list project files")], [tool]))

    assert [(event.kind, event.tool_call) for event in events] == [
        ("tool_call", ToolCall("dsml-1", "find_files", {"pattern": "**/*"})),
        ("completed", None),
    ]


def test_anthropic_provider_serializes_native_tool_history() -> None:
    from fakuicode.models import AgentMessage, ToolCall, ToolResult

    call = ToolCall("tool-1", "read_file", {"path": "README.md"})
    result = ToolResult("tool-1", "read_file", False, "not found", "file missing")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"] == [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "README.md"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": result.to_model_content(),
                        "is_error": True,
                    }
                ],
            },
        ]
        assert "thinking" not in body
        return httpx.Response(200, content=b'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"answer"}}\n\nevent: message_stop\ndata: {}\n\n')

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_agent([AgentMessage("assistant", tool_calls=(call,)), AgentMessage("user", tool_results=(result,))], []))

    assert [(event.kind, event.text) for event in events] == [("text_delta", "answer"), ("completed", "")]


def test_anthropic_agent_round_trips_unmodified_thinking_with_tool_results() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, ToolDefinition, ToolResult

    first_stream = b'''event: content_block_start\ndata: {"index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n\nevent: content_block_delta\ndata: {"index":0,"delta":{"type":"thinking_delta","thinking":"inspect first"}}\n\nevent: content_block_delta\ndata: {"index":0,"delta":{"type":"signature_delta","signature":"signed-reasoning"}}\n\nevent: content_block_stop\ndata: {"index":0}\n\nevent: content_block_start\ndata: {"index":1,"content_block":{"type":"redacted_thinking","data":"encrypted-reasoning"}}\n\nevent: content_block_stop\ndata: {"index":1}\n\nevent: content_block_start\ndata: {"index":2,"content_block":{"type":"tool_use","id":"tool-1","name":"read_file","input":{}}}\n\nevent: content_block_delta\ndata: {"index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\"}"}}\n\nevent: content_block_stop\ndata: {"index":2}\n\nevent: message_stop\ndata: {}\n\n'''
    final_stream = b'''event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"README loaded"}}\n\nevent: message_stop\ndata: {}\n\n'''
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, content=first_stream)
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["messages"][1]["content"] == [
            {
                "type": "thinking",
                "thinking": "inspect first",
                "signature": "signed-reasoning",
            },
            {
                "type": "redacted_thinking",
                "data": "encrypted-reasoning",
            },
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "read_file",
                "input": {"path": "README.md"},
            },
        ]
        return httpx.Response(200, content=final_stream)

    class Tools:
        def definitions(self, *, read_only_only: bool = False) -> list[ToolDefinition]:
            del read_only_only
            return [ToolDefinition("read_file", "Read a file.", {"type": "object"})]

        def is_known(self, name: str) -> bool:
            return name == "read_file"

        def is_read_only(self, name: str) -> bool:
            return name == "read_file"

        def execute(self, call, *, cancel_event=None, read_only_only=False) -> ToolResult:
            del cancel_event, read_only_only
            return ToolResult(call.id, call.name, True, "contents", "read README.md")

    provider = make_provider(httpx.MockTransport(handler))
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "inspect README")]))

    assert events[-1].kind == "completed"
    assert len(requests) == 2


def test_deepseek_anthropic_agent_continues_after_a_failed_tool_result() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    command = ["cmd", "/c", "mkdir", r"test\ecommerce\frontend\js"]
    first_events = [
        ("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "create folders"}}),
        ("content_block_stop", {"index": 0}),
        (
            "content_block_start",
            {
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "run_command",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"command": command}),
                },
            },
        ),
        ("content_block_stop", {"index": 1}),
        ("message_stop", {}),
    ]
    first_stream = "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n"
        for name, payload in first_events
    ).encode()
    final_stream = (
        'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"改用 write_file 创建文件。"}}\n\n'
        "event: message_stop\ndata: {}\n\n"
    ).encode()
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, content=first_stream)
        if body.get("thinking") != {"type": "enabled"}:
            return httpx.Response(400, json={"error": {"message": "invalid thinking mode"}})
        assistant = body["messages"][1]["content"]
        assert assistant[0] == {"type": "thinking", "thinking": "create folders"}
        assert assistant[1] == {
            "type": "tool_use",
            "id": "tool-1",
            "name": "run_command",
            "input": {"command": command},
        }
        assert body["messages"][2]["content"][0]["is_error"] is True
        return httpx.Response(200, content=final_stream)

    class Tools:
        def definitions(self, *, read_only_only: bool = False) -> list[ToolDefinition]:
            del read_only_only
            return [ToolDefinition("run_command", "Run a command.", {"type": "object"})]

        def is_known(self, name: str) -> bool:
            return name == "run_command"

        def is_read_only(self, name: str) -> bool:
            del name
            return False

        def execute(self, call: ToolCall, *, cancel_event=None, read_only_only=False) -> ToolResult:
            del cancel_event, read_only_only
            return ToolResult(
                call.id,
                call.name,
                False,
                "Permission denied: direct use of a general shell is blocked.",
                "permission denied",
            )

    provider = make_deepseek_provider(httpx.MockTransport(handler))
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "build the page")]))

    assert [event.tool_result.success for event in events if event.tool_result is not None] == [False]
    assert any(event.kind == "text_delta" and "write_file" in event.text for event in events)
    assert events[-1] == AgentStreamEvent("completed")
    assert len(requests) == 2
    assert all(request["thinking"] == {"type": "enabled"} for request in requests)


def test_deepseek_anthropic_agent_accepts_non_tool_block_stop_without_index() -> None:
    from fakuicode.models import AgentMessage, AgentStreamEvent

    stream = b'''event: content_block_start\ndata: {"content_block":{"type":"text","text":""}}\n\nevent: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"continue after tool"}}\n\nevent: content_block_stop\ndata: {}\n\nevent: message_stop\ndata: {}\n\n'''
    provider = make_deepseek_provider(
        httpx.MockTransport(lambda request: httpx.Response(200, content=stream))
    )

    events = list(provider.stream_agent([AgentMessage("user", "continue")], []))

    assert events == [
        AgentStreamEvent("text_delta", "continue after tool"),
        AgentStreamEvent("completed"),
    ]


def test_anthropic_provider_omits_tools_for_a_tool_free_summary_turn() -> None:
    from fakuicode.models import AgentMessage

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" not in body
        return httpx.Response(200, content=b'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"summary"}}\n\nevent: message_stop\ndata: {}\n\n')

    provider = make_provider(httpx.MockTransport(handler))
    events = list(provider.stream_agent([AgentMessage("user", "summarize")], []))
    assert [(event.kind, event.text) for event in events] == [("text_delta", "summary"), ("completed", "")]


def test_anthropic_agent_stream_emits_usage_and_appends_a_mode_instruction() -> None:
    from fakuicode.models import AgentMessage, TokenUsage
    from fakuicode.providers.base import AGENT_SYSTEM_PROMPT

    stream = b'''event: message_start\ndata: {"message":{"usage":{"input_tokens":12,"output_tokens":0}}}\n\nevent: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"plan"}}\n\nevent: message_delta\ndata: {"usage":{"output_tokens":7}}\n\nevent: message_stop\ndata: {}\n\n'''

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["system"] == f"{AGENT_SYSTEM_PROMPT}\n\nPlan using read-only tools."
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
        ("usage", TokenUsage(input_tokens=12, output_tokens=0, context_input_tokens=12)),
        ("text_delta", None),
        ("usage", TokenUsage(input_tokens=12, output_tokens=7, context_input_tokens=12)),
        ("completed", None),
    ]


def test_anthropic_structured_request_uses_a_cacheable_prefix_and_parses_cache_usage() -> None:
    from fakuicode.models import AgentMessage, TokenUsage
    from fakuicode.providers.base import AgentRequest

    stream = b'''event: message_start\ndata: {"message":{"usage":{"input_tokens":12,"cache_creation_input_tokens":8}}}\n\nevent: message_delta\ndata: {"usage":{"output_tokens":7,"cache_read_input_tokens":5}}\n\nevent: message_stop\ndata: {}\n\n'''

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["system"] == [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "<system-reminder>dynamic</system-reminder>"},
        ]
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

    usages = [event.usage for event in events if event.kind == "usage"]
    assert usages == [
        TokenUsage(12, None, None, 8, context_input_tokens=20),
        TokenUsage(12, 7, 5, 8, context_input_tokens=25),
    ]


def test_anthropic_maps_internal_output_limit_to_max_tokens() -> None:
    from fakuicode.models import AgentMessage
    from fakuicode.providers.base import AgentRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["max_tokens"] == 4_000
        return httpx.Response(200, content=b"event: message_stop\ndata: {}\n\n")

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


@pytest.mark.parametrize(
    ("payload", "expected_category"),
    [
        (
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"prompt is too long: {SERVER_DETAIL}",
                },
            },
            "context_overflow",
        ),
        (
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"another invalid request: {SERVER_DETAIL}",
                },
            },
            "other",
        ),
    ],
)
def test_anthropic_classifies_only_known_context_overflow_errors(
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
