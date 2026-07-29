from __future__ import annotations

from collections.abc import Iterator, Sequence
from threading import Event, Thread

import pytest


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[Sequence[object]] = []
        self.agent_requests: list[object] = []
        self.tool_sets: list[Sequence[object]] = []
        self.turn = 0

    def stream_agent(
        self,
        messages: Sequence[object],
        tools: Sequence[object],
        *,
        cancel_event: Event | None = None,
        request: object = None,
    ) -> Iterator[object]:
        from fakuicode.models import AgentStreamEvent, ToolCall

        self.requests.append(messages)
        self.agent_requests.append(request)
        self.tool_sets.append(tools)
        self.turn += 1
        if self.turn == 1:
            yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-1", "read_file", {"path": "README.md"}))
            yield AgentStreamEvent("completed")
            return
        yield AgentStreamEvent("text_delta", "README loaded")
        yield AgentStreamEvent("completed")


class RecordingTools:
    def definitions(self) -> list[object]:
        from fakuicode.models import ToolDefinition

        return [ToolDefinition("read_file", "Read a file.", {"type": "object"})]

    def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
        from fakuicode.models import ToolResult

        return ToolResult("call-1", "read_file", True, "contents", "read README.md")


def test_agent_runner_injects_skill_catalog_and_active_sop_on_every_round() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage

    class Skills:
        catalog_text = "- test: run tests"
        active_prompt = "### Skill: review\nOnly inspect."

        def set_mode(self, mode: str) -> None:
            self.mode = mode

    provider = RecordingProvider()
    skills = Skills()

    list(AgentRunner(provider, RecordingTools(), skill_manager=skills).run([AgentMessage("user", "inspect")]))

    assert skills.mode == "execute"
    assert all("- test: run tests" in request.system_supplement for request in provider.agent_requests)
    assert all("### Skill: review\nOnly inspect." in request.system_supplement for request in provider.agent_requests)


def test_agent_runner_reuses_memory_snapshot_and_only_injects_resume_reminder_once() -> None:
    from uuid import uuid4

    from fakuicode.agent import AgentRunner
    from fakuicode.memory.models import AgentTurnContext, MemorySnapshot
    from fakuicode.models import AgentMessage

    provider = RecordingProvider()
    snapshot = MemorySnapshot(
        "memory sentinel",
        frozenset({str(uuid4())}),
        None,
        "user-digest",
        None,
        (),
    )
    context = AgentTurnContext(snapshot, "resume reminder sentinel")

    events = list(
        AgentRunner(provider, RecordingTools()).run(
            [AgentMessage("user", "inspect")],
            turn_context=context,
        )
    )

    assert events[-1].kind == "completed"
    assert all("memory sentinel" in request.system_supplement for request in provider.agent_requests)
    assert "resume reminder sentinel" in provider.agent_requests[0].system_supplement
    assert "resume reminder sentinel" not in provider.agent_requests[1].system_supplement


def test_agent_runner_injects_automatic_memory_contract_when_the_index_is_empty() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.memory.models import AgentTurnContext, MemorySnapshot
    from fakuicode.models import AgentMessage

    snapshot = MemorySnapshot("", frozenset(), None, "user-digest", None, ())
    request = AgentRunner(RecordingProvider(), RecordingTools()).build_request(
        [AgentMessage("user", "请记住这个偏好")],
        turn_context=AgentTurnContext(snapshot),
    )

    assert "普通回复结束后由宿主异步维护" in request.system_supplement
    assert "不要尝试通过文件工具读写 AGENTS.md" in request.system_supplement


def test_agent_runner_executes_tools_and_continues_until_final_text() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage

    provider = RecordingProvider()
    events = list(
        AgentRunner(
            provider,
            RecordingTools(),
            custom_instructions="project sentinel",
        ).run([AgentMessage("user", "inspect README")])
    )

    assert [event.kind for event in events] == [
        "progress", "tool_call", "progress", "tool_result", "progress", "text_delta", "completed"
    ]
    assert events[3].tool_result is not None and events[3].tool_result.success is True
    assert len(provider.requests) == 2
    second_request = provider.requests[1]
    assert second_request[-2].tool_calls[0].name == "read_file"
    assert second_request[-1].tool_results[0].output == "contents"
    assert [definition.name for definition in provider.tool_sets[0]] == ["read_file"]
    assert [definition.name for definition in provider.tool_sets[1]] == ["read_file"]
    assert all(
        "project sentinel" in request.system_supplement
        for request in provider.agent_requests
    )


def test_agent_runner_injects_custom_instructions_into_a_structured_request() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent

    class Provider:
        def __init__(self) -> None:
            self.request = None

        def stream_agent(self, messages: object, tools: object, *, request: object) -> Iterator[object]:
            self.request = request
            yield AgentStreamEvent("text_delta", "done")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only: bool = False) -> list[object]:
            return []

    provider = Provider()
    list(
        AgentRunner(provider, Tools(), custom_instructions="project sentinel").run(
            [AgentMessage("user", "hello")]
        )
    )

    assert provider.request is not None
    assert "project sentinel" in provider.request.system_supplement


def test_agent_runner_recovers_when_the_model_only_announces_a_missing_tool_call() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolDefinition

    class Provider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []
            self.tool_sets: list[Sequence[object]] = []

        def stream_agent(
            self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None
        ) -> Iterator[object]:
            self.requests.append(messages)
            self.tool_sets.append(tools)
            if len(self.requests) == 1:
                yield AgentStreamEvent("text_delta", "Let me inspect pyproject.toml to confirm the entry point.")
            else:
                assert tools == ()
                assert "no executable tool call" in messages[-1].content
                yield AgentStreamEvent("text_delta", "I cannot verify files without a tool result.")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self) -> list[ToolDefinition]:
            return [ToolDefinition("read_file", "Read a file.", {"type": "object"})]

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("No tool should execute without a tool call")

    provider = Provider()
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "find the entry point")]))

    assert [event.kind for event in events] == ["progress", "text_delta", "completed"]
    assert events[-2].text == "Let me inspect pyproject.toml to confirm the entry point."
    assert len(provider.requests) == 1


def test_agent_runner_recovers_with_plain_text_after_a_follow_up_tool_request() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []
            self.tool_sets: list[Sequence[object]] = []

        def stream_agent(
            self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None
        ) -> Iterator[object]:
            self.requests.append(messages)
            self.tool_sets.append(tools)
            if len(self.requests) == 1:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-1", "read_file", {"path": "one.txt"}))
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-2", "read_file", {"path": "two.txt"}))
                yield AgentStreamEvent("completed")
                return
            if len(self.requests) == 2:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-3", "write_file", {"path": "three.txt", "content": "no"}))
                yield AgentStreamEvent("completed")
                return
            if len(self.requests) == 3:
                assert tools == []
                yield AgentStreamEvent("text_delta", "The two requested files were inspected.")
                yield AgentStreamEvent("completed")
                return
            raise AssertionError("Agent must not issue a fourth provider request")

    class Tools:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: ToolCall, *, cancel_event: Event | None = None) -> ToolResult:
            self.executed.append(call.id)
            return ToolResult(call.id, call.name, True, call.arguments["path"], f"read {call.arguments['path']}")

    provider = Provider()
    tools = Tools()
    events = list(AgentRunner(provider, tools).run([AgentMessage("user", "inspect two files")]))

    assert tools.executed == ["call-1", "call-2", "call-3"]
    assert len(provider.requests) == 3
    assert provider.tool_sets[1] == []
    assert provider.tool_sets[2] == []
    assert [event.kind for event in events] == [
        "progress", "tool_call", "tool_call", "progress", "tool_result", "tool_result",
        "progress", "tool_call", "progress", "tool_result", "progress", "text_delta", "completed",
    ]
    assert events[-2].text == "The two requested files were inspected."
    continuation = provider.requests[1]
    assert [call.id for call in continuation[-2].tool_calls] == ["call-1", "call-2"]
    assert [result.call_id for result in continuation[-1].tool_results] == ["call-1", "call-2"]


def test_agent_runner_recovers_when_the_tool_free_continuation_only_announces_more_file_reading() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []
            self.tool_sets: list[Sequence[object]] = []

        def stream_agent(
            self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None
        ) -> Iterator[object]:
            self.requests.append(messages)
            self.tool_sets.append(tools)
            if len(self.requests) == 1:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-1", "find_files", {"pattern": "**/*"}))
            elif len(self.requests) == 2:
                assert [definition.name for definition in tools] == ["find_files"]
                yield AgentStreamEvent(
                    "text_delta", "发现这是一个 Python 项目，通常入口文件是 __main__.py 或 cli.py。让我查看关键文件。"
                )
            elif len(self.requests) == 3:
                assert tools == ()
                assert "Do not emit DSML" in messages[-1].content
                yield AgentStreamEvent("text_delta", "入口是 pyproject.toml 中的 fakuicode.cli:main。")
            else:
                raise AssertionError("Agent must not issue a fourth provider request")
            yield AgentStreamEvent("completed")

    class Tools:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def definitions(self) -> list[ToolDefinition]:
            return [ToolDefinition("find_files", "Find files.", {"type": "object"})]

        def execute(self, call: ToolCall, *, cancel_event: Event | None = None) -> ToolResult:
            self.executed.append(call.id)
            return ToolResult(call.id, call.name, True, "pyproject.toml\nsrc/fakuicode/cli.py", "found 2 file(s)")

    provider = Provider()
    tools = Tools()
    events = list(AgentRunner(provider, tools).run([AgentMessage("user", "帮我看看项目入口文件有什么")]))

    assert tools.executed == ["call-1"]
    assert len(provider.requests) == 2
    assert [event.kind for event in events] == ["progress", "tool_call", "progress", "tool_result", "progress", "text_delta", "completed"]
    assert events[-2].text.startswith("发现这是一个 Python 项目")


def test_agent_runner_adds_a_visible_fallback_when_tool_continuation_has_no_text() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []

        def stream_agent(
            self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None
        ) -> Iterator[object]:
            self.requests.append(messages)
            if len(self.requests) == 1:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-1", "read_file", {"path": "README.md"}))
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: ToolCall, *, cancel_event: Event | None = None) -> ToolResult:
            return ToolResult(call.id, call.name, True, "contents", "read README.md")

    provider = Provider()
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "inspect README")]))

    assert len(provider.requests) == 2
    assert [event.kind for event in events] == [
        "progress", "tool_call", "progress", "tool_result", "progress", "text_delta", "completed"
    ]
    assert events[-2].text == (
        "Tool execution completed, but the model did not provide a final response. "
        "Please use the results above or ask a more specific follow-up."
    )


def test_agent_session_remembers_the_final_response_for_the_next_turn() -> None:
    from fakuicode.models import AgentStreamEvent, ToolDefinition
    from fakuicode.session import AgentSessionController

    class TextProvider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []

        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            self.requests.append(messages)
            yield AgentStreamEvent("text_delta", "answer")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    provider = TextProvider()
    session = AgentSessionController(provider, NoTools())
    list(session.send("first"))
    list(session.send("second"))

    assert [(message.role, message.content) for message in provider.requests[1]] == [
        ("user", "first"),
        ("assistant", "answer"),
        ("user", "second"),
    ]


def test_agent_session_persists_and_restores_user_and_assistant_turns(tmp_path) -> None:
    from fakuicode.models import AgentStreamEvent, ToolDefinition
    from fakuicode.session import AgentSessionController
    from fakuicode.storage import ConversationStore

    class TextProvider:
        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            yield AgentStreamEvent("text_delta", "saved answer")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    store = ConversationStore(tmp_path / "history.sqlite3")
    record = store.create_conversation("Saved", tmp_path, "default")
    session = AgentSessionController(TextProvider(), NoTools(), store=store, conversation_id=record.id)
    list(session.send("remember this"))

    restored = AgentSessionController(TextProvider(), NoTools(), store=store, conversation_id=record.id)
    assert [(message.role, message.content) for message in restored.history] == [
        ("user", "remember this"),
        ("assistant", "saved answer"),
    ]


def test_agent_session_persists_and_restores_native_tool_history(tmp_path) -> None:
    from fakuicode.models import AgentStreamEvent, ToolCall, ToolDefinition, ToolResult
    from fakuicode.session import AgentSessionController
    from fakuicode.storage import ConversationStore

    class Provider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []

        def stream_agent(
            self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None
        ) -> Iterator[object]:
            self.requests.append(messages)
            if len(self.requests) == 1:
                yield AgentStreamEvent("text_delta", "I will inspect it. ")
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("call-1", "read_file", {"path": "README.md"}))
                yield AgentStreamEvent("completed")
                return
            yield AgentStreamEvent("text_delta", "It is complete.")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: ToolCall, *, cancel_event: Event | None = None) -> ToolResult:
            return ToolResult(call.id, call.name, True, "contents", "read README.md")

    store = ConversationStore(tmp_path / "history.sqlite3")
    record = store.create_conversation("Tools", tmp_path, "default")
    session = AgentSessionController(Provider(), Tools(), store=store, conversation_id=record.id)
    list(session.send("inspect README"))

    events = store.load_events(record.id)
    assert [event.kind for event in events] == ["user", "assistant", "tool_call", "tool_result", "assistant"]
    assert events[1].metadata == {"tool_calls": [{"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}}]}
    assert events[3].metadata == {"tool_name": "read_file", "success": True, "summary": "read README.md"}

    restored = AgentSessionController(Provider(), Tools(), store=store, conversation_id=record.id)
    assert [(message.role, message.content) for message in restored.history] == [
        ("user", "inspect README"),
        ("assistant", "I will inspect it. "),
        ("user", ""),
        ("assistant", "It is complete."),
    ]
    assert restored.history[1].tool_calls[0].name == "read_file"
    assert restored.history[2].tool_results[0].to_model_content() == '{"success":true,"summary":"read README.md","output":"contents"}'


def test_agent_runner_stops_before_the_next_stream_event_when_cancelled() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolDefinition

    class Provider:
        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            yield AgentStreamEvent("text_delta", "first")
            yield AgentStreamEvent("text_delta", "second")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    cancelled = Event()
    events = AgentRunner(Provider(), NoTools()).run([AgentMessage("user", "stop")], cancel_event=cancelled)
    assert next(events).kind == "progress"
    assert next(events).text == "first"
    cancelled.set()
    assert next(events).kind == "cancelled"


def test_agent_runner_retries_a_retryable_provider_failure_before_output() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolDefinition

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("temporary failure", retryable=True)
            yield AgentStreamEvent("text_delta", "answer")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    provider = Provider()
    events = list(AgentRunner(provider, NoTools()).run([AgentMessage("user", "retry")]))
    assert [event.kind for event in events] == ["progress", "text_delta", "completed"]
    assert provider.calls == 2


def test_agent_runner_retries_after_usage_only_but_not_after_visible_output() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent, TokenUsage, ToolDefinition

    class UsageThenFailureProvider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            del messages, tools, cancel_event
            self.calls += 1
            if self.calls == 1:
                yield AgentStreamEvent("usage", usage=TokenUsage(12, 0))
                raise ProviderError("temporary failure", retryable=True)
            yield AgentStreamEvent("text_delta", "answer")
            yield AgentStreamEvent("completed")

    class TextThenFailureProvider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            del messages, tools, cancel_event
            self.calls += 1
            yield AgentStreamEvent("text_delta", "partial")
            raise ProviderError("temporary failure", retryable=True)

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call, *, cancel_event=None):
            raise AssertionError("no tool should run")

    usage_provider = UsageThenFailureProvider()
    usage_events = list(AgentRunner(usage_provider, NoTools()).run([AgentMessage("user", "retry")]))
    assert [event.kind for event in usage_events] == ["progress", "usage", "text_delta", "completed"]
    assert usage_provider.calls == 2

    text_provider = TextThenFailureProvider()
    text_events = list(AgentRunner(text_provider, NoTools()).run([AgentMessage("user", "do not retry")]))
    assert [event.kind for event in text_events] == ["progress", "text_delta", "error"]
    assert text_provider.calls == 1


def test_agent_runner_does_not_retry_after_a_tool_call() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            del messages, tools, cancel_event
            self.calls += 1
            yield AgentStreamEvent(
                "tool_call",
                tool_call=ToolCall("call-1", "read_file", {"path": "README.md"}),
            )
            raise ProviderError("temporary failure", retryable=True)

    class Tools:
        def definitions(self) -> list[ToolDefinition]:
            return [ToolDefinition("read_file", "Read.", {"type": "object"})]

        def execute(self, call, *, cancel_event=None):
            raise AssertionError("an incomplete model round must not execute its tool")

    provider = Provider()
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "inspect")]))

    assert [event.kind for event in events] == ["progress", "tool_call", "error"]
    assert provider.calls == 1


def test_agent_runner_returns_incomplete_tool_arguments_to_the_model_without_execution() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            del tools, cancel_event
            self.calls += 1
            if self.calls == 1:
                yield AgentStreamEvent(
                    "tool_call",
                    tool_call=ToolCall(
                        "call-1",
                        "write_file",
                        {},
                        argument_error="invalid_json",
                    ),
                )
                yield AgentStreamEvent("completed")
                return
            result = messages[-1].tool_results[0]
            assert result.success is False
            assert "smaller tool call" in result.output
            yield AgentStreamEvent("text_delta", "recovered")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self) -> list[ToolDefinition]:
            return [ToolDefinition("write_file", "Write.", {"type": "object"})]

        def is_known(self, name: str) -> bool:
            return name == "write_file"

        def is_read_only(self, name: str) -> bool:
            del name
            return False

        def execute(self, call, *, cancel_event=None):
            raise AssertionError("incomplete arguments must never reach the tool registry")

    provider = Provider()
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "build")]))

    results = [event.tool_result for event in events if event.tool_result is not None]
    assert len(results) == 1
    assert results[0].success is False
    assert any(event.kind == "text_delta" and event.text == "recovered" for event in events)
    assert events[-1].kind == "completed"
    assert provider.calls == 2


def test_agent_runner_exposes_only_structured_provider_diagnostics() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, ToolDefinition

    secret = "provider-detail-must-not-leak"

    class Provider:
        def stream_agent(self, messages, tools, *, cancel_event=None):
            del messages, tools, cancel_event
            raise ProviderError(
                secret,
                status_code=400,
                error_type="invalid_request_error",
                failure_phase="http_status",
                request_id="req_safe-123",
            )
            yield

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call, *, cancel_event=None):
            raise AssertionError("no tool should run")

    events = list(AgentRunner(Provider(), NoTools()).run([AgentMessage("user", "inspect")]))
    error_text = events[-1].text
    assert events[-1].kind == "error"
    assert "HTTP 400" in error_text
    assert "invalid_request_error" in error_text
    assert "http_status" in error_text
    assert "req_safe-123" in error_text
    assert secret not in error_text


def test_agent_runner_prepares_every_model_round_but_not_each_transient_retry() -> None:
    from dataclasses import replace

    from fakuicode.agent import AgentRunner
    from fakuicode.context_manager import ContextPreparationResult
    from fakuicode.models import AgentMessage, AgentStreamEvent, ContextStatus, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            del messages, tools, cancel_event
            self.calls += 1
            self.requests.append(request)
            if self.calls == 1:
                from fakuicode.errors import ProviderError

                raise ProviderError("temporary", retryable=True)
            if self.calls == 2:
                yield AgentStreamEvent(
                    "tool_call",
                    tool_call=ToolCall("call-1", "read_file", {"path": "README.md"}),
                )
                yield AgentStreamEvent("completed")
                return
            yield AgentStreamEvent("text_delta", "done")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            del read_only_only
            return [ToolDefinition("read_file", "Read.", {"type": "object"})]

        def is_known(self, name):
            return name == "read_file"

        def is_read_only(self, name):
            return name == "read_file"

        def execute(self, call, *, cancel_event=None, read_only_only=False):
            del cancel_event, read_only_only
            return ToolResult(call.id, call.name, True, "contents", "read README")

    class Manager:
        def __init__(self) -> None:
            self.calls = []

        def prepare_request(self, request):
            self.calls.append(request)
            status = (
                ContextStatus(
                    "automatic",
                    "compacted",
                    estimated_before=100,
                    estimated_after=50,
                )
                if len(self.calls) == 1
                else None
            )
            return ContextPreparationResult(
                replace(request, system_supplement=f"prepared-{len(self.calls)}"),
                status=status,
            )

    provider = Provider()
    manager = Manager()
    events = list(
        AgentRunner(provider, Tools(), context_manager=manager).run(
            [AgentMessage("user", "inspect")]
        )
    )

    assert len(manager.calls) == 2
    assert provider.calls == 3
    assert provider.requests[0] is provider.requests[1]
    assert provider.requests[0].system_supplement == "prepared-1"
    assert provider.requests[2].system_supplement == "prepared-2"
    assert manager.calls[1].messages[-1].tool_results[0].output == "contents"
    assert [event.kind for event in events].count("context_status") == 1


def test_agent_runner_recovers_one_pre_output_context_overflow_and_bypasses_breaker(
    tmp_path,
) -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.context import ContextPolicy, SUMMARY_HEADINGS
    from fakuicode.context_manager import ContextManager
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent
    from fakuicode.storage import ConversationStore

    summary = "\n\n".join(f"## {heading}\nrecovered" for heading in SUMMARY_HEADINGS)

    class Provider:
        def __init__(self) -> None:
            self.normal_calls = 0
            self.summary_calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            del messages, tools, cancel_event
            if request.output_token_limit is not None:
                self.summary_calls += 1
                yield AgentStreamEvent("text_delta", summary)
                yield AgentStreamEvent("completed")
                return
            self.normal_calls += 1
            if self.normal_calls == 1:
                raise ProviderError("too large", category="context_overflow")
            yield AgentStreamEvent("text_delta", "done")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self, *, read_only_only=False):
            del read_only_only
            return []

    store = ConversationStore(tmp_path / "history.sqlite3")
    conversation = store.create_conversation("overflow", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"user-{index} " + "u" * 200)
        store.append_event(conversation.id, "assistant", f"answer-{index} " + "a" * 200)
    provider = Provider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=ContextPolicy(
            recent_history_target_tokens=40,
            recent_history_min_groups=2,
            older_user_messages_target_tokens=200,
        ),
    )
    for _ in range(3):
        manager.record_summary_failure()

    events = list(
        AgentRunner(provider, NoTools(), context_manager=manager).run(
            [AgentMessage("user", "continue")]
        )
    )

    assert provider.normal_calls == 2
    assert provider.summary_calls == 1
    assert manager.automatic_compaction_disabled is False
    assert [event.kind for event in events] == [
        "progress",
        "context_status",
        "text_delta",
        "completed",
    ]
    assert events[1].context_status is not None
    assert events[1].context_status.trigger == "emergency"


def test_plan_mode_compacts_with_no_summary_tools_and_keeps_mcp_filtered(tmp_path) -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.context import ContextPolicy, SUMMARY_HEADINGS
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentStreamEvent, ToolDefinition
    from fakuicode.storage import ConversationStore

    summary = "\n\n".join(f"## {heading}\nretained" for heading in SUMMARY_HEADINGS)

    class Provider:
        def __init__(self) -> None:
            self.summary_tool_names = []
            self.plan_tool_names = []
            self.plan_supplement = ""

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            del messages, cancel_event
            if request.output_token_limit is not None:
                self.summary_tool_names.append([tool.name for tool in tools])
                yield AgentStreamEvent("text_delta", summary)
                yield AgentStreamEvent("completed")
                return
            self.plan_tool_names.append([tool.name for tool in tools])
            self.plan_supplement = request.system_supplement
            yield AgentStreamEvent("text_delta", "read-only plan")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            read = ToolDefinition("read_file", "read", {"type": "object"})
            mcp = ToolDefinition("mcp__docs__lookup", "remote", {"type": "object"})
            return [read] if read_only_only else [read, mcp]

        def is_known(self, name):
            return name in {"read_file", "mcp__docs__lookup"}

        def is_read_only(self, name):
            return name == "read_file"

        def execute(self, call, *, cancel_event=None, read_only_only=False):
            raise AssertionError("the provider should not request tools")

    store = ConversationStore(tmp_path / "history.sqlite3")
    conversation = store.create_conversation("plan", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"user-{index} " + "u" * 200)
        store.append_event(conversation.id, "assistant", f"answer-{index} " + "a" * 200)
    provider = Provider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=ContextPolicy(
            automatic_reserve_tokens=127_900,
            recent_history_target_tokens=40,
            recent_history_min_groups=2,
            older_user_messages_target_tokens=200,
        ),
    )

    events = list(
        AgentRunner(provider, Tools(), context_manager=manager).run(
            manager.active_messages(),
            mode="plan",
        )
    )

    assert provider.summary_tool_names == [[]]
    assert provider.plan_tool_names == [["read_file"]]
    assert "<context-summary>" in provider.plan_supplement
    assert "mcp__docs__lookup" not in provider.plan_tool_names[0]
    assert events[-1].kind == "completed"


def test_agent_runner_does_not_loop_after_the_overflow_retry_also_overflows(tmp_path) -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.context import ContextPolicy, SUMMARY_HEADINGS
    from fakuicode.context_manager import ContextManager
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent
    from fakuicode.storage import ConversationStore

    summary = "\n\n".join(f"## {heading}\nretry" for heading in SUMMARY_HEADINGS)

    class Provider:
        def __init__(self) -> None:
            self.normal_calls = 0
            self.summary_calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            del messages, tools, cancel_event
            if request.output_token_limit is not None:
                self.summary_calls += 1
                yield AgentStreamEvent("text_delta", summary)
                yield AgentStreamEvent("completed")
                return
            self.normal_calls += 1
            raise ProviderError("still too large", category="context_overflow")
            yield

    class NoTools:
        def definitions(self, *, read_only_only=False):
            del read_only_only
            return []

    store = ConversationStore(tmp_path / "history.sqlite3")
    conversation = store.create_conversation("overflow", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"user-{index} " + "u" * 200)
        store.append_event(conversation.id, "assistant", f"answer-{index} " + "a" * 200)
    provider = Provider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=ContextPolicy(
            recent_history_target_tokens=40,
            recent_history_min_groups=2,
            older_user_messages_target_tokens=200,
        ),
    )

    events = list(
        AgentRunner(provider, NoTools(), context_manager=manager).run(
            [AgentMessage("user", "continue")]
        )
    )

    assert provider.normal_calls == 2
    assert provider.summary_calls == 1
    assert events[-1].kind == "error"


def test_agent_runner_never_retries_overflow_after_visible_output(tmp_path) -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.context_manager import ContextManager
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            del messages, tools, cancel_event, request
            self.calls += 1
            yield AgentStreamEvent("text_delta", "partial")
            raise ProviderError("too large", category="context_overflow")

    class NoTools:
        def definitions(self, *, read_only_only=False):
            del read_only_only
            return []

    provider = Provider()
    manager = ContextManager(provider, workspace=tmp_path, context_window=128_000)
    events = list(
        AgentRunner(provider, NoTools(), context_manager=manager).run(
            [AgentMessage("user", "continue")]
        )
    )

    assert provider.calls == 1
    assert [event.kind for event in events] == ["progress", "text_delta", "error"]


def test_agent_runner_stops_when_forced_overflow_summary_fails(tmp_path) -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.context import ContextPolicy
    from fakuicode.context_manager import ContextManager
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent
    from fakuicode.storage import ConversationStore

    class Provider:
        def __init__(self) -> None:
            self.normal_calls = 0
            self.summary_calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            del messages, tools, cancel_event
            if request.output_token_limit is not None:
                self.summary_calls += 1
                yield AgentStreamEvent("text_delta", "invalid summary")
                yield AgentStreamEvent("completed")
                return
            self.normal_calls += 1
            raise ProviderError("too large", category="context_overflow")
            yield

    class NoTools:
        def definitions(self, *, read_only_only=False):
            del read_only_only
            return []

    store = ConversationStore(tmp_path / "history.sqlite3")
    conversation = store.create_conversation("overflow", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"user-{index} " + "u" * 200)
        store.append_event(conversation.id, "assistant", f"answer-{index} " + "a" * 200)
    provider = Provider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=ContextPolicy(
            recent_history_target_tokens=40,
            recent_history_min_groups=2,
            older_user_messages_target_tokens=200,
        ),
    )

    events = list(
        AgentRunner(provider, NoTools(), context_manager=manager).run(
            [AgentMessage("user", "continue")]
        )
    )

    assert provider.normal_calls == 1
    assert provider.summary_calls == 1
    assert manager.consecutive_summary_failures == 1
    assert [event.kind for event in events] == ["progress", "context_status", "error"]
    assert events[1].context_status is not None
    assert events[1].context_status.result == "failed"


def test_agent_runner_closes_a_provider_that_supports_active_stream_cancellation() -> None:
    from fakuicode.agent import AgentRunner

    class Provider:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            return iter(())

    class NoTools:
        def definitions(self) -> list[object]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    provider = Provider()
    AgentRunner(provider, NoTools()).cancel()
    assert provider.cancelled is True


def test_agent_runner_turns_an_unclassified_provider_exception_into_a_safe_error() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, ToolDefinition

    class Provider:
        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            raise RuntimeError("backend implementation detail")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    events = list(AgentRunner(Provider(), NoTools()).run([AgentMessage("user", "hello")]))
    assert [event.kind for event in events] == ["progress", "error"]
    assert "implementation detail" not in events[-1].text


def test_agent_session_clear_context_keeps_history_but_excludes_it_from_the_next_request(tmp_path) -> None:
    from fakuicode.models import AgentStreamEvent, ToolDefinition
    from fakuicode.session import AgentSessionController
    from fakuicode.storage import ConversationStore

    class Provider:
        def __init__(self) -> None:
            self.requests: list[Sequence[object]] = []

        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            self.requests.append(messages)
            yield AgentStreamEvent("text_delta", "answer")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    store = ConversationStore(tmp_path / "history.sqlite3")
    record = store.create_conversation("Clear", tmp_path, "default")
    provider = Provider()
    session = AgentSessionController(provider, NoTools(), store=store, conversation_id=record.id)
    list(session.send("old request"))
    session.clear_context()
    list(session.send("new request"))

    assert [(message.role, message.content) for message in provider.requests[-1]] == [("user", "new request")]
    assert [event.content for event in store.load_events(record.id) if event.kind == "user"] == ["old request", "new request"]


def test_agent_session_can_persist_a_turn_from_the_stream_worker_thread(tmp_path) -> None:
    from fakuicode.models import AgentStreamEvent, ToolDefinition
    from fakuicode.session import AgentSessionController
    from fakuicode.storage import ConversationStore

    class Provider:
        def stream_agent(self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None) -> Iterator[object]:
            yield AgentStreamEvent("text_delta", "answer")
            yield AgentStreamEvent("completed")

    class NoTools:
        def definitions(self) -> list[ToolDefinition]:
            return []

        def execute(self, call: object, *, cancel_event: Event | None = None) -> object:
            raise AssertionError("no tool should run")

    store = ConversationStore(tmp_path / "history.sqlite3")
    record = store.create_conversation("Worker", tmp_path, "default")
    session = AgentSessionController(Provider(), NoTools(), store=store, conversation_id=record.id)
    errors: list[Exception] = []

    def send_from_worker() -> None:
        try:
            list(session.send("hello"))
        except Exception as error:
            errors.append(error)

    worker = Thread(target=send_from_worker)
    worker.start()
    worker.join()

    assert errors == []
    assert [(event.kind, event.content) for event in store.load_events(record.id)] == [
        ("user", "hello"),
        ("assistant", "answer"),
    ]


def test_agent_runner_reacts_across_multiple_tool_rounds_until_the_model_finishes() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.requests: list[tuple[Sequence[object], Sequence[object]]] = []

        def stream_agent(
            self, messages: Sequence[object], tools: Sequence[object], *, cancel_event: Event | None = None
        ) -> Iterator[object]:
            self.requests.append((messages, tools))
            turn = len(self.requests)
            if turn == 1:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("read", "read_file", {"path": "README.md"}))
            elif turn == 2:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("search", "search_code", {"query": "main"}))
            else:
                yield AgentStreamEvent("text_delta", "入口已经确认。")
            yield AgentStreamEvent("completed")

    class Tools:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def definitions(self, *, read_only_only: bool = False) -> list[ToolDefinition]:
            return [
                ToolDefinition("read_file", "Read", {"type": "object"}),
                ToolDefinition("search_code", "Search", {"type": "object"}),
            ]

        def is_known(self, name: str) -> bool:
            return name in {"read_file", "search_code"}

        def is_read_only(self, name: str) -> bool:
            return self.is_known(name)

        def execute(self, call: ToolCall, *, cancel_event: Event | None = None) -> ToolResult:
            self.executed.append(call.id)
            return ToolResult(call.id, call.name, True, call.id, f"ran {call.name}")

    provider = Provider()
    tools = Tools()
    events = list(AgentRunner(provider, tools).run([AgentMessage("user", "找到入口")]))

    assert tools.executed == ["read", "search"]
    assert len(provider.requests) == 3
    assert [event.kind for event in events] == [
        "progress",
        "tool_call",
        "progress",
        "tool_result",
        "progress",
        "tool_call",
        "progress",
        "tool_result",
        "progress",
        "text_delta",
        "completed",
    ]


def test_agent_session_saves_a_read_only_plan_then_prepares_explicit_execution() -> None:
    from fakuicode.models import AgentStreamEvent, ToolDefinition
    from fakuicode.session import AgentSessionController

    class Provider:
        def __init__(self) -> None:
            self.instructions: list[str] = []

        def stream_agent(
            self,
            messages: Sequence[object],
            tools: Sequence[object],
            *,
            cancel_event: Event | None = None,
            system_instruction: str = "",
        ) -> Iterator[object]:
            self.instructions.append(system_instruction)
            yield AgentStreamEvent("text_delta", "1. 读取配置\n2. 修改入口")
            yield AgentStreamEvent("completed")

    class Tools:
        def __init__(self) -> None:
            self.read_only_requests: list[bool] = []

        def definitions(self, *, read_only_only: bool = False) -> list[ToolDefinition]:
            self.read_only_requests.append(read_only_only)
            return [ToolDefinition("read_file", "Read", {"type": "object"})]

        def is_known(self, name: str) -> bool:
            return name == "read_file"

        def is_read_only(self, name: str) -> bool:
            return name == "read_file"

        def execute(self, call, *, cancel_event: Event | None = None):
            raise AssertionError("the model should not need a tool")

    provider = Provider()
    tools = Tools()
    session = AgentSessionController(provider, tools)

    session.enable_plan_mode()
    list(session.send("检查入口"))

    assert tools.read_only_requests == [True]
    assert "计划模式" in provider.instructions[0]
    assert session.mode == "plan"
    assert session.saved_plan == "1. 读取配置\n2. 修改入口"
    assert session.prepare_plan_execution() == "1. 读取配置\n2. 修改入口"
    assert session.mode == "execute"


def test_agent_runner_stops_at_the_iteration_limit_after_recording_the_last_results() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            self.calls += 1
            yield AgentStreamEvent("tool_call", tool_call=ToolCall("one", "read_file", {"path": "x"}))
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return [ToolDefinition("read_file", "Read", {"type": "object"})]

        def is_known(self, name):
            return True

        def is_read_only(self, name):
            return True

        def execute(self, call, *, cancel_event=None):
            return ToolResult(call.id, call.name, True, "ok", "read x")

    provider = Provider()
    events = list(AgentRunner(provider, Tools(), max_iterations=1).run([AgentMessage("user", "loop")]))

    assert provider.calls == 1
    assert [event.kind for event in events] == ["progress", "tool_call", "progress", "tool_result", "error"]
    assert "1-round safety limit" in events[-1].text


def test_agent_runner_finishes_turn_after_deferred_system_tool_result() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import (
        AgentMessage,
        AgentStreamEvent,
        ToolCall,
        ToolDefinition,
        ToolResult,
    )

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None, system_instruction=""):
            del messages, tools, cancel_event, system_instruction
            self.calls += 1
            yield AgentStreamEvent(
                "tool_call",
                tool_call=ToolCall(
                    f"call-{self.calls}",
                    "agent",
                    {"prompt": "plan", "description": "make a plan"},
                ),
            )
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            del read_only_only
            return [ToolDefinition("agent", "delegate", {"type": "object"})]

        def is_known(self, name):
            return name == "agent"

        def is_read_only(self, name):
            del name
            return False

        def execute(self, call, *, cancel_event=None, read_only_only=False):
            del cancel_event, read_only_only
            return ToolResult(
                call.id,
                call.name,
                True,
                '{"status":"async_launched"}',
                "子 Agent 已在后台启动",
                metadata={
                    "finish_agent_turn": True,
                    "finish_agent_turn_message": (
                        "子 Agent planner 已在后台启动（task-1），完成后会自动汇报结果。"
                    ),
                },
            )

        def finish_turn_message(self, results):
            messages = [
                str(result.metadata["finish_agent_turn_message"])
                for result in results
                if result.metadata is not None
                and result.metadata.get("finish_agent_turn") is True
            ]
            return "\n".join(messages) if len(messages) == len(results) else None

    provider = Provider()
    events = list(
        AgentRunner(provider, Tools(), max_iterations=3).run(
            [AgentMessage("user", "launch a planner")]
        )
    )

    assert provider.calls == 1
    assert [event.kind for event in events][-2:] == ["text_delta", "completed"]
    assert "完成后会自动汇报结果" in events[-2].text


def test_agent_runner_allows_more_than_twelve_rounds_by_default() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            self.calls += 1
            if self.calls <= 13:
                yield AgentStreamEvent(
                    "tool_call",
                    tool_call=ToolCall(f"call-{self.calls}", "write_file", {"path": f"file-{self.calls}.txt"}),
                )
            else:
                yield AgentStreamEvent("text_delta", "done")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return [ToolDefinition("write_file", "Write", {})]

        def is_known(self, name):
            return True

        def is_read_only(self, name):
            return False

        def execute(self, call, *, cancel_event=None):
            return ToolResult(call.id, call.name, True, "", "wrote file")

    provider = Provider()
    events = list(AgentRunner(provider, Tools()).run([AgentMessage("user", "build")]))

    assert provider.calls == 14
    assert events[-1].kind == "completed"


def test_agent_runner_stops_after_two_consecutive_unknown_tool_calls() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolResult

    class Provider:
        def stream_agent(self, messages, tools, *, cancel_event=None):
            yield AgentStreamEvent("tool_call", tool_call=ToolCall("one", "missing", {}))
            yield AgentStreamEvent("tool_call", tool_call=ToolCall("two", "also_missing", {}))
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return []

        def is_known(self, name):
            return False

        def is_read_only(self, name):
            return False

        def execute(self, call, *, cancel_event=None):
            return ToolResult(call.id, call.name, False, "unknown", "unknown tool")

    events = list(AgentRunner(Provider(), Tools()).run([AgentMessage("user", "run")]))

    assert [event.kind for event in events] == ["progress", "tool_call", "tool_call", "progress", "tool_result", "tool_result", "error"]
    assert events[-1].text == "Agent stopped after two consecutive unknown tool calls."


def test_agent_runner_cancellation_synthesizes_results_for_all_announced_tools() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.errors import RequestCancelled
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition

    cancelled = Event()

    class Provider:
        def stream_agent(self, messages, tools, *, cancel_event=None):
            yield AgentStreamEvent("tool_call", tool_call=ToolCall("one", "write_file", {"path": "one"}))
            yield AgentStreamEvent("tool_call", tool_call=ToolCall("two", "write_file", {"path": "two"}))
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return [ToolDefinition("write_file", "Write", {"type": "object"})]

        def is_known(self, name):
            return True

        def is_read_only(self, name):
            return False

        def execute(self, call, *, cancel_event=None):
            cancelled.set()
            raise RequestCancelled()

    events = list(AgentRunner(Provider(), Tools()).run([AgentMessage("user", "stop")], cancel_event=cancelled))
    results = [event.tool_result for event in events if event.kind == "tool_result"]

    assert [result.call_id for result in results] == ["one", "two"]
    assert all(result.success is False and "cancelled" in result.output for result in results)
    assert events[-1].kind == "cancelled"


def test_agent_runner_runs_a_contiguous_read_only_batch_concurrently_but_emits_model_order() -> None:
    from threading import Barrier

    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    barrier = Barrier(2)

    class Provider:
        calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            self.calls += 1
            if self.calls == 1:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("first", "read_file", {"path": "a"}))
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("second", "search_code", {"query": "b"}))
            else:
                yield AgentStreamEvent("text_delta", "done")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return [ToolDefinition("read_file", "Read", {}), ToolDefinition("search_code", "Search", {})]

        def is_known(self, name):
            return True

        def is_read_only(self, name):
            return True

        def execute(self, call, *, cancel_event=None):
            barrier.wait(timeout=1)
            return ToolResult(call.id, call.name, True, call.id, f"ran {call.name}")

    events = list(AgentRunner(Provider(), Tools()).run([AgentMessage("user", "inspect")]))
    result_ids = [event.tool_result.call_id for event in events if event.kind == "tool_result"]

    assert result_ids == ["first", "second"]


def test_agent_runner_uses_the_shared_read_only_scheduler_budget() -> None:
    from threading import Event, Lock, Thread

    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult
    from fakuicode.tool_scheduler import ReadOnlyToolScheduler

    first_started = Event()
    release_first = Event()
    second_started = Event()
    active = 0
    maximum_active = 0
    active_lock = Lock()

    class Provider:
        calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            self.calls += 1
            if self.calls == 1:
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("first", "read_file", {"path": "a"}))
                yield AgentStreamEvent("tool_call", tool_call=ToolCall("second", "search_code", {"query": "b"}))
            else:
                yield AgentStreamEvent("text_delta", "done")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return [ToolDefinition("read_file", "Read", {}), ToolDefinition("search_code", "Search", {})]

        def is_known(self, name):
            return True

        def is_read_only(self, name):
            return True

        def execute(self, call, *, cancel_event=None):
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            if call.id == "first":
                first_started.set()
                assert release_first.wait(timeout=1)
            else:
                second_started.set()
            with active_lock:
                active -= 1
            return ToolResult(call.id, call.name, True, call.id, f"ran {call.name}")

    scheduler = ReadOnlyToolScheduler(max_workers=1)
    events: list[object] = []
    worker = Thread(
        target=lambda: events.extend(
            AgentRunner(Provider(), Tools(), read_only_scheduler=scheduler).run(
                [AgentMessage("user", "inspect")]
            )
        )
    )
    worker.start()
    try:
        assert first_started.wait(timeout=1)
        assert not second_started.wait(timeout=0.05)
        release_first.set()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert maximum_active == 1
        result_ids = [
            event.tool_result.call_id
            for event in events
            if getattr(event, "kind", None) == "tool_result"
        ]
        assert result_ids == ["first", "second"]
    finally:
        release_first.set()
        scheduler.close()
        worker.join(timeout=1)


def test_agent_session_accumulates_exact_usage_and_marks_missing_usage_unavailable() -> None:
    from fakuicode.models import AgentStreamEvent, TokenUsage
    from fakuicode.session import AgentSessionController

    class Provider:
        def __init__(self) -> None:
            self.turn = 0

        def stream_agent(self, messages, tools, *, cancel_event=None):
            self.turn += 1
            if self.turn == 1:
                yield AgentStreamEvent("usage", usage=TokenUsage(3, 5))
            yield AgentStreamEvent("text_delta", "answer")
            yield AgentStreamEvent("completed")

    class Tools:
        def definitions(self, *, read_only_only=False):
            return []

        def is_known(self, name):
            return False

        def is_read_only(self, name):
            return False

        def execute(self, call, *, cancel_event=None):
            raise AssertionError("no tool expected")

    session = AgentSessionController(Provider(), Tools())
    list(session.send("first"))
    assert session.token_usage == TokenUsage(3, 5)

    list(session.send("second"))
    assert session.token_usage is None


def test_permission_denial_is_returned_to_the_model_and_agent_can_change_strategy(tmp_path) -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall
    from fakuicode.permissions.config import PermissionConfigSnapshot
    from fakuicode.permissions.manager import PermissionManager
    from fakuicode.permissions.models import ApprovalChoice
    from fakuicode.permissions.safety import DangerousCommandGuard
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    (tmp_path / "README.md").write_text("safe", encoding="utf-8")

    class DenyApproval:
        def request(self, request, *, cancel_event=None):
            del request, cancel_event
            return ApprovalChoice.DENY

    class Provider:
        def __init__(self) -> None:
            self.requests = []

        def stream_agent(self, messages, tools, *, cancel_event=None):
            del tools, cancel_event
            self.requests.append(messages)
            if len(self.requests) == 1:
                yield AgentStreamEvent(
                    "tool_call",
                    tool_call=ToolCall("write-1", "write_file", {"path": "blocked.txt", "content": "no"}),
                )
            elif len(self.requests) == 2:
                assert self.requests[-1][-1].tool_results[0].success is False
                yield AgentStreamEvent(
                    "tool_call", tool_call=ToolCall("read-1", "read_file", {"path": "README.md"})
                )
            else:
                assert self.requests[-1][-1].tool_results[0].success is True
                yield AgentStreamEvent("text_delta", "Used a safe read instead.")
            yield AgentStreamEvent("completed")

    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
        approval_handler=DenyApproval(),
    )
    registry = ToolRegistry(WorkspacePolicy(tmp_path), permission_manager=permissions)

    events = list(AgentRunner(Provider(), registry).run([AgentMessage("user", "inspect safely")]))

    results = [event.tool_result for event in events if event.kind == "tool_result"]
    assert [result.success for result in results if result is not None] == [False, True]
    assert not (tmp_path / "blocked.txt").exists()
    assert events[-2].text == "Used a safe read instead."


def test_agent_runner_starts_one_permission_request_scope_and_passes_plan_boundary() -> None:
    from fakuicode.agent import AgentRunner
    from fakuicode.models import AgentMessage, AgentStreamEvent, ToolCall, ToolDefinition, ToolResult

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, cancel_event=None, system_instruction=""):
            del messages, tools, cancel_event, system_instruction
            self.calls += 1
            if self.calls == 1:
                yield AgentStreamEvent(
                    "tool_call", tool_call=ToolCall("call-plan", "write_file", {"path": "no.txt"})
                )
            else:
                yield AgentStreamEvent("text_delta", "planned")
            yield AgentStreamEvent("completed")

    class Tools:
        def __init__(self) -> None:
            self.request_starts = 0
            self.read_only_flags = []

        def begin_request(self) -> None:
            self.request_starts += 1

        def definitions(self, *, read_only_only=False):
            del read_only_only
            return [ToolDefinition("write_file", "write", {"type": "object"})]

        def is_known(self, name):
            return name == "write_file"

        def is_read_only(self, name):
            del name
            return False

        def execute(self, call, *, cancel_event=None, read_only_only=False):
            del cancel_event
            self.read_only_flags.append(read_only_only)
            return ToolResult(call.id, call.name, False, "plan denied", "permission denied")

    tools = Tools()

    events = list(AgentRunner(Provider(), tools).run([AgentMessage("user", "plan")], mode="plan"))

    assert events[-1].kind == "completed"
    assert tools.request_starts == 1
    assert tools.read_only_flags == [True]


def test_agent_session_close_releases_provider_and_permission_tools() -> None:
    from fakuicode.session import AgentSessionController

    class Provider:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    class Tools:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    provider = Provider()
    tools = Tools()
    session = AgentSessionController(provider, tools)
    scheduler = session.read_only_scheduler

    session.close()

    assert provider.cancelled is True
    assert tools.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.submit(lambda: None)


def test_agent_session_does_not_close_a_host_owned_read_only_scheduler() -> None:
    from fakuicode.session import AgentSessionController
    from fakuicode.tool_scheduler import ReadOnlyToolScheduler

    class Provider:
        def cancel(self) -> None:
            pass

    class Tools:
        def close(self) -> None:
            pass

    scheduler = ReadOnlyToolScheduler(max_workers=1)
    session = AgentSessionController(
        Provider(),
        Tools(),
        read_only_scheduler=scheduler,
    )

    try:
        session.close()
        assert scheduler.submit(lambda: "still available").result(timeout=1) == "still available"
    finally:
        scheduler.close()
