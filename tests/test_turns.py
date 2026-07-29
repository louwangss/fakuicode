from __future__ import annotations

import pytest


def test_turn_recorder_persists_tool_cycle_and_builds_provider_history() -> None:
    from fakuicode.models import (
        AgentMessage,
        AgentProgress,
        AgentStreamEvent,
        ProviderMessageState,
        TimelineEvent,
        TokenUsage,
        ToolCall,
        ToolResult,
    )
    from fakuicode.turns import TurnRecorder

    persisted: list[TimelineEvent] = []
    usages: list[TokenUsage | None] = []

    def append_event(kind, content, *, call_id=None, metadata=None):
        event = TimelineEvent(len(persisted) + 1, kind, content, call_id, metadata)
        persisted.append(event)
        return event

    current = AgentMessage("user", "inspect")
    recorder = TurnRecorder(current, append_event=append_event, usage_sink=usages.append)
    state = ProviderMessageState(
        "anthropic",
        ({"type": "thinking", "thinking": "inspect", "signature": "sig"},),
    )
    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    result = ToolResult("call-1", "read_file", True, "contents", "read README")

    assert recorder.consume(AgentStreamEvent("progress", progress=AgentProgress(1, "model"))) is None
    recorder.consume(AgentStreamEvent("thinking_end", provider_state=state))
    recorder.consume(AgentStreamEvent("text_delta", "checking"))
    recorder.consume(AgentStreamEvent("tool_call", tool_call=call))
    recorder.consume(AgentStreamEvent("tool_result", tool_result=result))
    recorder.consume(AgentStreamEvent("progress", progress=AgentProgress(2, "model")))
    recorder.consume(AgentStreamEvent("usage", usage=TokenUsage(10, 2)))
    recorder.consume(AgentStreamEvent("text_delta", "done"))

    assert recorder.consume(AgentStreamEvent("completed")) == "completed"
    recorder.ensure_terminal()

    assert recorder.history == (
        current,
        AgentMessage("assistant", "checking", (call,), provider_state=state),
        AgentMessage("user", tool_results=(result,)),
        AgentMessage("assistant", "done"),
    )
    assert [event.kind for event in persisted] == [
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
    ]
    assert recorder.assistant_event is persisted[-1]
    assert recorder.answer == "done"
    assert recorder.safe_tool_summaries[0].summary == "read README"
    assert usages == [TokenUsage(10, 2)]


def test_turn_recorder_preserves_partial_tool_history_on_termination() -> None:
    from fakuicode.models import AgentMessage, AgentStreamEvent, TimelineEvent, ToolCall, ToolResult
    from fakuicode.turns import TurnRecorder

    persisted: list[TimelineEvent] = []

    def append_event(kind, content, *, call_id=None, metadata=None):
        event = TimelineEvent(len(persisted) + 1, kind, content, call_id, metadata)
        persisted.append(event)
        return event

    current = AgentMessage("user", "inspect")
    recorder = TurnRecorder(current, append_event=append_event, usage_sink=lambda usage: None)
    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    result = ToolResult("call-1", "read_file", True, "contents", "read README")
    recorder.consume(AgentStreamEvent("tool_call", tool_call=call))
    recorder.consume(AgentStreamEvent("tool_result", tool_result=result))

    assert recorder.consume(AgentStreamEvent("cancelled", "cancelled")) == "terminated"
    recorder.ensure_terminal()

    assert recorder.history == (
        current,
        AgentMessage("assistant", "", (call,)),
        AgentMessage("user", tool_results=(result,)),
    )
    assert persisted[-1].kind == "system"
    assert persisted[-1].content == "cancelled"


def test_turn_recorder_rejects_stream_without_terminal_event() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.models import AgentMessage, AgentStreamEvent
    from fakuicode.turns import TurnRecorder

    recorder = TurnRecorder(
        AgentMessage("user", "hello"),
        append_event=lambda *args, **kwargs: None,
        usage_sink=lambda usage: None,
    )
    recorder.consume(AgentStreamEvent("text_delta", "partial"))

    with pytest.raises(ProviderError, match="ended before completion"):
        recorder.ensure_terminal()
