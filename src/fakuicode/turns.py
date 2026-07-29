"""Per-turn stream recording and durable timeline projection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from fakuicode.errors import ProviderError
from fakuicode.memory.models import SafeToolSummary
from fakuicode.models import (
    AgentMessage,
    AgentStreamEvent,
    ProviderMessageState,
    TimelineEvent,
    TimelineEventKind,
    TokenUsage,
    ToolCall,
    ToolResult,
)


TurnOutcome = Literal["completed", "terminated"]


class TimelineAppender(Protocol):
    def __call__(
        self,
        kind: TimelineEventKind,
        content: str,
        *,
        call_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TimelineEvent | None: ...


class TurnRecorder:
    """Collect one agent stream without owning session-level policy."""

    def __init__(
        self,
        current: AgentMessage,
        *,
        append_event: TimelineAppender,
        usage_sink: Callable[[TokenUsage | None], None],
    ) -> None:
        self._append_event = append_event
        self._usage_sink = usage_sink
        self._history: list[AgentMessage] = [current]
        self._response_text: list[str] = []
        self._calls: list[ToolCall] = []
        self._provider_states: list[ProviderMessageState] = []
        self._pending_results: list[ToolResult] = []
        self._safe_tool_summaries: list[SafeToolSummary] = []
        self._tool_turn_persisted = False
        self._round_usage: TokenUsage | None = None
        self._outcome: TurnOutcome | None = None
        self._answer = ""
        self._assistant_event: TimelineEvent | None = None

    @property
    def history(self) -> tuple[AgentMessage, ...]:
        return tuple(self._history)

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def assistant_event(self) -> TimelineEvent | None:
        return self._assistant_event

    @property
    def safe_tool_summaries(self) -> tuple[SafeToolSummary, ...]:
        return tuple(self._safe_tool_summaries)

    def consume(self, event: AgentStreamEvent) -> TurnOutcome | None:
        if self._outcome is not None:
            raise ProviderError("Agent emitted events after its terminal event.")
        if event.provider_state is not None:
            self._provider_states.append(event.provider_state)
        if event.kind == "progress" and event.progress is not None:
            if event.progress.phase == "model":
                self._flush_results()
                self._response_text.clear()
                self._calls.clear()
                self._provider_states.clear()
                self._tool_turn_persisted = False
            else:
                self._flush_usage()
        elif event.kind == "usage" and event.usage is not None:
            self._round_usage = event.usage
        elif event.kind == "text_delta":
            self._response_text.append(event.text)
        elif event.kind == "tool_call" and event.tool_call is not None:
            self._calls.append(event.tool_call)
        elif event.kind == "tool_result" and event.tool_result is not None:
            self._record_tool_result(event.tool_result)
        elif event.kind == "completed":
            self._complete()
        elif event.kind in {"cancelled", "error"}:
            self._terminate(event.text or event.kind)
        return self._outcome

    def ensure_terminal(self) -> None:
        if self._outcome is None:
            raise ProviderError("Agent stream ended before completion.")

    def _record_tool_result(self, result: ToolResult) -> None:
        if not self._tool_turn_persisted:
            provider_state = _merge_provider_states(self._provider_states)
            assistant = AgentMessage(
                "assistant",
                "".join(self._response_text),
                tuple(self._calls),
                provider_state=provider_state,
            )
            self._history.append(assistant)
            assistant_metadata: dict[str, object] = {
                "tool_calls": _tool_call_metadata(self._calls)
            }
            if provider_state is not None:
                assistant_metadata["provider_state"] = _provider_state_metadata(
                    provider_state
                )
            self._append_event(
                "assistant",
                assistant.content,
                metadata=assistant_metadata,
            )
            for call in self._calls:
                self._append_event(
                    "tool_call",
                    call.name,
                    call_id=call.id,
                    metadata={"arguments": dict(call.arguments)},
                )
            self._tool_turn_persisted = True
        self._pending_results.append(result)
        self._safe_tool_summaries.append(
            SafeToolSummary(result.tool_name, result.success, result.summary)
        )
        self._append_event(
            "tool_result",
            result.output,
            call_id=result.call_id,
            metadata={
                "tool_name": result.tool_name,
                "success": result.success,
                "summary": result.summary,
                **(
                    {"duration_seconds": result.duration_seconds}
                    if result.duration_seconds is not None
                    else {}
                ),
                **(dict(result.metadata) if result.metadata is not None else {}),
            },
        )

    def _complete(self) -> None:
        self._flush_usage()
        self._flush_results()
        answer = "".join(self._response_text)
        if not answer.strip():
            raise ProviderError("Provider completed without text content.")
        self._answer = answer
        self._history.append(AgentMessage("assistant", answer))
        self._assistant_event = self._append_event("assistant", answer)
        self._outcome = "completed"

    def _terminate(self, message: str) -> None:
        self._flush_usage()
        self._flush_results()
        self._append_event("system", message)
        self._outcome = "terminated"

    def _flush_results(self) -> None:
        if self._pending_results:
            self._history.append(
                AgentMessage("user", tool_results=tuple(self._pending_results))
            )
            self._pending_results.clear()

    def _flush_usage(self) -> None:
        self._usage_sink(self._round_usage)
        self._round_usage = None


def _tool_call_metadata(calls: list[ToolCall]) -> list[dict[str, object]]:
    return [
        {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
        for call in calls
    ]


def _merge_provider_states(
    states: list[ProviderMessageState],
) -> ProviderMessageState | None:
    if not states:
        return None
    protocol = states[0].protocol
    if any(state.protocol != protocol for state in states):
        raise ProviderError("Provider emitted incompatible message state.")
    return ProviderMessageState(
        protocol,
        tuple(block for state in states for block in state.thinking_blocks),
    )


def _provider_state_metadata(state: ProviderMessageState) -> dict[str, object]:
    return {
        "protocol": state.protocol,
        "thinking_blocks": [dict(block) for block in state.thinking_blocks],
    }
