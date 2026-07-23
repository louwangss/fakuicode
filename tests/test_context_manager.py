from __future__ import annotations

from pathlib import Path

import pytest


class _NoopProvider:
    def stream_agent(self, messages, tools, **kwargs):
        del messages, tools, kwargs
        raise AssertionError("provider should not be called by lifecycle reconstruction")


def test_context_compaction_emits_balanced_lifecycle_callbacks(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.providers.base import AgentRequest

    events: list[tuple[str, object]] = []
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        lifecycle_callback=lambda event, payload: events.append((event, payload)),
    )

    result = manager.compact_request(AgentRequest((), ()), trigger="manual")

    assert result.status is not None and result.status.result == "noop"
    assert events == [
        ("pre_compact", {"compact": {"trigger": "manual", "outcome": "started"}}),
        ("post_compact", {"compact": {"trigger": "manual", "outcome": "noop"}}),
    ]


def test_context_manager_rebuilds_only_events_after_the_latest_clear_boundary(
    tmp_path: Path,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentMessage
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("restore", tmp_path, "default")
    store.append_event(conversation.id, "user", "old user")
    store.append_event(conversation.id, "assistant", "old answer")
    store.append_clear_boundary(conversation.id)
    store.append_event(conversation.id, "user", "new user")
    store.append_event(conversation.id, "assistant", "new answer")

    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )

    assert manager.active_messages() == (
        AgentMessage("user", "new user"),
        AgentMessage("assistant", "new answer"),
    )


def test_context_manager_does_not_reload_content_covered_by_latest_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("bounded restore", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"old-user-{index}")
        store.append_event(conversation.id, "assistant", f"old-answer-{index}")
    store.append_context_summary(
        conversation.id,
        "summary",
        through_sequence=16,
        preserved_user_sequences=(1, 3),
        trigger="automatic",
        estimated_before=100,
        estimated_after=50,
        format_version=1,
    )
    store.append_event(conversation.id, "user", "new user")
    store.append_event(conversation.id, "assistant", "new answer")
    loaded_boundaries: list[int] = []
    loaded_sequences: list[tuple[int, ...]] = []
    original_load = store.load_events
    original_load_sequences = store.load_events_by_sequences

    def track_load(conversation_id, *, after_sequence=0, through_sequence=None):
        loaded_boundaries.append(after_sequence)
        return original_load(
            conversation_id,
            after_sequence=after_sequence,
            through_sequence=through_sequence,
        )

    def track_sequences(conversation_id, sequences):
        loaded_sequences.append(sequences)
        return original_load_sequences(conversation_id, sequences)

    monkeypatch.setattr(store, "load_events", track_load)
    monkeypatch.setattr(store, "load_events_by_sequences", track_sequences)
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )

    messages = manager.active_messages()

    assert [message.content for message in messages] == [
        "old-user-0",
        "old-user-1",
        "new user",
        "new answer",
    ]
    assert loaded_sequences == [(1, 3)]
    assert loaded_boundaries == [16]


def test_context_manager_uses_in_memory_messages_without_a_store(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentMessage

    fallback = (AgentMessage("user", "one"), AgentMessage("assistant", "two"))
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
    )

    assert manager.active_messages(fallback) == fallback


def test_context_manager_reset_clears_runtime_anchor_failures_and_breaker(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentMessage, TokenUsage
    from fakuicode.providers.base import AgentRequest

    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
    )
    request = AgentRequest((AgentMessage("user", "hello"),), ())
    manager.observe_usage(request, TokenUsage(context_input_tokens=10))
    for _ in range(3):
        manager.record_summary_failure()

    assert manager.usage_anchor is not None
    assert manager.consecutive_summary_failures == 3
    assert manager.automatic_compaction_disabled is True

    manager.reset()

    assert manager.usage_anchor is None
    assert manager.consecutive_summary_failures == 0
    assert manager.automatic_compaction_disabled is False


def _persist_tool_round(store, conversation_id: str, outputs: list[str]) -> None:
    calls = [
        {"id": f"call-{index}", "name": "run_command", "arguments": {"n": index}}
        for index in range(len(outputs))
    ]
    store.append_event(
        conversation_id,
        "assistant",
        "running",
        metadata={"tool_calls": calls},
    )
    for index, output in enumerate(outputs):
        store.append_event(
            conversation_id,
            "tool_call",
            "run_command",
            call_id=f"call-{index}",
            metadata={"arguments": {"n": index}},
        )
        store.append_event(
            conversation_id,
            "tool_result",
            output,
            call_id=f"call-{index}",
            metadata={
                "tool_name": "run_command",
                "success": True,
                "summary": "command complete",
            },
        )


def test_light_preparation_offloads_a_large_result_without_changing_the_timeline(
    tmp_path: Path,
) -> None:
    from fakuicode.context import approximate_token_count
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import TokenUsage
    from fakuicode.providers.base import AgentRequest
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("offload", tmp_path, "default")
    output = "head-marker\n" + "x" * 24_000 + "\ntail-marker"
    _persist_tool_round(store, conversation.id, [output])
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )
    request = AgentRequest(manager.active_messages(), ())
    manager.observe_usage(request, TokenUsage(context_input_tokens=7_000))

    prepared = manager.prepare_light(request)

    assert prepared.status is None
    assert prepared.structure_changed is True
    assert prepared.anchor_invalidated is True
    assert manager.usage_anchor is None
    assert len(prepared.artifacts) == 1
    artifact = prepared.artifacts[0]
    assert (tmp_path / artifact.read_path).read_text(encoding="utf-8") == output
    preview = prepared.request.messages[-1].tool_results[0].output
    assert "head-marker" in preview and "tail-marker" in preview
    assert artifact.read_path in preview
    assert approximate_token_count(preview) <= 500
    assert [
        event.content
        for event in store.load_events(conversation.id)
        if event.kind == "tool_result"
    ] == [output]

    repeated = manager.prepare_light(request)
    assert repeated.artifacts[0].read_path == artifact.read_path
    assert len(list((tmp_path / artifact.read_path).parent.glob("*.txt"))) == 1
    restored_manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )
    restored = restored_manager.prepare_light(
        AgentRequest(restored_manager.active_messages(), ())
    )
    assert restored.artifacts[0].read_path == artifact.read_path
    diagnostics = [
        event
        for event in store.load_events(conversation.id)
        if event.kind == "context_diagnostic"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].content == ""
    assert diagnostics[0].metadata is not None
    assert diagnostics[0].metadata["result"] == "offloaded"
    assert diagnostics[0].metadata["artifact_count"] == 1
    assert diagnostics[0].metadata["artifact_bytes"] == len(output.encode("utf-8"))
    assert diagnostics[0].metadata["threshold"] == 5_000
    assert diagnostics[0].metadata["estimated_before"] > diagnostics[0].metadata["estimated_after"]
    assert isinstance(diagnostics[0].metadata["duration_ms"], int)
    assert "head-marker" not in str(diagnostics[0].metadata)
    assert artifact.read_path not in str(diagnostics[0].metadata)


def test_light_preparation_offloads_largest_results_until_the_round_is_bounded(
    tmp_path: Path,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.providers.base import AgentRequest
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("round", tmp_path, "default")
    outputs = ["a" * 16_000, "b" * 15_000, "c" * 14_000]
    _persist_tool_round(store, conversation.id, outputs)
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )

    prepared = manager.prepare_light(AgentRequest(manager.active_messages(), ()))

    assert len(prepared.artifacts) == 1
    previews = [result.output for result in prepared.request.messages[-1].tool_results]
    assert prepared.artifacts[0].source_sequence == 3
    assert previews[0] != outputs[0]
    assert previews[1:] == outputs[1:]
    diagnostic = [
        event for event in store.load_events(conversation.id) if event.kind == "context_diagnostic"
    ][0]
    assert diagnostic.metadata["threshold"] == 10_000


def test_diagnostic_write_failure_does_not_block_context_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.providers.base import AgentRequest
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("diagnostic", tmp_path, "default")
    _persist_tool_round(store, conversation.id, ["x" * 24_000])
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )
    monkeypatch.setattr(
        store,
        "append_context_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("diagnostic unavailable")),
    )

    prepared = manager.prepare_light(AgentRequest(manager.active_messages(), ()))

    assert len(prepared.artifacts) == 1
    assert manager.diagnostic_write_failed is True


def test_light_preparation_blocks_when_artifact_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakuicode.context_manager import ContextManagementError, ContextManager
    from fakuicode.providers.base import AgentRequest
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("failure", tmp_path, "default")
    output = "x" * 24_000
    _persist_tool_round(store, conversation.id, [output])
    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
    )
    request = AgentRequest(manager.active_messages(), ())
    assert manager.artifact_store is not None
    monkeypatch.setattr(
        manager.artifact_store,
        "write_tool_result",
        lambda **kwargs: (_ for _ in ()).throw(OSError("injected write failure")),
    )

    with pytest.raises(ContextManagementError, match="artifact"):
        manager.prepare_light(request)

    assert request.messages[-1].tool_results[0].output == output
    events = store.load_events(conversation.id)
    assert [event.content for event in events if event.kind == "tool_result"] == [output]
    diagnostic = [event for event in events if event.kind == "context_diagnostic"][-1]
    assert diagnostic.content == ""
    assert diagnostic.metadata["result"] == "blocked"
    assert diagnostic.metadata["error_category"] == "artifact_write"


def _request_with_estimated_tokens(target: int):
    from fakuicode.context import estimate_request_tokens
    from fakuicode.models import AgentMessage
    from fakuicode.providers.base import AgentRequest

    empty = AgentRequest((AgentMessage("user", ""),), ())
    overhead = estimate_request_tokens(empty)
    request = AgentRequest(
        (AgentMessage("user", "x" * max(0, (target - overhead) * 4)),),
        (),
    )
    return request


def test_budget_assessment_uses_13k_automatic_and_6k_hard_reserves(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager

    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=128_000,
    )

    below = manager.assess_request(_request_with_estimated_tokens(114_900))
    automatic = manager.assess_request(_request_with_estimated_tokens(115_100))
    hard = manager.assess_request(_request_with_estimated_tokens(122_100))

    assert below.automatic_compaction_required is False
    assert below.safe_to_send is True
    assert automatic.automatic_compaction_required is True
    assert automatic.safe_to_send is True
    assert hard.automatic_compaction_required is True
    assert hard.safe_to_send is False
    assert automatic.automatic_trigger_tokens == 115_000
    assert automatic.hard_input_limit_tokens == 122_000


@pytest.mark.parametrize("trigger", ["automatic", "manual", "emergency"])
def test_hard_limit_blocks_every_request_type_with_an_actionable_hint(
    tmp_path: Path, trigger: str
) -> None:
    from fakuicode.context_manager import ContextLimitError, ContextManager

    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=20_000,
    )
    request = _request_with_estimated_tokens(14_100)

    with pytest.raises(ContextLimitError) as error:
        manager.ensure_hard_limit(request, trigger=trigger)

    assert error.value.status.result == "blocked"
    assert error.value.status.trigger == trigger
    assert "/compact" in error.value.status.recovery_hint
    assert "/clear" in error.value.status.recovery_hint


def _valid_summary(section_text: str = "无") -> str:
    from fakuicode.context import SUMMARY_HEADINGS

    return "\n\n".join(f"## {heading}\n{section_text}" for heading in SUMMARY_HEADINGS)


def test_summary_request_uses_current_provider_without_tools_or_recursive_preparation(
    tmp_path: Path,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentStreamEvent, TimelineEvent

    class Provider:
        def __init__(self) -> None:
            self.calls = []

        def stream_agent(self, messages, tools, *, cancel_event=None, request=None):
            self.calls.append((messages, tools, cancel_event, request))
            yield AgentStreamEvent("text_delta", _valid_summary("verified"))
            yield AgentStreamEvent("completed")

    provider = Provider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
    )

    summary = manager.generate_summary(
        [TimelineEvent(1, "user", "keep the original requirement")],
        trigger="automatic",
    )

    assert summary == _valid_summary("verified")
    assert len(provider.calls) == 1
    messages, tools, cancel_event, request = provider.calls[0]
    assert tools == ()
    assert cancel_event is None
    assert request.tools == ()
    assert request.output_token_limit == 4_000
    prompt = messages[0].content
    assert "不得调用任何工具" in prompt
    assert "先在内部组织分析草稿" in prompt
    assert "只输出正式摘要" in prompt
    assert "keep the original requirement" in prompt


@pytest.mark.parametrize("behavior", ["tool_call", "incomplete", "invalid_format"])
def test_summary_generation_rejects_non_summary_provider_results(
    tmp_path: Path, behavior: str
) -> None:
    from fakuicode.context_manager import ContextManager, SummaryGenerationError
    from fakuicode.models import AgentStreamEvent, TimelineEvent, ToolCall

    class Provider:
        def stream_agent(self, messages, tools, *, request=None):
            del messages, tools, request
            if behavior == "tool_call":
                yield AgentStreamEvent(
                    "tool_call",
                    tool_call=ToolCall("call-1", "read_file", {"path": "README.md"}),
                )
                yield AgentStreamEvent("completed")
            elif behavior == "incomplete":
                yield AgentStreamEvent("text_delta", _valid_summary())
            else:
                yield AgentStreamEvent("text_delta", "not a structured summary")
                yield AgentStreamEvent("completed")

    manager = ContextManager(
        Provider(),
        workspace=tmp_path,
        context_window=128_000,
    )

    with pytest.raises(SummaryGenerationError):
        manager.generate_summary(
            [TimelineEvent(1, "user", "hello")],
            trigger="automatic",
        )


def test_oversized_summary_request_is_blocked_before_calling_the_provider(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextLimitError, ContextManager
    from fakuicode.models import TimelineEvent

    provider = _NoopProvider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=8_000,
    )

    with pytest.raises(ContextLimitError):
        manager.generate_summary(
            [TimelineEvent(1, "user", "x" * 12_000)],
            trigger="manual",
        )


def test_compaction_persists_one_rolling_summary_and_rebuilds_a_system_boundary(
    tmp_path: Path,
) -> None:
    from fakuicode.context import COMPACTION_BOUNDARY_MESSAGE, ContextPolicy
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentStreamEvent
    from fakuicode.providers.base import AgentRequest
    from fakuicode.storage import ConversationStore

    class Provider:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def stream_agent(self, messages, tools, *, request=None):
            del tools, request
            self.prompts.append(messages[0].content)
            marker = f"summary-{len(self.prompts)}"
            yield AgentStreamEvent("text_delta", _valid_summary(marker))
            yield AgentStreamEvent("completed")

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("rolling", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"user-{index} " + "u" * 200)
        store.append_event(conversation.id, "assistant", f"answer-{index} " + "a" * 200)
    original_users = [
        event.content for event in store.load_events(conversation.id) if event.kind == "user"
    ]
    provider = Provider()
    policy = ContextPolicy(
        recent_history_target_tokens=40,
        recent_history_min_groups=2,
        older_user_messages_target_tokens=200,
    )
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=policy,
    )

    first = manager.compact_request(
        AgentRequest(manager.active_messages(), (), system_supplement="mode reminder"),
        trigger="manual",
    )

    assert first.status is not None and first.status.result == "compacted"
    assert "summary-1" in first.request.system_supplement
    assert COMPACTION_BOUNDARY_MESSAGE in first.request.system_supplement
    assert "mode reminder" in first.request.system_supplement
    assert all(
        message.content != COMPACTION_BOUNDARY_MESSAGE for message in first.request.messages
    )
    stored_first = store.load_latest_context_summary(conversation.id)
    assert stored_first is not None
    assert stored_first.metadata is not None
    assert stored_first.metadata["through_sequence"] > 0
    assert isinstance(stored_first.metadata["preserved_user_sequences"], list)

    store.append_event(conversation.id, "user", "new user " + "n" * 200)
    store.append_event(conversation.id, "assistant", "new answer " + "z" * 200)
    second = manager.compact_request(
        AgentRequest(manager.active_messages(), (), system_supplement="mode reminder"),
        trigger="manual",
    )

    assert len(provider.prompts) == 2
    assert "summary-1" in provider.prompts[1]
    assert "summary-2" in second.request.system_supplement
    assert "summary-1" not in second.request.system_supplement
    assert len(
        [event for event in store.load_events(conversation.id) if event.kind == "summary"]
    ) == 2
    assert [
        event.content for event in store.load_events(conversation.id) if event.kind == "user"
    ] == original_users + ["new user " + "n" * 200]
    diagnostics = [
        event
        for event in store.load_events(conversation.id)
        if event.kind == "context_diagnostic"
    ]
    assert [event.metadata["result"] for event in diagnostics] == ["compacted", "compacted"]
    assert all(event.content == "" for event in diagnostics)

    restored_manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=policy,
    )
    restored = restored_manager.activate_request(
        AgentRequest((), (), system_supplement="mode reminder")
    )
    assert restored.system_supplement.count("summary-2") == 8
    assert "summary-1" not in restored.system_supplement
    assert restored.messages == restored_manager.active_messages()


def _automatic_test_policy():
    from fakuicode.context import ContextPolicy

    return ContextPolicy(
        automatic_reserve_tokens=127_900,
        hard_reserve_tokens=6_000,
        recent_history_target_tokens=40,
        recent_history_min_groups=2,
        older_user_messages_target_tokens=200,
    )


def _store_with_compaction_history(tmp_path: Path):
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("breaker", tmp_path, "default")
    for index in range(8):
        store.append_event(conversation.id, "user", f"user-{index} " + "u" * 200)
        store.append_event(conversation.id, "assistant", f"answer-{index} " + "a" * 200)
    return store, conversation


def test_three_summary_failures_trip_the_automatic_breaker_and_skip_the_fourth_call(
    tmp_path: Path,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentStreamEvent
    from fakuicode.providers.base import AgentRequest

    class InvalidProvider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, request=None):
            del messages, tools, request
            self.calls += 1
            yield AgentStreamEvent("text_delta", "invalid summary")
            yield AgentStreamEvent("completed")

    store, conversation = _store_with_compaction_history(tmp_path)
    provider = InvalidProvider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=_automatic_test_policy(),
    )
    request = AgentRequest(manager.active_messages(), ())

    statuses = [manager.prepare_request(request).status for _ in range(3)]
    skipped = manager.prepare_request(request)

    assert [status.consecutive_failures for status in statuses if status is not None] == [
        1,
        2,
        3,
    ]
    assert all(status is not None and status.result == "failed" for status in statuses)
    assert skipped.status is not None and skipped.status.result == "breaker"
    assert provider.calls == 3
    assert manager.automatic_compaction_disabled is True
    diagnostics = [
        event.metadata
        for event in store.load_events(conversation.id)
        if event.kind == "context_diagnostic"
    ]
    assert [metadata["result"] for metadata in diagnostics] == [
        "failed",
        "failed",
        "failed",
        "breaker",
    ]
    assert [metadata["consecutive_failures"] for metadata in diagnostics] == [1, 2, 3, 3]
    assert all("invalid summary" not in str(metadata) for metadata in diagnostics)


def test_manual_compaction_bypasses_breaker_and_success_resets_it(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentStreamEvent
    from fakuicode.providers.base import AgentRequest

    class RecoveringProvider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, request=None):
            del messages, tools, request
            self.calls += 1
            response = "invalid" if self.calls <= 3 else _valid_summary("recovered")
            yield AgentStreamEvent("text_delta", response)
            yield AgentStreamEvent("completed")

    store, conversation = _store_with_compaction_history(tmp_path)
    provider = RecoveringProvider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=_automatic_test_policy(),
    )
    request = AgentRequest(manager.active_messages(), ())
    for _ in range(3):
        manager.prepare_request(request)

    recovered = manager.compact_manually(request)

    assert recovered.status is not None and recovered.status.result == "compacted"
    assert provider.calls == 4
    assert manager.consecutive_summary_failures == 0
    assert manager.automatic_compaction_disabled is False


def test_failed_manual_compaction_keeps_the_breaker_open(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.models import AgentStreamEvent
    from fakuicode.providers.base import AgentRequest

    class InvalidProvider:
        def __init__(self) -> None:
            self.calls = 0

        def stream_agent(self, messages, tools, *, request=None):
            del messages, tools, request
            self.calls += 1
            yield AgentStreamEvent("text_delta", "invalid")
            yield AgentStreamEvent("completed")

    store, conversation = _store_with_compaction_history(tmp_path)
    provider = InvalidProvider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=_automatic_test_policy(),
    )
    request = AgentRequest(manager.active_messages(), ())
    for _ in range(3):
        manager.prepare_request(request)

    failed = manager.compact_manually(request)

    assert failed.status is not None and failed.status.result == "failed"
    assert provider.calls == 4
    assert manager.consecutive_summary_failures == 4
    assert manager.automatic_compaction_disabled is True


def test_failed_manual_compaction_returns_actionable_status_even_above_hard_limit(
    tmp_path: Path,
) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.providers.base import AgentRequest

    store, conversation = _store_with_compaction_history(tmp_path)
    provider = _NoopProvider()
    manager = ContextManager(
        provider,
        workspace=tmp_path,
        context_window=5_000,
        store=store,
        conversation_id=conversation.id,
    )

    result = manager.compact_manually(AgentRequest(manager.active_messages(), ()))

    assert result.status is not None
    assert result.status.trigger == "manual"
    assert result.status.result == "failed"
    assert result.status.consecutive_failures == 1
    assert "/compact" in result.status.recovery_hint
    assert "/clear" in result.status.recovery_hint


def test_cancelled_summary_does_not_increment_failures(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextManager
    from fakuicode.errors import RequestCancelled
    from fakuicode.providers.base import AgentRequest

    class CancelledProvider:
        def stream_agent(self, messages, tools, *, request=None):
            del messages, tools, request
            raise RequestCancelled()
            yield

    store, conversation = _store_with_compaction_history(tmp_path)
    manager = ContextManager(
        CancelledProvider(),
        workspace=tmp_path,
        context_window=128_000,
        store=store,
        conversation_id=conversation.id,
        policy=_automatic_test_policy(),
    )

    with pytest.raises(RequestCancelled):
        manager.prepare_request(AgentRequest(manager.active_messages(), ()))

    assert manager.consecutive_summary_failures == 0


def test_breaker_blocks_an_unsafe_request_without_calling_summary_provider(tmp_path: Path) -> None:
    from fakuicode.context_manager import ContextLimitError, ContextManager

    manager = ContextManager(
        _NoopProvider(),
        workspace=tmp_path,
        context_window=8_000,
        policy=_automatic_test_policy(),
    )
    for _ in range(3):
        manager.record_summary_failure()

    with pytest.raises(ContextLimitError) as error:
        manager.prepare_request(_request_with_estimated_tokens(2_100))

    assert error.value.status.consecutive_failures == 3
    assert error.value.status.result == "blocked"
