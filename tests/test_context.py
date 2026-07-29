from __future__ import annotations

import pytest

from fakuicode.models import (
    AgentMessage,
    ProviderMessageState,
    TimelineEvent,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from fakuicode.providers.base import AgentRequest


def test_context_policy_exposes_the_approved_mvp_defaults() -> None:
    from fakuicode.context import ContextPolicy

    policy = ContextPolicy()

    assert policy.single_tool_result_tokens == 5_000
    assert policy.tool_round_tokens == 10_000
    assert policy.tool_preview_ratio == 0.10
    assert policy.tool_preview_max_tokens == 500
    assert policy.automatic_reserve_tokens == 13_000
    assert policy.hard_reserve_tokens == 6_000
    assert policy.recent_history_target_tokens == 10_000
    assert policy.recent_history_min_groups == 5
    assert policy.older_user_messages_target_tokens == 20_000
    assert policy.summary_target_tokens == 2_000
    assert policy.summary_hard_max_tokens == 4_000
    assert policy.summary_failure_limit == 3
    assert policy.overflow_retry_limit == 1
    assert policy.automatic_trigger_tokens(128_000) == 115_000
    assert policy.hard_input_limit_tokens(128_000) == 122_000


def test_context_policy_fails_closed_for_windows_smaller_than_a_reserve() -> None:
    from fakuicode.context import ContextPolicy

    policy = ContextPolicy()

    assert policy.automatic_trigger_tokens(10_000) == 0
    assert policy.hard_input_limit_tokens(5_000) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"single_tool_result_tokens": 0},
        {"tool_preview_ratio": 1.1},
        {"recent_history_min_groups": 0},
        {"summary_target_tokens": 4_001},
        {"automatic_reserve_tokens": 5_999},
    ],
)
def test_context_policy_rejects_invalid_internal_defaults(overrides: dict[str, int | float]) -> None:
    from fakuicode.context import ContextPolicy

    with pytest.raises(ValueError):
        ContextPolicy(**overrides)


def test_approximate_token_count_uses_utf8_bytes_and_rounds_up() -> None:
    from fakuicode.context import approximate_token_count

    assert approximate_token_count("") == 0
    assert approximate_token_count("abcd") == 1
    assert approximate_token_count("abcde") == 2
    assert approximate_token_count("你") == 1
    assert approximate_token_count("你好") == 2


def test_request_serialization_is_stable_across_mapping_insertion_order() -> None:
    from fakuicode.context import serialize_agent_request

    first = AgentRequest(
        messages=(AgentMessage("assistant", tool_calls=(ToolCall("1", "read", {"path": "a", "limit": 2}),)),),
        tools=(ToolDefinition("read", "Read", {"type": "object", "properties": {"path": {"type": "string"}}}),),
        system_prompt="stable",
        system_supplement="dynamic",
    )
    second = AgentRequest(
        messages=(AgentMessage("assistant", tool_calls=(ToolCall("1", "read", {"limit": 2, "path": "a"}),)),),
        tools=(ToolDefinition("read", "Read", {"properties": {"path": {"type": "string"}}, "type": "object"}),),
        system_prompt="stable",
        system_supplement="dynamic",
    )

    assert serialize_agent_request(first) == serialize_agent_request(second)


def test_usage_anchor_estimates_only_messages_appended_to_an_unchanged_request() -> None:
    from fakuicode.context import UsageAnchor, approximate_token_count, estimate_request_tokens, serialize_agent_messages

    original = AgentRequest(messages=(AgentMessage("user", "first"),), tools=(), system_prompt="system")
    appended_message = AgentMessage("assistant", "second")
    extended = AgentRequest(messages=original.messages + (appended_message,), tools=(), system_prompt="system")
    anchor = UsageAnchor.from_request(original, context_input_tokens=1_000)

    assert estimate_request_tokens(extended, anchor=anchor) == 1_000 + approximate_token_count(
        serialize_agent_messages((appended_message,))
    )


@pytest.mark.parametrize("change", ["system", "supplement", "tools", "earlier_message", "clear"])
def test_usage_anchor_is_not_reused_after_the_anchored_request_changes(change: str) -> None:
    from fakuicode.context import UsageAnchor, approximate_token_count, estimate_request_tokens, serialize_agent_request

    tool = ToolDefinition("read", "Read", {"type": "object"})
    original = AgentRequest(
        messages=(AgentMessage("user", "first"), AgentMessage("assistant", "answer")),
        tools=(tool,),
        system_prompt="system",
        system_supplement="supplement",
    )
    changed = {
        "system": AgentRequest(original.messages, original.tools, "changed", original.system_supplement),
        "supplement": AgentRequest(original.messages, original.tools, original.system_prompt, "changed"),
        "tools": AgentRequest(original.messages, (), original.system_prompt, original.system_supplement),
        "earlier_message": AgentRequest(
            (AgentMessage("user", "rewritten"), original.messages[1]),
            original.tools,
            original.system_prompt,
            original.system_supplement,
        ),
        "clear": AgentRequest((), original.tools, original.system_prompt, original.system_supplement),
    }[change]
    anchor = UsageAnchor.from_request(original, context_input_tokens=100_000)

    assert estimate_request_tokens(changed, anchor=anchor) == approximate_token_count(serialize_agent_request(changed))


def test_request_serialization_includes_tool_result_payloads_but_not_cancel_state() -> None:
    from threading import Event

    from fakuicode.context import serialize_agent_request

    request = AgentRequest(
        messages=(
            AgentMessage(
                "user",
                tool_results=(ToolResult("1", "read", True, "complete output", "read file"),),
            ),
        ),
        tools=(),
        cancel_event=Event(),
    )

    serialized = serialize_agent_request(request)

    assert "complete output" in serialized
    assert "cancel" not in serialized


def test_context_groups_keep_ordinary_messages_separate_and_report_user_sequences() -> None:
    from fakuicode.context import group_context_events

    groups = group_context_events(
        [
            TimelineEvent(2, "user", "request"),
            TimelineEvent(3, "assistant", "answer"),
        ]
    )

    assert [group.start_sequence for group in groups] == [2, 3]
    assert [group.end_sequence for group in groups] == [2, 3]
    assert [group.user_sequences for group in groups] == [(2,), ()]
    assert all(group.estimated_tokens > 0 for group in groups)


def test_context_groups_keep_multiple_tool_calls_and_results_in_one_ordered_group() -> None:
    from fakuicode.context import group_context_events

    events = [
        TimelineEvent(
            1,
            "assistant",
            "checking",
            metadata={
                "tool_calls": [
                    {"id": "a", "name": "read", "arguments": {"path": "a"}},
                    {"id": "b", "name": "read", "arguments": {"path": "b"}},
                ]
            },
        ),
        TimelineEvent(2, "tool_call", "read", call_id="a"),
        TimelineEvent(3, "tool_call", "read", call_id="b"),
        TimelineEvent(4, "tool_result", "A", call_id="a", metadata={"tool_name": "read", "success": True}),
        TimelineEvent(5, "tool_result", "B", call_id="b", metadata={"tool_name": "read", "success": True}),
        TimelineEvent(6, "assistant", "done"),
    ]

    groups = group_context_events(events)

    assert [[event.sequence for event in group.events] for group in groups] == [[1, 2, 3, 4, 5], [6]]


def test_context_groups_conservatively_keep_incomplete_or_legacy_tool_segments_together() -> None:
    from fakuicode.context import group_context_events

    events = [
        TimelineEvent(1, "assistant", "legacy assistant without call metadata"),
        TimelineEvent(2, "tool_call", "read", call_id="legacy"),
        TimelineEvent(3, "tool_result", "output", call_id="legacy", metadata={"tool_name": "read", "success": True}),
        TimelineEvent(
            4,
            "assistant",
            "new call missing its result",
            metadata={"tool_calls": [{"id": "missing", "name": "read", "arguments": {}}]},
        ),
        TimelineEvent(5, "tool_call", "read", call_id="missing"),
    ]

    groups = group_context_events(events)

    assert [[event.sequence for event in group.events] for group in groups] == [[1, 2, 3], [4, 5]]


def test_context_groups_keep_orphan_tool_events_without_dropping_or_reordering_them() -> None:
    from fakuicode.context import group_context_events

    events = [
        TimelineEvent(7, "tool_call", "read", call_id="orphan"),
        TimelineEvent(8, "tool_result", "output", call_id="orphan", metadata={"tool_name": "read", "success": True}),
        TimelineEvent(9, "user", "continue"),
    ]

    groups = group_context_events(events)

    assert [[event.sequence for event in group.events] for group in groups] == [[7, 8], [9]]


def test_lightweight_plan_offloads_a_single_result_over_five_thousand_tokens() -> None:
    from fakuicode.context import group_context_events, plan_tool_result_offloads

    output = "H" + ("x" * 23_998) + "T"
    events = [
        TimelineEvent(
            1,
            "assistant",
            "run",
            metadata={"tool_calls": [{"id": "large", "name": "command", "arguments": {}}]},
        ),
        TimelineEvent(2, "tool_call", "command", call_id="large"),
        TimelineEvent(
            3,
            "tool_result",
            output,
            call_id="large",
            metadata={"tool_name": "command", "success": True},
        ),
    ]

    candidates = plan_tool_result_offloads(group_context_events(events))

    assert len(candidates) == 1
    assert candidates[0].event_sequence == 3
    assert candidates[0].original_tokens == 6_000
    assert candidates[0].preview_budget_tokens == 500
    assert candidates[0].reason == "single_result"


def test_lightweight_plan_offloads_largest_results_until_the_round_is_under_ten_thousand_tokens() -> None:
    from fakuicode.context import group_context_events, plan_tool_result_offloads

    sizes = {"largest": 4_500, "middle": 3_500, "smallest": 3_000}
    events = [
        TimelineEvent(
            1,
            "assistant",
            "run",
            metadata={
                "tool_calls": [
                    {"id": call_id, "name": "command", "arguments": {}} for call_id in sizes
                ]
            },
        )
    ]
    for sequence, call_id in enumerate(sizes, start=2):
        events.append(TimelineEvent(sequence, "tool_call", "command", call_id=call_id))
    for sequence, (call_id, tokens) in enumerate(sizes.items(), start=5):
        events.append(
            TimelineEvent(
                sequence,
                "tool_result",
                "x" * (tokens * 4),
                call_id=call_id,
                metadata={"tool_name": "command", "success": True},
            )
        )

    candidates = plan_tool_result_offloads(group_context_events(events))

    assert [(candidate.call_id, candidate.reason) for candidate in candidates] == [("largest", "round_total")]


def test_lightweight_plan_does_not_trigger_at_exact_thresholds() -> None:
    from fakuicode.context import group_context_events, plan_tool_result_offloads

    events = [
        TimelineEvent(
            1,
            "assistant",
            "read",
            metadata={
                "tool_calls": [
                    {"id": "a", "name": "read", "arguments": {}},
                    {"id": "b", "name": "read", "arguments": {}},
                ]
            },
        ),
        TimelineEvent(2, "tool_call", "read", call_id="a"),
        TimelineEvent(3, "tool_call", "read", call_id="b"),
        TimelineEvent(4, "tool_result", "a" * 20_000, call_id="a", metadata={"tool_name": "read", "success": True}),
        TimelineEvent(5, "tool_result", "b" * 20_000, call_id="b", metadata={"tool_name": "read", "success": True}),
    ]

    assert plan_tool_result_offloads(group_context_events(events)) == []


def test_tool_result_preview_is_bounded_and_keeps_metadata_head_and_tail() -> None:
    from fakuicode.context import approximate_token_count, build_tool_result_preview

    output = "HEAD-MARKER\n" + ("x" * 23_976) + "\nTAIL-MARKER"
    preview = build_tool_result_preview(
        output,
        original_tokens=6_000,
        success=True,
        read_path=".fakuicode/context-artifacts/conversation/result.txt",
        budget_tokens=500,
    )

    assert approximate_token_count(preview) <= 500
    assert "HEAD-MARKER" in preview
    assert "TAIL-MARKER" in preview
    assert "24000" in preview
    assert "6000" in preview
    assert "成功" in preview
    assert ".fakuicode/context-artifacts/conversation/result.txt" in preview


def test_stored_tool_result_preview_is_bounded_without_the_complete_output() -> None:
    from fakuicode.context import approximate_token_count, build_stored_tool_result_preview

    preview = build_stored_tool_result_preview(
        head="HEAD-MARKER\n" + "h" * 2_000,
        tail="t" * 2_000 + "\nTAIL-MARKER",
        original_bytes=2_000_000,
        original_tokens=500_000,
        success=True,
        read_path=".fakuicode/context-artifacts/conversation/command-digest.txt",
        budget_tokens=500,
    )

    assert approximate_token_count(preview) <= 500
    assert "HEAD-MARKER" in preview
    assert "TAIL-MARKER" in preview
    assert "2000000 bytes" in preview


def test_retention_selection_keeps_recent_target_and_at_least_five_complete_groups() -> None:
    from fakuicode.context import ContextGroup, select_compaction_history

    groups = [
        ContextGroup(
            events=(TimelineEvent(sequence, "user", f"request {sequence}"),),
            start_sequence=sequence,
            end_sequence=sequence,
            estimated_tokens=2_500,
            user_sequences=(sequence,),
        )
        for sequence in range(1, 9)
    ]

    selection = select_compaction_history(groups, retained_token_budget=40_000)

    assert [group.start_sequence for group in selection.recent_groups] == [4, 5, 6, 7, 8]
    assert selection.recent_tokens == 12_500
    assert [group.start_sequence for group in selection.summary_groups] == [1, 2, 3]
    assert [event.sequence for event in selection.preserved_user_events] == [1, 2, 3]


def test_retention_selection_keeps_the_newest_older_user_messages_within_twenty_thousand_tokens() -> None:
    from fakuicode.context import ContextGroup, select_compaction_history

    groups = [
        ContextGroup(
            events=(TimelineEvent(sequence, "user", "x" * 20_000),),
            start_sequence=sequence,
            end_sequence=sequence,
            estimated_tokens=5_000,
            user_sequences=(sequence,),
        )
        for sequence in range(1, 8)
    ]
    groups.extend(
        ContextGroup(
            events=(TimelineEvent(sequence, "assistant", "recent"),),
            start_sequence=sequence,
            end_sequence=sequence,
            estimated_tokens=2_000,
            user_sequences=(),
        )
        for sequence in range(8, 13)
    )

    selection = select_compaction_history(groups, retained_token_budget=40_000)

    assert [event.sequence for event in selection.preserved_user_events] == [4, 5, 6, 7]
    assert selection.preserved_user_tokens <= 20_000


def test_retention_selection_drops_soft_targets_without_splitting_groups_when_budget_is_tight() -> None:
    from fakuicode.context import ContextGroup, select_compaction_history

    groups = [
        ContextGroup(
            events=(TimelineEvent(sequence, "assistant", f"group {sequence}"),),
            start_sequence=sequence,
            end_sequence=sequence,
            estimated_tokens=4_000,
            user_sequences=(),
        )
        for sequence in range(1, 8)
    ]

    selection = select_compaction_history(groups, retained_token_budget=12_000)

    assert [group.start_sequence for group in selection.recent_groups] == [5, 6, 7]
    assert selection.recent_tokens == 12_000
    assert [group.start_sequence for group in selection.summary_groups] == [1, 2, 3, 4]


def test_retention_selection_can_summarize_an_oversized_group_instead_of_splitting_it() -> None:
    from fakuicode.context import ContextGroup, select_compaction_history

    oversized = ContextGroup(
        events=(TimelineEvent(1, "assistant", "oversized"), TimelineEvent(2, "tool_result", "output")),
        start_sequence=1,
        end_sequence=2,
        estimated_tokens=20_000,
        user_sequences=(),
    )

    selection = select_compaction_history([oversized], retained_token_budget=10_000)

    assert selection.recent_groups == ()
    assert selection.summary_groups == (oversized,)


def _summary_with_sections(contents: list[str] | None = None) -> str:
    headings = [
        "当前目标",
        "用户要求与硬约束",
        "已确认的决策",
        "已完成工作与验证证据",
        "关键文件、符号与当前代码状态",
        "失败尝试、已排除方案与已知风险",
        "未完成工作与下一步",
        "可重新读取的源码及工具产物路径",
    ]
    section_contents = contents or [f"内容 {index}" for index in range(len(headings))]
    return "\n\n".join(f"## {heading}\n{content}" for heading, content in zip(headings, section_contents, strict=True))


def test_summary_normalization_requires_the_eight_fixed_headings_and_fills_empty_sections() -> None:
    from fakuicode.context import SUMMARY_HEADINGS, normalize_structured_summary

    normalized = normalize_structured_summary(_summary_with_sections(["goal", "", "decision", "", "state", "risk", "next", "paths"]))

    assert [line[3:] for line in normalized.splitlines() if line.startswith("## ")] == list(SUMMARY_HEADINGS)
    assert "## 用户要求与硬约束\n无" in normalized
    assert "## 已完成工作与验证证据\n无" in normalized


@pytest.mark.parametrize(
    "invalid_summary",
    [
        "analysis draft\n" + _summary_with_sections(),
        _summary_with_sections().replace("## 当前目标\n内容 0\n\n", "", 1),
        _summary_with_sections().replace("## 当前目标", "## 已确认的决策", 1),
        _summary_with_sections() + "\n\n## 当前目标\nduplicate",
        _summary_with_sections() + "\n\n## 额外标题\nnot allowed",
    ],
)
def test_summary_normalization_rejects_drafts_missing_reordered_duplicate_or_extra_headings(invalid_summary: str) -> None:
    from fakuicode.context import normalize_structured_summary

    with pytest.raises(ValueError):
        normalize_structured_summary(invalid_summary)


def test_summary_normalization_rejects_content_over_four_thousand_tokens() -> None:
    from fakuicode.context import normalize_structured_summary

    oversized = _summary_with_sections(["x" * 17_000] + ["content"] * 7)

    with pytest.raises(ValueError):
        normalize_structured_summary(oversized)


def test_compaction_boundary_warns_to_reread_sources_without_impersonating_a_user() -> None:
    from fakuicode.context import COMPACTION_BOUNDARY_MESSAGE

    assert "摘要可能省略细节" in COMPACTION_BOUNDARY_MESSAGE
    assert "重新读取" in COMPACTION_BOUNDARY_MESSAGE
    assert "不得" in COMPACTION_BOUNDARY_MESSAGE
    assert "用户说" not in COMPACTION_BOUNDARY_MESSAGE


def test_context_builder_requests_a_summary_and_keeps_recent_original_messages() -> None:
    from fakuicode.context import ContextBuilder

    events = [
        TimelineEvent(1, "user", "old request " * 20),
        TimelineEvent(2, "assistant", "old answer " * 20),
        TimelineEvent(3, "user", "recent request"),
        TimelineEvent(4, "assistant", "recent answer"),
    ]

    plan = ContextBuilder(max_characters=100).build(events)

    assert [event.sequence for event in plan.events_for_summary] == [1, 2]
    assert [message.content for message in plan.messages] == ["recent request", "recent answer"]


def test_context_builder_uses_existing_summary_without_discarding_original_history() -> None:
    from fakuicode.context import ContextBuilder

    events = [
        TimelineEvent(1, "user", "original request"),
        TimelineEvent(2, "assistant", "original answer"),
        TimelineEvent(3, "summary", "Earlier work was completed.", metadata={"through_sequence": 2}),
        TimelineEvent(4, "user", "what changed next?"),
    ]

    plan = ContextBuilder(max_characters=1_000).build(events)

    assert [(message.role, message.content) for message in plan.messages] == [
        ("assistant", "Earlier work was completed."),
        ("user", "what changed next?"),
    ]
    assert [event.sequence for event in events[:2]] == [1, 2]


def test_local_summary_is_bounded_and_labels_original_event_roles() -> None:
    from fakuicode.context import summarize_events

    summary = summarize_events([TimelineEvent(1, "user", "first request"), TimelineEvent(2, "assistant", "first answer")])

    assert "user: first request" in summary
    assert "assistant: first answer" in summary


def test_context_builder_rebuilds_a_tool_call_and_result_as_an_atomic_native_history_group() -> None:
    from fakuicode.context import ContextBuilder

    events = [
        TimelineEvent(1, "user", "inspect README"),
        TimelineEvent(
            2,
            "assistant",
            "I will inspect it.",
            metadata={"tool_calls": [{"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}}]},
        ),
        TimelineEvent(3, "tool_call", "read_file", call_id="call-1", metadata={"arguments": {"path": "README.md"}}),
        TimelineEvent(
            4,
            "tool_result",
            "contents",
            call_id="call-1",
            metadata={"tool_name": "read_file", "success": True, "summary": "read README.md"},
        ),
        TimelineEvent(5, "assistant", "It is complete."),
    ]

    plan = ContextBuilder(max_characters=1_000).build(events)

    assert [(message.role, message.content) for message in plan.messages] == [
        ("user", "inspect README"),
        ("assistant", "I will inspect it."),
        ("user", ""),
        ("assistant", "It is complete."),
    ]
    assert plan.messages[1].tool_calls[0].arguments == {"path": "README.md"}
    assert plan.messages[2].tool_results[0].summary == "read README.md"


def test_context_rebuilds_provider_state_required_by_an_incomplete_tool_cycle() -> None:
    from fakuicode.context import messages_from_events

    thinking_block = {
        "type": "thinking",
        "thinking": "inspect first",
        "signature": "signed-reasoning",
    }
    events = [
        TimelineEvent(
            1,
            "assistant",
            "",
            metadata={
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                ],
                "provider_state": {
                    "protocol": "anthropic",
                    "thinking_blocks": [thinking_block],
                },
            },
        ),
        TimelineEvent(
            2,
            "tool_result",
            "contents",
            call_id="tool-1",
            metadata={
                "tool_name": "read_file",
                "success": True,
                "summary": "read README.md",
            },
        ),
    ]

    messages = messages_from_events(events)

    assert messages[0].provider_state == ProviderMessageState(
        "anthropic",
        (thinking_block,),
    )
