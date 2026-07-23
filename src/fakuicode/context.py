"""Model-context selection that preserves complete local tool history."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Mapping, Sequence

from fakuicode.models import AgentMessage, ProviderMessageState, TimelineEvent, ToolCall, ToolResult

if TYPE_CHECKING:
    from fakuicode.providers.base import AgentRequest


SUMMARY_HEADINGS = (
    "当前目标",
    "用户要求与硬约束",
    "已确认的决策",
    "已完成工作与验证证据",
    "关键文件、符号与当前代码状态",
    "失败尝试、已排除方案与已知风险",
    "未完成工作与下一步",
    "可重新读取的源码及工具产物路径",
)

COMPACTION_BOUNDARY_MESSAGE = (
    "此前历史已压缩为结构化摘要，摘要可能省略细节。"
    "涉及代码、文件或工具结果的具体内容时，必须重新读取摘要列出的来源或上下文产物，"
    "不得根据摘要猜测、补全或编造细节。"
)


@dataclass(frozen=True)
class ContextPolicy:
    """Centralized, adjustable MVP defaults for active-context management."""

    single_tool_result_tokens: int = 5_000
    tool_round_tokens: int = 10_000
    tool_preview_ratio: float = 0.10
    tool_preview_max_tokens: int = 500
    automatic_reserve_tokens: int = 13_000
    hard_reserve_tokens: int = 6_000
    recent_history_target_tokens: int = 10_000
    recent_history_min_groups: int = 5
    older_user_messages_target_tokens: int = 20_000
    summary_target_tokens: int = 2_000
    summary_hard_max_tokens: int = 4_000
    summary_failure_limit: int = 3
    overflow_retry_limit: int = 1

    def __post_init__(self) -> None:
        integer_values = (
            self.single_tool_result_tokens,
            self.tool_round_tokens,
            self.tool_preview_max_tokens,
            self.automatic_reserve_tokens,
            self.hard_reserve_tokens,
            self.recent_history_target_tokens,
            self.recent_history_min_groups,
            self.older_user_messages_target_tokens,
            self.summary_target_tokens,
            self.summary_hard_max_tokens,
            self.summary_failure_limit,
            self.overflow_retry_limit,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("Context policy token limits and counts must be positive.")
        if not 0 < self.tool_preview_ratio <= 1:
            raise ValueError("tool_preview_ratio must be in the interval (0, 1].")
        if self.summary_target_tokens > self.summary_hard_max_tokens:
            raise ValueError("summary_target_tokens cannot exceed summary_hard_max_tokens.")
        if self.automatic_reserve_tokens < self.hard_reserve_tokens:
            raise ValueError("automatic_reserve_tokens cannot be smaller than hard_reserve_tokens.")

    def automatic_trigger_tokens(self, context_window: int) -> int:
        return max(0, context_window - self.automatic_reserve_tokens)

    def hard_input_limit_tokens(self, context_window: int) -> int:
        return max(0, context_window - self.hard_reserve_tokens)


def approximate_token_count(value: str) -> int:
    """Estimate tokens as UTF-8 bytes divided by four, rounded up."""

    byte_count = len(value.encode("utf-8"))
    return (byte_count + 3) // 4


def normalize_structured_summary(summary: str, *, policy: ContextPolicy | None = None) -> str:
    """Validate and normalize the single allowed structured-summary shape."""

    active_policy = policy or ContextPolicy()
    stripped = summary.strip()
    lines = stripped.splitlines()
    heading_rows = [(index, line[3:].strip()) for index, line in enumerate(lines) if line.startswith("## ")]
    if [heading for _, heading in heading_rows] != list(SUMMARY_HEADINGS):
        raise ValueError("The summary must contain exactly the eight required headings in order.")
    if not heading_rows or heading_rows[0][0] != 0:
        raise ValueError("The summary cannot contain a draft or preamble before its first heading.")

    sections: list[str] = []
    for position, (line_index, heading) in enumerate(heading_rows):
        next_index = heading_rows[position + 1][0] if position + 1 < len(heading_rows) else len(lines)
        content = "\n".join(lines[line_index + 1 : next_index]).strip() or "无"
        sections.append(f"## {heading}\n{content}")
    normalized = "\n\n".join(sections)
    if approximate_token_count(normalized) > active_policy.summary_hard_max_tokens:
        raise ValueError("The structured summary exceeds the hard token limit.")
    return normalized


def serialize_agent_messages(messages: Sequence[AgentMessage]) -> str:
    """Serialize model-visible messages deterministically for estimation."""

    return _stable_json([_message_payload(message) for message in messages])


def serialize_agent_request(request: AgentRequest) -> str:
    """Serialize the complete model-visible request without runtime controls."""

    payload = _request_envelope_payload(request)
    payload["messages"] = [_message_payload(message) for message in request.messages]
    return _stable_json(payload)


@dataclass(frozen=True)
class UsageAnchor:
    """Exact provider input usage tied to a stable request prefix."""

    context_input_tokens: int
    envelope_fingerprint: str
    message_fingerprints: tuple[str, ...]

    @classmethod
    def from_request(cls, request: AgentRequest, *, context_input_tokens: int) -> UsageAnchor:
        if context_input_tokens < 0:
            raise ValueError("context_input_tokens cannot be negative.")
        return cls(
            context_input_tokens=context_input_tokens,
            envelope_fingerprint=_fingerprint(_stable_json(_request_envelope_payload(request))),
            message_fingerprints=tuple(_message_fingerprint(message) for message in request.messages),
        )


def estimate_request_tokens(request: AgentRequest, *, anchor: UsageAnchor | None = None) -> int:
    """Estimate a request, reusing exact usage only for an unchanged prefix."""

    if anchor is not None:
        envelope = _fingerprint(_stable_json(_request_envelope_payload(request)))
        current_messages = tuple(_message_fingerprint(message) for message in request.messages)
        anchored_count = len(anchor.message_fingerprints)
        if envelope == anchor.envelope_fingerprint and current_messages[:anchored_count] == anchor.message_fingerprints:
            appended = request.messages[anchored_count:]
            if not appended:
                return anchor.context_input_tokens
            return anchor.context_input_tokens + approximate_token_count(serialize_agent_messages(appended))
    return approximate_token_count(serialize_agent_request(request))


def _request_envelope_payload(request: AgentRequest) -> dict[str, object]:
    return {
        "system_prompt": request.system_prompt,
        "system_supplement": request.system_supplement,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in request.tools
        ],
    }


def _message_payload(message: AgentMessage) -> dict[str, object]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": dict(call.arguments)} for call in message.tool_calls
        ],
        "tool_results": [
            {
                "call_id": result.call_id,
                "tool_name": result.tool_name,
                "success": result.success,
                "output": result.output,
                "summary": result.summary,
                "duration_seconds": result.duration_seconds,
                "metadata": dict(result.metadata) if result.metadata is not None else None,
            }
            for result in message.tool_results
        ],
        "provider_state": (
            {
                "protocol": message.provider_state.protocol,
                "thinking_blocks": [
                    dict(block) for block in message.provider_state.thinking_blocks
                ],
            }
            if message.provider_state is not None
            else None
        ),
    }


def _message_fingerprint(message: AgentMessage) -> str:
    return _fingerprint(_stable_json(_message_payload(message)))


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class ContextGroup:
    """An indivisible, ordered section of persisted conversation history."""

    events: tuple[TimelineEvent, ...]
    start_sequence: int
    end_sequence: int
    estimated_tokens: int
    user_sequences: tuple[int, ...]


@dataclass(frozen=True)
class ToolResultOffload:
    """A persisted tool result selected for replacement in active context."""

    group_start_sequence: int
    event_sequence: int
    call_id: str | None
    tool_name: str
    success: bool
    output: str
    original_tokens: int
    preview_budget_tokens: int
    reason: str


@dataclass(frozen=True)
class CompactionSelection:
    """Complete groups and verbatim older user events retained after compacting."""

    recent_groups: tuple[ContextGroup, ...]
    summary_groups: tuple[ContextGroup, ...]
    preserved_user_events: tuple[TimelineEvent, ...]
    recent_tokens: int
    preserved_user_tokens: int

    @property
    def through_sequence(self) -> int:
        return self.summary_groups[-1].end_sequence if self.summary_groups else 0


def group_context_events(events: Sequence[TimelineEvent]) -> list[ContextGroup]:
    """Group ordinary messages and contiguous tool segments without reordering."""

    groups: list[ContextGroup] = []
    index = 0
    while index < len(events):
        grouped = [events[index]]
        first_kind = events[index].kind
        index += 1
        if first_kind == "assistant":
            while index < len(events) and events[index].kind in {"tool_call", "tool_result"}:
                grouped.append(events[index])
                index += 1
        elif first_kind in {"tool_call", "tool_result"}:
            while index < len(events) and events[index].kind in {"tool_call", "tool_result"}:
                grouped.append(events[index])
                index += 1
        groups.append(_context_group(grouped))
    return groups


def plan_tool_result_offloads(
    groups: Sequence[ContextGroup],
    *,
    policy: ContextPolicy | None = None,
) -> list[ToolResultOffload]:
    """Select oversized tool results, then enforce the per-round active cap."""

    active_policy = policy or ContextPolicy()
    planned: list[ToolResultOffload] = []
    for group in groups:
        results = [_offload_candidate(group, event, active_policy) for event in group.events if event.kind == "tool_result"]
        if not results:
            continue
        selected: dict[int, ToolResultOffload] = {}
        for candidate in sorted(results, key=lambda item: (-item.original_tokens, item.event_sequence)):
            if candidate.original_tokens > active_policy.single_tool_result_tokens:
                chosen = _with_offload_reason(candidate, "single_result")
                selected[candidate.event_sequence] = chosen
                planned.append(chosen)

        active_total = sum(
            selected[result.event_sequence].preview_budget_tokens
            if result.event_sequence in selected
            else result.original_tokens
            for result in results
        )
        if active_total <= active_policy.tool_round_tokens:
            continue
        remaining = sorted(
            (result for result in results if result.event_sequence not in selected),
            key=lambda item: (-item.original_tokens, item.event_sequence),
        )
        for candidate in remaining:
            chosen = _with_offload_reason(candidate, "round_total")
            selected[candidate.event_sequence] = chosen
            planned.append(chosen)
            active_total -= candidate.original_tokens - candidate.preview_budget_tokens
            if active_total <= active_policy.tool_round_tokens:
                break
    return planned


def select_compaction_history(
    groups: Sequence[ContextGroup],
    *,
    retained_token_budget: int,
    policy: ContextPolicy | None = None,
) -> CompactionSelection:
    """Select recent complete groups and verbatim older user messages."""

    active_policy = policy or ContextPolicy()
    budget = max(0, retained_token_budget)
    recent_reversed: list[ContextGroup] = []
    recent_tokens = 0
    for group in reversed(groups):
        recent_reversed.append(group)
        recent_tokens += group.estimated_tokens
        if (
            recent_tokens >= active_policy.recent_history_target_tokens
            and len(recent_reversed) >= active_policy.recent_history_min_groups
        ):
            break
    recent = list(reversed(recent_reversed))
    summary = list(groups[: len(groups) - len(recent)])

    preserved_newest_first: list[TimelineEvent] = []
    preserved_tokens = 0
    older_users = sorted(
        (event for group in summary for event in group.events if event.kind == "user"),
        key=lambda event: event.sequence,
        reverse=True,
    )
    for event in older_users:
        event_tokens = approximate_token_count(event.content)
        if preserved_tokens + event_tokens > active_policy.older_user_messages_target_tokens:
            continue
        preserved_newest_first.append(event)
        preserved_tokens += event_tokens
    preserved = sorted(preserved_newest_first, key=lambda event: event.sequence)

    while preserved and recent_tokens + preserved_tokens > budget:
        removed = preserved.pop(0)
        preserved_tokens -= approximate_token_count(removed.content)
    while recent and recent_tokens + preserved_tokens > budget:
        removed_group = recent.pop(0)
        recent_tokens -= removed_group.estimated_tokens
        summary.append(removed_group)

    return CompactionSelection(
        recent_groups=tuple(recent),
        summary_groups=tuple(summary),
        preserved_user_events=tuple(preserved),
        recent_tokens=recent_tokens,
        preserved_user_tokens=preserved_tokens,
    )


def build_tool_result_preview(
    output: str,
    *,
    original_tokens: int,
    success: bool,
    read_path: str,
    budget_tokens: int,
) -> str:
    """Build a bounded model-visible preview with balanced head and tail."""

    if original_tokens < 0 or budget_tokens < 1:
        raise ValueError("Token counts must be non-negative and the preview budget must be positive.")
    original_bytes = len(output.encode("utf-8"))
    metadata = (
        "[工具结果已外置]\n"
        f"状态：{'成功' if success else '失败'}\n"
        f"原始大小：{original_bytes} bytes，约 {original_tokens} tokens\n"
        f"完整结果：{read_path}\n"
    )
    separators = "--- 开头 ---\n\n--- 结尾 ---\n"
    fixed_tokens = approximate_token_count(metadata + separators)
    if fixed_tokens >= budget_tokens:
        raise ValueError("The preview budget is too small for required artifact metadata.")
    available_bytes = (budget_tokens - fixed_tokens) * 4
    head_bytes = available_bytes // 2
    tail_bytes = available_bytes - head_bytes
    while True:
        head = _utf8_prefix(output, head_bytes)
        tail = _utf8_suffix(output, tail_bytes)
        preview = f"{metadata}--- 开头 ---\n{head}\n--- 结尾 ---\n{tail}"
        excess = approximate_token_count(preview) - budget_tokens
        if excess <= 0:
            return preview
        reduction = max(1, excess * 2)
        head_bytes = max(0, head_bytes - reduction)
        tail_bytes = max(0, tail_bytes - reduction)


def _offload_candidate(
    group: ContextGroup,
    event: TimelineEvent,
    policy: ContextPolicy,
) -> ToolResultOffload:
    metadata = event.metadata or {}
    original_tokens = approximate_token_count(event.content)
    preview_budget = min(
        max(1, int(original_tokens * policy.tool_preview_ratio)),
        policy.tool_preview_max_tokens,
    )
    return ToolResultOffload(
        group_start_sequence=group.start_sequence,
        event_sequence=event.sequence,
        call_id=event.call_id,
        tool_name=metadata.get("tool_name") if isinstance(metadata.get("tool_name"), str) else "unknown_tool",
        success=metadata.get("success") if isinstance(metadata.get("success"), bool) else False,
        output=event.content,
        original_tokens=original_tokens,
        preview_budget_tokens=preview_budget,
        reason="",
    )


def _with_offload_reason(candidate: ToolResultOffload, reason: str) -> ToolResultOffload:
    return ToolResultOffload(
        group_start_sequence=candidate.group_start_sequence,
        event_sequence=candidate.event_sequence,
        call_id=candidate.call_id,
        tool_name=candidate.tool_name,
        success=candidate.success,
        output=candidate.output,
        original_tokens=candidate.original_tokens,
        preview_budget_tokens=candidate.preview_budget_tokens,
        reason=reason,
    )


def _utf8_prefix(value: str, byte_limit: int) -> str:
    return value.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")


def _utf8_suffix(value: str, byte_limit: int) -> str:
    if byte_limit <= 0:
        return ""
    return value.encode("utf-8")[-byte_limit:].decode("utf-8", errors="ignore")


def _context_group(events: list[TimelineEvent]) -> ContextGroup:
    native_messages = messages_from_events(events)
    if native_messages:
        estimated_tokens = approximate_token_count(serialize_agent_messages(native_messages))
    else:
        estimated_tokens = sum(approximate_token_count(event.content) for event in events)
    return ContextGroup(
        events=tuple(events),
        start_sequence=events[0].sequence,
        end_sequence=events[-1].sequence,
        estimated_tokens=estimated_tokens,
        user_sequences=tuple(event.sequence for event in events if event.kind == "user"),
    )


@dataclass(frozen=True)
class ContextPlan:
    """Messages ready for a model request and older events needing a summary."""

    messages: list[AgentMessage]
    events_for_summary: list[TimelineEvent]


class ContextBuilder:
    """Build a bounded request context without separating tool calls from results."""

    def __init__(self, *, max_characters: int) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive.")
        self.max_characters = max_characters

    def build(self, events: list[TimelineEvent]) -> ContextPlan:
        latest_summary = _latest_summary(events)
        covered_through = _covered_through(latest_summary)
        candidates = [
            event
            for event in events
            if event.sequence > covered_through and event.kind in {"user", "assistant", "tool_call", "tool_result"}
        ]
        groups = _event_groups(candidates)
        if _character_count(groups) <= self.max_characters:
            return ContextPlan(_messages(latest_summary, candidates), [])

        recent: list[list[TimelineEvent]] = []
        budget = max(1, self.max_characters // 2)
        used = 0
        for group in reversed(groups):
            size = _character_count((group,))
            if recent and used + size > budget:
                break
            recent.append(group)
            used += size
        recent.reverse()
        retained = [event for group in recent for event in group]
        summary_groups = groups[: len(groups) - len(recent)]
        return ContextPlan(_messages(latest_summary, retained), [event for group in summary_groups for event in group])


def messages_from_events(events: list[TimelineEvent]) -> list[AgentMessage]:
    """Recreate provider-native messages from the persisted timeline."""
    messages: list[AgentMessage] = []
    pending_results: list[ToolResult] = []

    def flush_results() -> None:
        if pending_results:
            messages.append(AgentMessage("user", tool_results=tuple(pending_results)))
            pending_results.clear()

    for event in events:
        if event.kind == "user":
            flush_results()
            messages.append(AgentMessage("user", event.content))
        elif event.kind == "assistant":
            flush_results()
            messages.append(
                AgentMessage(
                    "assistant",
                    event.content,
                    _tool_calls(event.metadata),
                    provider_state=_provider_state(event.metadata),
                )
            )
        elif event.kind == "tool_result":
            result = _tool_result(event)
            if result is not None:
                pending_results.append(result)
    flush_results()
    return messages


def _latest_summary(events: list[TimelineEvent]) -> TimelineEvent | None:
    summaries = [event for event in events if event.kind == "summary"]
    return summaries[-1] if summaries else None


def _covered_through(summary: TimelineEvent | None) -> int:
    if summary is None or summary.metadata is None:
        return 0
    value = summary.metadata.get("through_sequence")
    return value if isinstance(value, int) and value >= 0 else 0


def _event_groups(events: list[TimelineEvent]) -> list[list[TimelineEvent]]:
    return [list(group.events) for group in group_context_events(events)]


def _messages(summary: TimelineEvent | None, events: list[TimelineEvent]) -> list[AgentMessage]:
    messages = [AgentMessage("assistant", summary.content)] if summary is not None else []
    messages.extend(messages_from_events(events))
    return messages


def _tool_calls(metadata: Mapping[str, object] | None) -> tuple[ToolCall, ...]:
    if metadata is None or not isinstance(metadata.get("tool_calls"), list):
        return ()
    calls: list[ToolCall] = []
    for item in metadata["tool_calls"]:
        if not isinstance(item, Mapping):
            continue
        call_id, name, arguments = item.get("id"), item.get("name"), item.get("arguments")
        if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, Mapping):
            calls.append(ToolCall(call_id, name, dict(arguments)))
    return tuple(calls)


def _provider_state(metadata: Mapping[str, object] | None) -> ProviderMessageState | None:
    if metadata is None:
        return None
    raw_state = metadata.get("provider_state")
    if not isinstance(raw_state, Mapping):
        return None
    protocol = raw_state.get("protocol")
    raw_blocks = raw_state.get("thinking_blocks")
    if protocol not in {"anthropic", "openai"} or not isinstance(raw_blocks, list):
        return None
    blocks: list[dict[str, object]] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping) or raw_block.get("type") not in {
            "thinking",
            "redacted_thinking",
        }:
            return None
        blocks.append(dict(raw_block))
    return ProviderMessageState(protocol, tuple(blocks))


def _tool_result(event: TimelineEvent) -> ToolResult | None:
    metadata = event.metadata or {}
    tool_name = metadata.get("tool_name")
    success = metadata.get("success")
    summary = metadata.get("summary")
    if not isinstance(event.call_id, str) or not isinstance(tool_name, str) or not isinstance(success, bool):
        return None
    return ToolResult(event.call_id, tool_name, success, event.content, summary if isinstance(summary, str) else "restored tool result")


def _character_count(groups: tuple[list[TimelineEvent], ...] | list[list[TimelineEvent]]) -> int:
    return sum(len(event.content) for group in groups for event in group)


def summarize_events(events: list[TimelineEvent], *, max_characters: int = 4_000) -> str:
    """Create a bounded local fallback summary without deleting original events."""
    lines: list[str] = []
    used = 0
    for event in events:
        excerpt = " ".join(event.content.split())
        line = f"{event.kind}: {excerpt}"
        remaining = max_characters - used
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[:remaining] + "…"
        lines.append(line)
        used += len(line) + 1
    return "此前对话摘要：\n" + "\n".join(lines)
