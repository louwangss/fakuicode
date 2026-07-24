"""Session-scoped preparation of bounded, recoverable provider context."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Callable, Literal, Mapping, Sequence

from fakuicode.context import (
    ContextPolicy,
    COMPACTION_BOUNDARY_MESSAGE,
    SUMMARY_HEADINGS,
    ToolResultOffload,
    UsageAnchor,
    build_tool_result_preview,
    estimate_request_tokens,
    group_context_events,
    messages_from_events,
    normalize_structured_summary,
    plan_tool_result_offloads,
    select_compaction_history,
)
from fakuicode.context_artifacts import ContextArtifactRef, ContextArtifactStore
from fakuicode.errors import ProviderError, RequestCancelled
from fakuicode.models import AgentMessage, ContextStatus, TimelineEvent, TokenUsage
from fakuicode.providers.base import AgentProvider, AgentRequest
from fakuicode.providers.invocation import invoke_provider_stream
from fakuicode.storage import ConversationStore


class ContextManagementError(RuntimeError):
    """Raised when a request cannot be prepared without risking context loss."""


ContextTrigger = Literal["automatic", "manual", "emergency"]


class ContextLimitError(ContextManagementError):
    def __init__(self, status: ContextStatus) -> None:
        super().__init__(status.recovery_hint)
        self.status = status


class SummaryGenerationError(ContextManagementError):
    def __init__(self, message: str, *, error_category: str = "invalid_summary") -> None:
        super().__init__(message)
        self.error_category = error_category


class ContextRecoveryError(ContextManagementError):
    def __init__(self, status: ContextStatus) -> None:
        super().__init__(status.recovery_hint)
        self.status = status


@dataclass(frozen=True)
class ContextBudgetAssessment:
    estimated_tokens: int
    automatic_trigger_tokens: int
    hard_input_limit_tokens: int
    automatic_compaction_required: bool
    safe_to_send: bool


_SUMMARY_SYSTEM_PROMPT = (
    "你是 fakuiCode 的内部上下文摘要器。不得调用任何工具，也不得建议由本次摘要调用工具。"
    "你的响应只会作为内部活动上下文，不要与用户对话。"
)


@dataclass(frozen=True)
class ContextPreparationResult:
    """One model-ready request plus non-content effects of preparing it."""

    request: AgentRequest
    artifacts: tuple[ContextArtifactRef, ...] = ()
    status: ContextStatus | None = None
    structure_changed: bool = False
    anchor_invalidated: bool = False


class ContextManager:
    """Own runtime context state for exactly one active conversation."""

    def __init__(
        self,
        provider: AgentProvider,
        *,
        workspace: Path,
        context_window: int,
        store: ConversationStore | None = None,
        conversation_id: str | None = None,
        policy: ContextPolicy | None = None,
        lifecycle_callback: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        if context_window < 1:
            raise ValueError("context_window must be positive.")
        if (store is None) != (conversation_id is None):
            raise ValueError("store and conversation_id must be provided together.")
        self.provider = provider
        self.workspace = workspace.resolve(strict=True)
        self.context_window = context_window
        self.store = store
        self.conversation_id = conversation_id
        self.policy = policy or ContextPolicy()
        self.lifecycle_callback = lifecycle_callback
        self.artifact_store = (
            ContextArtifactStore(self.workspace, conversation_id)
            if conversation_id is not None
            else None
        )
        self.artifact_cleanup_failed = False
        if self.artifact_store is not None:
            assert store is not None
            try:
                retained = {
                    record.id
                    for record in store.list_conversations()
                    if record.workspace.resolve(strict=False) == self.workspace
                }
                self.artifact_store.cleanup_stale_tombstones(
                    retained_conversation_ids=retained,
                )
            except OSError:
                self.artifact_cleanup_failed = True
        self._usage_anchor: UsageAnchor | None = None
        self._consecutive_summary_failures = 0
        self._reported_offloads: set[tuple[tuple[int, str], ...]] = set()
        self.diagnostic_write_failed = False

    @property
    def usage_anchor(self) -> UsageAnchor | None:
        return self._usage_anchor

    @property
    def consecutive_summary_failures(self) -> int:
        return self._consecutive_summary_failures

    @property
    def automatic_compaction_disabled(self) -> bool:
        return self._consecutive_summary_failures >= self.policy.summary_failure_limit

    def active_messages(
        self,
        fallback: Sequence[AgentMessage] = (),
    ) -> tuple[AgentMessage, ...]:
        """Rebuild only model-visible active history, never mutate its source timeline."""

        if self.store is None or self.conversation_id is None:
            return tuple(fallback)
        return tuple(messages_from_events(self._active_events()))

    def _active_events(self):
        assert self.store is not None and self.conversation_id is not None
        boundary = self.store.latest_clear_sequence(self.conversation_id)
        summary = self.store.load_latest_context_summary(self.conversation_id)
        if summary is not None:
            metadata = summary.metadata or {}
            through_sequence = metadata.get("through_sequence")
            raw_preserved = metadata.get("preserved_user_sequences")
            if isinstance(through_sequence, int) and isinstance(raw_preserved, list):
                preserved = tuple(
                    sequence
                    for sequence in raw_preserved
                    if isinstance(sequence, int)
                    and not isinstance(sequence, bool)
                    and boundary < sequence <= through_sequence
                )
                events = [
                    *self.store.load_events_by_sequences(
                        self.conversation_id,
                        preserved,
                    ),
                    *self.store.load_events(
                        self.conversation_id,
                        after_sequence=max(boundary, through_sequence),
                    ),
                ]
                return sorted(
                    (
                        event
                        for event in events
                        if event.kind in {"user", "assistant", "tool_call", "tool_result", "agent_result"}
                    ),
                    key=lambda event: event.sequence,
                )
        return [
            event
            for event in self.store.load_events(
                self.conversation_id,
                after_sequence=boundary,
            )
            if event.kind in {"user", "assistant", "tool_call", "tool_result", "agent_result"}
        ]

    def prepare_light(self, request: AgentRequest) -> ContextPreparationResult:
        """Offload selected complete tool results before any heavyweight decision."""

        result, _ = self._prepare_light_with_events(request, trigger="automatic")
        return result

    def _prepare_light_with_events(
        self,
        request: AgentRequest,
        *,
        trigger: ContextTrigger,
    ) -> tuple[ContextPreparationResult, list[TimelineEvent]]:

        started = monotonic()
        if self.store is None or self.conversation_id is None or self.artifact_store is None:
            return ContextPreparationResult(request), []
        events = self._active_events()
        offloads = plan_tool_result_offloads(group_context_events(events), policy=self.policy)
        if not offloads:
            rebuilt = tuple(messages_from_events(events))
            return (
                ContextPreparationResult(
                    replace(request, messages=rebuilt) if rebuilt != request.messages else request
                ),
                events,
            )

        replacements: dict[int, str] = {}
        artifacts: list[ContextArtifactRef] = []
        try:
            for offload in offloads:
                reference = self.artifact_store.write_tool_result(
                    source_sequence=offload.event_sequence,
                    output=offload.output,
                    success=offload.success,
                    provider_call_id=offload.call_id,
                )
                preview = build_tool_result_preview(
                    offload.output,
                    original_tokens=offload.original_tokens,
                    success=offload.success,
                    read_path=reference.read_path,
                    budget_tokens=offload.preview_budget_tokens,
                )
                artifacts.append(reference)
                replacements[offload.event_sequence] = preview
        except (OSError, ValueError) as error:
            self._append_diagnostic(
                trigger=trigger,
                result="blocked",
                estimated_before=sum(item.original_tokens for item in offloads),
                threshold=_offload_threshold(offloads, self.policy),
                artifacts=artifacts,
                duration_seconds=monotonic() - started,
                error_category="artifact_write",
            )
            raise ContextManagementError(
                "Context artifact could not be stored safely; the provider request was blocked."
            ) from error

        prepared_events = [
            replace(event, content=replacements[event.sequence])
            if event.sequence in replacements
            else event
            for event in events
        ]
        prepared_request = replace(
            request,
            messages=tuple(messages_from_events(prepared_events)),
        )
        self.invalidate_anchor()
        offload_fingerprint = tuple(
            (artifact.source_sequence, artifact.content_sha256) for artifact in artifacts
        )
        if (
            any(artifact.newly_created for artifact in artifacts)
            and offload_fingerprint not in self._reported_offloads
        ):
            if self._append_diagnostic(
                trigger=trigger,
                result="offloaded",
                estimated_before=sum(item.original_tokens for item in offloads),
                estimated_after=sum(item.preview_budget_tokens for item in offloads),
                threshold=_offload_threshold(offloads, self.policy),
                artifacts=artifacts,
                duration_seconds=monotonic() - started,
            ):
                self._reported_offloads.add(offload_fingerprint)
        return (
            ContextPreparationResult(
                prepared_request,
                artifacts=tuple(artifacts),
                structure_changed=True,
                anchor_invalidated=True,
            ),
            prepared_events,
        )

    def activate_request(self, request: AgentRequest) -> AgentRequest:
        """Apply restored active messages and the latest single summary boundary."""

        if self.store is None or self.conversation_id is None:
            return request
        rebuilt = replace(request, messages=self.active_messages(request.messages))
        return self._add_latest_summary(rebuilt)

    def _add_latest_summary(self, request: AgentRequest) -> AgentRequest:
        if self.store is None or self.conversation_id is None:
            return request
        summary = self.store.load_latest_context_summary(self.conversation_id)
        if summary is None:
            return request
        return replace(
            request,
            system_supplement=_summary_supplement(request.system_supplement, summary.content),
        )

    def compact_request(
        self,
        request: AgentRequest,
        *,
        trigger: ContextTrigger,
        _light: ContextPreparationResult | None = None,
        _prepared_events: list[TimelineEvent] | None = None,
        _started: float | None = None,
    ) -> ContextPreparationResult:
        self._notify_lifecycle("pre_compact", {"compact": {"trigger": trigger, "outcome": "started"}})
        try:
            result = self._compact_request(
                request,
                trigger=trigger,
                _light=_light,
                _prepared_events=_prepared_events,
                _started=_started,
            )
        except Exception:
            self._notify_lifecycle(
                "post_compact", {"compact": {"trigger": trigger, "outcome": "failed"}}
            )
            raise
        outcome = result.status.result if result.status is not None else "completed"
        self._notify_lifecycle(
            "post_compact", {"compact": {"trigger": trigger, "outcome": outcome}}
        )
        return result

    def _compact_request(
        self,
        request: AgentRequest,
        *,
        trigger: ContextTrigger,
        _light: ContextPreparationResult | None = None,
        _prepared_events: list[TimelineEvent] | None = None,
        _started: float | None = None,
    ) -> ContextPreparationResult:
        """Run light preparation, replace older groups with one rolling summary, and persist it."""

        started = monotonic() if _started is None else _started
        if _light is None or _prepared_events is None:
            light, prepared_events = self._prepare_light_with_events(request, trigger=trigger)
        else:
            light, prepared_events = _light, _prepared_events
        estimated_before = self.assess_request(light.request, use_anchor=False).estimated_tokens
        if self.store is None or self.conversation_id is None or not prepared_events:
            activated = self._add_latest_summary(light.request)
            self.ensure_hard_limit(activated, trigger=trigger, use_anchor=False)
            status = ContextStatus(
                trigger=trigger,
                result="noop",
                estimated_before=estimated_before,
                estimated_after=estimated_before,
                artifact_count=len(light.artifacts),
                artifact_bytes=sum(item.byte_size for item in light.artifacts),
                duration_seconds=monotonic() - started,
            )
            self._append_diagnostic(
                trigger=trigger,
                result="noop",
                estimated_before=estimated_before,
                estimated_after=estimated_before,
                artifacts=light.artifacts,
                duration_seconds=status.duration_seconds,
            )
            return replace(
                light,
                request=activated,
                status=status,
            )

        empty_request = replace(light.request, messages=())
        retained_budget = max(
            0,
            self.policy.hard_input_limit_tokens(self.context_window)
            - estimate_request_tokens(empty_request)
            - self.policy.summary_hard_max_tokens,
        )
        selection = select_compaction_history(
            group_context_events(prepared_events),
            retained_token_budget=retained_budget,
            policy=self.policy,
        )
        if not selection.summary_groups:
            activated = self._add_latest_summary(light.request)
            estimated_after = self.ensure_hard_limit(
                activated,
                trigger=trigger,
                use_anchor=False,
            )
            status = ContextStatus(
                trigger=trigger,
                result="noop",
                estimated_before=estimated_before,
                estimated_after=estimated_after,
                artifact_count=len(light.artifacts),
                artifact_bytes=sum(item.byte_size for item in light.artifacts),
                duration_seconds=monotonic() - started,
            )
            self._append_diagnostic(
                trigger=trigger,
                result="noop",
                estimated_before=estimated_before,
                estimated_after=estimated_after,
                artifacts=light.artifacts,
                duration_seconds=status.duration_seconds,
            )
            return replace(
                light,
                request=activated,
                status=status,
            )

        previous_summary = self.store.load_latest_context_summary(self.conversation_id)
        summary_events = (
            ([previous_summary] if previous_summary is not None else [])
            + [event for group in selection.summary_groups for event in group.events]
        )
        summary = self.generate_summary(
            summary_events,
            trigger=trigger,
            cancel_event=request.cancel_event,
        )
        retained_events = sorted(
            [
                *selection.preserved_user_events,
                *(event for group in selection.recent_groups for event in group.events),
            ],
            key=lambda event: event.sequence,
        )
        compacted_request = replace(
            light.request,
            messages=tuple(messages_from_events(retained_events)),
            system_supplement=_summary_supplement(light.request.system_supplement, summary),
        )
        estimated_after = self.ensure_hard_limit(
            compacted_request,
            trigger=trigger,
            use_anchor=False,
        )
        previous_through = 0
        if previous_summary is not None and previous_summary.metadata is not None:
            value = previous_summary.metadata.get("through_sequence")
            previous_through = value if isinstance(value, int) else 0
        through_sequence = max(previous_through, selection.through_sequence)
        preserved_user_sequences = tuple(
            event.sequence
            for event in retained_events
            if event.kind == "user" and event.sequence <= through_sequence
        )
        self.store.append_context_summary(
            self.conversation_id,
            summary,
            through_sequence=through_sequence,
            preserved_user_sequences=preserved_user_sequences,
            trigger=trigger,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            format_version=1,
        )
        self.invalidate_anchor()
        self.record_summary_success()
        duration_seconds = monotonic() - started
        self._append_diagnostic(
            trigger=trigger,
            result="compacted",
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            threshold=(
                self.policy.automatic_trigger_tokens(self.context_window)
                if trigger == "automatic"
                else None
            ),
            artifacts=light.artifacts,
            duration_seconds=duration_seconds,
        )
        return ContextPreparationResult(
            compacted_request,
            artifacts=light.artifacts,
            status=ContextStatus(
                trigger=trigger,
                result="compacted",
                estimated_before=estimated_before,
                estimated_after=estimated_after,
                artifact_count=len(light.artifacts),
                artifact_bytes=sum(item.byte_size for item in light.artifacts),
                duration_seconds=duration_seconds,
            ),
            structure_changed=True,
            anchor_invalidated=True,
        )

    def _notify_lifecycle(self, event: str, payload: Mapping[str, object]) -> None:
        if self.lifecycle_callback is None:
            return
        try:
            self.lifecycle_callback(event, payload)
        except Exception:
            pass

    def prepare_request(self, request: AgentRequest) -> ContextPreparationResult:
        """Prepare one ordinary request in the fixed light → automatic → hard-check order."""

        started = monotonic()
        light, prepared_events = self._prepare_light_with_events(request, trigger="automatic")
        activated = self._add_latest_summary(light.request)
        assessment = self.assess_request(
            activated,
            use_anchor=not light.anchor_invalidated,
        )
        if not assessment.automatic_compaction_required:
            self.ensure_hard_limit(
                activated,
                trigger="automatic",
                use_anchor=not light.anchor_invalidated,
            )
            return replace(light, request=activated)
        if self.automatic_compaction_disabled:
            estimated_after = self.ensure_hard_limit(
                activated,
                trigger="automatic",
                use_anchor=not light.anchor_invalidated,
            )
            status = ContextStatus(
                trigger="automatic",
                result="breaker",
                estimated_before=assessment.estimated_tokens,
                estimated_after=estimated_after,
                artifact_count=len(light.artifacts),
                artifact_bytes=sum(item.byte_size for item in light.artifacts),
                duration_seconds=monotonic() - started,
                consecutive_failures=self._consecutive_summary_failures,
                recovery_hint="Automatic compaction is disabled; use /compact or /clear.",
            )
            self._append_diagnostic(
                trigger="automatic",
                result="breaker",
                estimated_before=assessment.estimated_tokens,
                estimated_after=estimated_after,
                threshold=assessment.automatic_trigger_tokens,
                artifacts=light.artifacts,
                duration_seconds=status.duration_seconds,
            )
            return replace(
                light,
                request=activated,
                status=status,
            )
        try:
            return self.compact_request(
                request,
                trigger="automatic",
                _light=light,
                _prepared_events=prepared_events,
                _started=started,
            )
        except RequestCancelled:
            raise
        except (SummaryGenerationError, ContextLimitError) as error:
            return self._failed_compaction(light, "automatic", error, started)

    def compact_manually(self, request: AgentRequest) -> ContextPreparationResult:
        """Bypass the automatic breaker for exactly one user-requested attempt."""

        started = monotonic()
        light, prepared_events = self._prepare_light_with_events(request, trigger="manual")
        try:
            return self.compact_request(
                request,
                trigger="manual",
                _light=light,
                _prepared_events=prepared_events,
                _started=started,
            )
        except RequestCancelled:
            raise
        except (SummaryGenerationError, ContextLimitError) as error:
            return self._failed_compaction(light, "manual", error, started)

    def recover_overflow(self, request: AgentRequest) -> ContextPreparationResult:
        """Bypass the breaker once for a provider-confirmed overflow recovery."""

        started = monotonic()
        light, prepared_events = self._prepare_light_with_events(request, trigger="emergency")
        try:
            return self.compact_request(
                request,
                trigger="emergency",
                _light=light,
                _prepared_events=prepared_events,
                _started=started,
            )
        except RequestCancelled:
            raise
        except (SummaryGenerationError, ContextLimitError) as error:
            self.record_summary_failure()
            error_category = (
                error.error_category
                if isinstance(error, SummaryGenerationError)
                else "hard_limit"
            )
            status = ContextStatus(
                trigger="emergency",
                result="failed",
                estimated_before=self.assess_request(
                    self._add_latest_summary(light.request),
                    use_anchor=False,
                ).estimated_tokens,
                artifact_count=len(light.artifacts),
                artifact_bytes=sum(item.byte_size for item in light.artifacts),
                duration_seconds=monotonic() - started,
                consecutive_failures=self._consecutive_summary_failures,
                error_category=error_category,
                recovery_hint="Overflow recovery failed; use /compact or /clear.",
            )
            self._append_diagnostic(
                trigger="emergency",
                result="failed",
                estimated_before=status.estimated_before,
                artifacts=light.artifacts,
                duration_seconds=status.duration_seconds,
                error_category=error_category,
            )
            raise ContextRecoveryError(
                status
            ) from error

    def _failed_compaction(
        self,
        light: ContextPreparationResult,
        trigger: ContextTrigger,
        error: SummaryGenerationError | ContextLimitError,
        started: float,
    ) -> ContextPreparationResult:
        self.record_summary_failure()
        activated = self._add_latest_summary(light.request)
        estimated_before = self.assess_request(activated, use_anchor=False).estimated_tokens
        assessment = self.assess_request(activated, use_anchor=False)
        estimated_after = assessment.estimated_tokens
        error_category = (
            error.error_category
            if isinstance(error, SummaryGenerationError)
            else "hard_limit"
        )
        breaker_note = (
            " Automatic compaction is now disabled."
            if self.automatic_compaction_disabled
            else ""
        )
        status = ContextStatus(
            trigger=trigger,
            result="failed",
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            artifact_count=len(light.artifacts),
            artifact_bytes=sum(item.byte_size for item in light.artifacts),
            duration_seconds=monotonic() - started,
            consecutive_failures=self._consecutive_summary_failures,
            error_category=error_category,
            recovery_hint=f"Compaction failed.{breaker_note} Use /compact or /clear.",
        )
        self._append_diagnostic(
            trigger=trigger,
            result="failed",
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            artifacts=light.artifacts,
            duration_seconds=status.duration_seconds,
            error_category=error_category,
        )
        if trigger != "manual" and not assessment.safe_to_send:
            self.ensure_hard_limit(
                activated,
                trigger=trigger,
                use_anchor=False,
            )
        return replace(
            light,
            request=activated,
            status=status,
        )

    def assess_request(
        self,
        request: AgentRequest,
        *,
        use_anchor: bool = True,
    ) -> ContextBudgetAssessment:
        estimated = estimate_request_tokens(
            request,
            anchor=self._usage_anchor if use_anchor else None,
        )
        automatic_trigger = self.policy.automatic_trigger_tokens(self.context_window)
        hard_limit = self.policy.hard_input_limit_tokens(self.context_window)
        return ContextBudgetAssessment(
            estimated_tokens=estimated,
            automatic_trigger_tokens=automatic_trigger,
            hard_input_limit_tokens=hard_limit,
            automatic_compaction_required=estimated >= automatic_trigger,
            safe_to_send=estimated <= hard_limit,
        )

    def ensure_hard_limit(
        self,
        request: AgentRequest,
        *,
        trigger: ContextTrigger,
        use_anchor: bool = False,
    ) -> int:
        assessment = self.assess_request(request, use_anchor=use_anchor)
        if assessment.safe_to_send:
            return assessment.estimated_tokens
        status = ContextStatus(
            trigger=trigger,
            result="blocked",
            estimated_before=assessment.estimated_tokens,
            estimated_after=None,
            consecutive_failures=self._consecutive_summary_failures,
            error_category="hard_limit",
            recovery_hint=(
                "Context cannot be sent safely. Use /compact to retry compaction "
                "or /clear to start a fresh active context."
            ),
        )
        self._append_diagnostic(
            trigger=trigger,
            result="blocked",
            estimated_before=assessment.estimated_tokens,
            threshold=assessment.hard_input_limit_tokens,
            error_category="hard_limit",
        )
        raise ContextLimitError(status)

    def generate_summary(
        self,
        events: Sequence[TimelineEvent],
        *,
        trigger: ContextTrigger,
        cancel_event: Event | None = None,
    ) -> str:
        """Run one dedicated, tool-free summary request without entering preparation."""

        headings = "\n".join(f"## {heading}" for heading in SUMMARY_HEADINGS)
        source = json.dumps(
            [
                {
                    "sequence": event.sequence,
                    "kind": event.kind,
                    "content": event.content,
                    "call_id": event.call_id,
                    "metadata": event.metadata,
                }
                for event in events
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        prompt = (
            "请压缩下面的较早活动历史，供同一任务后续继续执行。\n"
            "不得调用任何工具。先在内部组织分析草稿，核对目标、约束、决策、证据、"
            "文件状态、失败尝试和下一步；内部草稿用完即丢弃，绝不能输出。\n"
            "只输出正式摘要，不要前言、解释、草稿、代码围栏或额外标题。"
            f"正式摘要目标约 {self.policy.summary_target_tokens} tokens，硬上限 "
            f"{self.policy.summary_hard_max_tokens} tokens。\n"
            "必须严格按以下八个 Markdown 二级标题的顺序输出；空章节写“无”：\n"
            f"{headings}\n\n"
            "待摘要的模型可见历史（稳定 JSON）：\n"
            f"{source}"
        )
        request = AgentRequest(
            (AgentMessage("user", prompt),),
            (),
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            cancel_event=cancel_event,
            output_token_limit=self.policy.summary_hard_max_tokens,
        )
        self.ensure_hard_limit(request, trigger=trigger, use_anchor=False)
        parts: list[str] = []
        completed = False
        emitted_tool_call = False
        try:
            for event in invoke_provider_stream(self.provider, request, preserve_tool_tuple=True):
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled()
                if event.kind == "text_delta":
                    parts.append(event.text)
                elif event.kind == "tool_call":
                    emitted_tool_call = True
                elif event.kind == "completed":
                    completed = True
                elif event.kind == "cancelled":
                    raise RequestCancelled()
                elif event.kind == "error":
                    raise SummaryGenerationError("Summary provider stream failed.", error_category="provider")
        except RequestCancelled:
            raise
        except SummaryGenerationError:
            raise
        except ProviderError as error:
            raise SummaryGenerationError(
                "Summary provider request failed.",
                error_category=error.category,
            ) from error
        except Exception as error:
            raise SummaryGenerationError(
                "Summary provider request failed.",
                error_category="provider",
            ) from error
        if emitted_tool_call:
            raise SummaryGenerationError("Summary response attempted a tool call.")
        if not completed:
            raise SummaryGenerationError("Summary provider stream ended before completion.")
        try:
            return normalize_structured_summary("".join(parts), policy=self.policy)
        except ValueError as error:
            raise SummaryGenerationError("Summary response failed structural validation.") from error

    def observe_usage(self, request: AgentRequest, usage: TokenUsage) -> None:
        """Anchor future estimates only to a provider-normalized successful input count."""

        if usage.context_input_tokens is None:
            return
        self._usage_anchor = UsageAnchor.from_request(
            request,
            context_input_tokens=usage.context_input_tokens,
        )

    def invalidate_anchor(self) -> None:
        self._usage_anchor = None

    def _append_diagnostic(
        self,
        *,
        trigger: ContextTrigger,
        result: str,
        estimated_before: int | None = None,
        estimated_after: int | None = None,
        threshold: int | None = None,
        artifacts: Sequence[ContextArtifactRef] = (),
        duration_seconds: float | None = None,
        error_category: str | None = None,
    ) -> bool:
        if self.store is None or self.conversation_id is None:
            return False
        metadata: dict[str, object] = {
            "trigger": trigger,
            "result": result,
            "artifact_count": len(artifacts),
            "artifact_bytes": sum(item.byte_size for item in artifacts),
            "consecutive_failures": self._consecutive_summary_failures,
            "error_category": (
                error_category
                if error_category
                in {
                    "none",
                    "provider",
                    "context_overflow",
                    "invalid_summary",
                    "artifact_write",
                    "hard_limit",
                    "cancelled",
                    "other",
                }
                else "other" if error_category else "none"
            ),
        }
        for name, value in (
            ("estimated_before", estimated_before),
            ("estimated_after", estimated_after),
            ("threshold", threshold),
        ):
            if value is not None:
                metadata[name] = max(0, value)
        if duration_seconds is not None:
            metadata["duration_ms"] = max(0, round(duration_seconds * 1_000))
        try:
            self.store.append_context_diagnostic(self.conversation_id, metadata)
        except Exception:
            self.diagnostic_write_failed = True
            return False
        return True

    def record_summary_failure(self) -> None:
        self._consecutive_summary_failures += 1

    def record_summary_success(self) -> None:
        self._consecutive_summary_failures = 0

    def reset(self) -> None:
        """Reset ephemeral estimation and breaker state after `/clear`."""

        self._usage_anchor = None
        self._consecutive_summary_failures = 0
        self._reported_offloads.clear()


def _summary_supplement(existing: str, summary: str) -> str:
    context = (
        "<context-summary>\n"
        f"{summary}\n"
        "</context-summary>\n\n"
        f"<context-boundary>\n{COMPACTION_BOUNDARY_MESSAGE}\n</context-boundary>"
    )
    return f"{existing}\n\n{context}" if existing else context


def _offload_threshold(
    offloads: Sequence[ToolResultOffload],
    policy: ContextPolicy,
) -> int:
    if any(item.reason == "round_total" for item in offloads):
        return policy.tool_round_tokens
    return policy.single_tool_result_tokens
