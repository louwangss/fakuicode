"""Bounded, tool-free model maintenance for automatic memory."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import json
from threading import Condition, Thread

from fakuicode.errors import ProviderCapabilityError
from fakuicode.memory.content_policy import MemoryValidationError, parse_operation_batch, serialize_entry
from fakuicode.memory.models import (
    CompletedTurn,
    CreateEntry,
    MemoryEntry,
    MemoryLimits,
    MemoryOperationBatch,
    MemoryScopeRef,
    MemorySnapshot,
    MergeEntries,
    UpdateEntry,
    canonical_uuid,
)
from fakuicode.memory.repository import MemoryRepository, MemoryRepositoryError
from fakuicode.models import AgentMessage, AgentStreamEvent, ProviderConfig
from fakuicode.providers.base import AgentRequest
from fakuicode.providers.invocation import invoke_provider_stream


_OPERATION_CONTRACT = """operations 只能使用以下结构之一（不得输出 sources，来源由宿主绑定）：
- 无变化：{"kind":"noop"}
- 新增：{"kind":"create","scope":"user|project","category":"user_preference|correction|project_knowledge|reference","summary":"简短摘要","body":"正文","evidence":{"text":"用户原文中的逐字片段","intent":"cross_project|project_only"}}
- 更新：{"kind":"update","entry_id":"当前索引中的 UUID","summary":"简短摘要","body":"正文","evidence":{"text":"用户原文中的逐字片段","intent":"cross_project|project_only"}}
- 合并：{"kind":"merge","entry_ids":["至少两个当前 UUID"],"scope":"user|project","category":"user_preference|correction|project_knowledge|reference","summary":"简短摘要","body":"正文","evidence":{"text":"用户原文中的逐字片段","intent":"cross_project|project_only"}}
- 删除已被替代项：{"kind":"delete","entry_ids":["当前索引中的 UUID"]}

规则：
- user 作用域只允许用户明确表达的跨项目偏好或纠正；必须提供 evidence，text 必须逐字出现在 user_text 中，跨项目意图使用 cross_project。
- 明确跨项目证据示例：{"text":"以后所有项目都默认使用简体中文","intent":"cross_project"}。
- project 作用域用于当前项目知识、参考资料或仅限当前项目的偏好；需要证据时使用 project_only。
- project_knowledge 和 reference 只能使用 project 作用域。
- 没有值得长期保存的内容时，operations 必须且只能包含 {"kind":"noop"}。
- summary、body 和 evidence 不得包含密钥、令牌、密码、完整配置或其他敏感信息。
"""

_EXTRACTION_PROMPT = f"""你是 fakuiCode 的自动记忆维护器。
只根据给定的本轮最小输入和当前索引提出受控 JSON 操作。
不得调用工具、推断秘密、指定路径或把项目内容提升为用户级偏好。
只输出单个 JSON 对象，禁止 Markdown 代码围栏和解释文字。
顶层字段必须且仅能是 expected_user_digest、expected_project_digest、needs_details、operations。
expected_user_digest 和 expected_project_digest 必须原样复制输入中的同名摘要值。
needs_details 只能包含 current_index 中的精确 UUID；不需详情时返回空数组。

{_OPERATION_CONTRACT}"""
_MERGE_PROMPT = f"""你是 fakuiCode 的自动记忆合并器。
根据给定候选和明确选中的少量现有详情，输出最终受控操作批次。
不得调用工具、指定路径或输出额外字段。只输出单个 JSON 对象，禁止 Markdown 代码围栏和解释文字。
顶层字段必须且仅能是 expected_user_digest、expected_project_digest、operations；摘要值必须原样复制输入。

{_OPERATION_CONTRACT}"""


@dataclass(frozen=True)
class MaintenanceJob:
    turn: CompletedTurn
    snapshot: MemorySnapshot


@dataclass(frozen=True)
class MaintenanceProposalResult:
    batch: MemoryOperationBatch | None
    code: str


class MemoryMaintenanceRunner:
    """Perform at most two structured calls without entering the Agent loop."""

    def __init__(
        self,
        provider_factory: Callable[[ProviderConfig], object],
        repository: MemoryRepository,
        *,
        limits: MemoryLimits = MemoryLimits(),
    ) -> None:
        self.provider_factory = provider_factory
        self.repository = repository
        self.limits = limits

    def propose(self, job: MaintenanceJob) -> MemoryOperationBatch | None:
        return self.propose_result(job).batch

    def propose_result(self, job: MaintenanceJob) -> MaintenanceProposalResult:
        minimal_input = self._minimal_input(job)
        if minimal_input is None:
            return MaintenanceProposalResult(None, "maintenance_input_rejected")
        try:
            first = self._call(job.turn.profile_config, _EXTRACTION_PROMPT, minimal_input)
            envelope = _parse_envelope(first, job.snapshot)
            details = self._selected_details(job.snapshot, envelope["needs_details"])
            if details:
                second_input = json.dumps(
                    {
                        "turn": json.loads(minimal_input),
                        "proposal": envelope,
                        "selected_details": details,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if len(second_input.encode("utf-8")) > (
                    self.limits.maintenance_input_max_bytes
                    + self.limits.candidate_detail_max_bytes
                ):
                    return MaintenanceProposalResult(None, "maintenance_input_rejected")
                final_payload = self._call(job.turn.profile_config, _MERGE_PROMPT, second_input)
            else:
                direct = dict(envelope)
                direct.pop("needs_details")
                final_payload = json.dumps(direct, ensure_ascii=False, separators=(",", ":"))
            visible = self._visible_entries(job.snapshot)
            final_payload = _bind_model_metadata(final_payload, job.turn)
            batch = parse_operation_batch(
                final_payload,
                user_text=job.turn.user_text,
                visible_entries=visible,
                limits=self.limits,
            )
            if (
                batch.expected_user_digest != job.snapshot.user_digest
                or batch.expected_project_digest != job.snapshot.project_digest
            ):
                return MaintenanceProposalResult(None, "maintenance_invalid_output")
            _validate_batch_sources(batch, job.turn, visible)
            return MaintenanceProposalResult(batch, "proposed")
        except ProviderCapabilityError:
            return MaintenanceProposalResult(None, "maintenance_unsupported")
        except MemoryRepositoryError:
            return MaintenanceProposalResult(None, "maintenance_storage_failure")
        except (MemoryValidationError, AttributeError, TypeError, ValueError):
            return MaintenanceProposalResult(None, "maintenance_invalid_output")
        except RuntimeError:
            return MaintenanceProposalResult(None, "maintenance_failed")

    def _call(self, config: ProviderConfig, system_prompt: str, user_payload: str) -> str:
        provider = self.provider_factory(config)
        try:
            request = AgentRequest(
                (AgentMessage("user", user_payload),),
                (),
                system_prompt=system_prompt,
                system_supplement="",
                output_token_limit=self.limits.maintenance_output_token_limit,
            )
            return _collect_structured_stream(
                invoke_provider_stream(provider, request),
                max_bytes=self.limits.maintenance_output_max_bytes,
            )
        finally:
            close_provider = getattr(provider, "close", None)
            if callable(close_provider):
                close_provider()

    def _minimal_input(self, job: MaintenanceJob) -> str | None:
        turn = job.turn
        payload = {
            "conversation_id": turn.conversation_id,
            "user_event_sequence": turn.user_event_sequence,
            "assistant_event_sequence": turn.assistant_event_sequence,
            "user_text": turn.user_text,
            "final_answer": turn.final_answer,
            "tool_summaries": [
                {"name": item.name, "success": item.success, "summary": item.summary}
                for item in turn.tool_summaries
            ],
            "current_index": job.snapshot.rendered,
            "user_digest": job.snapshot.user_digest,
            "project_digest": job.snapshot.project_digest,
            "project_id": job.snapshot.project_id,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= self.limits.maintenance_input_max_bytes:
            return encoded
        payload["tool_summaries"] = []
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return encoded if len(encoded.encode("utf-8")) <= self.limits.maintenance_input_max_bytes else None

    def _selected_details(
        self,
        snapshot: MemorySnapshot,
        requested_ids: list[str],
    ) -> list[dict[str, object]]:
        if len(requested_ids) > self.limits.candidate_detail_max_count:
            raise MemoryValidationError("detail_limit")
        result: list[dict[str, object]] = []
        total = 0
        for entry_id in requested_ids:
            entry = self._read_snapshot_entry(snapshot, entry_id)
            serialized = serialize_entry(entry, limits=self.limits)
            total += len(serialized)
            if total > self.limits.candidate_detail_max_bytes:
                raise MemoryValidationError("detail_limit")
            result.append(
                {
                    "id": entry.id,
                    "scope": entry.scope,
                    "category": entry.category,
                    "summary": entry.summary,
                    "body": entry.body,
                    "updated_at": entry.updated_at,
                }
            )
        return result

    def _visible_entries(self, snapshot: MemorySnapshot) -> dict[str, MemoryEntry]:
        return {entry_id: self._read_snapshot_entry(snapshot, entry_id) for entry_id in snapshot.active_ids}

    def _read_snapshot_entry(self, snapshot: MemorySnapshot, entry_id: str) -> MemoryEntry:
        if entry_id not in snapshot.active_ids:
            raise MemoryRepositoryError("entry_unavailable")
        try:
            return self.repository.read_active(MemoryScopeRef("user"), entry_id)
        except MemoryRepositoryError:
            if snapshot.project_id is None:
                raise
            return self.repository.read_active(
                MemoryScopeRef("project", snapshot.project_id), entry_id
            )


def _validate_batch_sources(
    batch: MemoryOperationBatch,
    turn: CompletedTurn,
    visible_entries: Mapping[str, MemoryEntry],
) -> None:
    """Accept only existing provenance or a bounded reference to the current turn."""

    existing_sources = {
        source for entry in visible_entries.values() for source in entry.sources
    }
    for operation in batch.operations:
        if not isinstance(operation, CreateEntry | UpdateEntry | MergeEntries):
            continue
        has_current_source = False
        for source in operation.sources:
            is_current_source = (
                source.conversation_id == turn.conversation_id
                and source.source_type == "user_turn"
                and source.event_sequence == turn.user_event_sequence
            ) or (
                source.conversation_id == turn.conversation_id
                and source.source_type == "assistant_final"
                and source.event_sequence == turn.assistant_event_sequence
            ) or (
                source.conversation_id == turn.conversation_id
                and source.source_type == "tool_summary"
                and bool(turn.tool_summaries)
                and turn.user_event_sequence < source.event_sequence < turn.assistant_event_sequence
            )
            if is_current_source:
                has_current_source = True
                continue
            if source in existing_sources:
                continue
            raise MemoryValidationError("invalid_source")
        if not has_current_source:
            raise MemoryValidationError("invalid_source")


def _bind_model_metadata(payload: str, turn: CompletedTurn) -> str:
    """Bind trusted provenance and derive offsets from an exact user quote."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MemoryValidationError("duplicate_field")
            result[key] = value
        return result

    try:
        loaded = json.loads(payload, object_pairs_hook=unique)
    except json.JSONDecodeError as error:
        raise MemoryValidationError("invalid_json") from error
    if not isinstance(loaded, dict) or not isinstance(loaded.get("operations"), list):
        raise MemoryValidationError("invalid_schema")

    operations: list[dict[str, object]] = []
    for raw in loaded["operations"]:
        if not isinstance(raw, Mapping):
            raise MemoryValidationError("invalid_schema")
        operation = dict(raw)
        if operation.get("kind") in {"create", "update", "merge"}:
            operation.setdefault(
                "sources",
                [
                    {
                        "conversation_id": turn.conversation_id,
                        "event_sequence": turn.user_event_sequence,
                        "source_type": "user_turn",
                    }
                ],
            )
            evidence = operation.get("evidence")
            if isinstance(evidence, Mapping) and set(evidence) == {"text", "intent"}:
                quote = evidence.get("text")
                if not isinstance(quote, str) or not quote:
                    raise MemoryValidationError("invalid_evidence")
                start = turn.user_text.find(quote)
                if start < 0:
                    raise MemoryValidationError("invalid_evidence")
                operation["evidence"] = {
                    "start": start,
                    "end": start + len(quote),
                    "intent": evidence.get("intent"),
                    "text": quote,
                }
        operations.append(operation)
    loaded["operations"] = operations
    return json.dumps(loaded, ensure_ascii=False, separators=(",", ":"))


class MemoryMaintenanceCoordinator:
    """Daemon-backed single worker with exactly one replaceable pending slot."""

    def __init__(
        self,
        processor: Callable[[MaintenanceJob], str | MemoryOperationBatch | None],
        *,
        on_result: Callable[[str | MemoryOperationBatch | None], None] | None = None,
    ) -> None:
        self._processor = processor
        self._on_result = on_result
        self._condition = Condition()
        self._pending: MaintenanceJob | None = None
        self._closed = False
        self._running = False
        self._thread = Thread(target=self._work, name="fakuicode-memory", daemon=True)
        self._thread.start()

    def submit(self, job: MaintenanceJob) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._pending = job
            self._condition.notify()
            return True

    def close(self, *, wait: bool = False) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if wait:
            self._thread.join()

    @property
    def pending_count(self) -> int:
        with self._condition:
            return int(self._pending is not None)

    def _work(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                job = self._pending
                self._pending = None
                self._running = True
            assert job is not None
            try:
                result = self._processor(job)
            except Exception:
                result = None
            if self._on_result is not None:
                try:
                    self._on_result(result)
                except Exception:
                    pass
            with self._condition:
                self._running = False
                if self._closed:
                    self._pending = None
                self._condition.notify_all()


def _collect_structured_stream(events: Iterator[AgentStreamEvent], *, max_bytes: int) -> str:
    chunks: list[str] = []
    byte_count = 0
    completed = False
    thinking_open = False
    for event in events:
        if event.kind == "text_delta":
            if completed or thinking_open:
                raise MemoryValidationError("invalid_stream")
            byte_count += len(event.text.encode("utf-8"))
            if byte_count > max_bytes:
                raise MemoryValidationError("output_limit")
            chunks.append(event.text)
        elif event.kind == "thinking_start":
            if completed or thinking_open or chunks:
                raise MemoryValidationError("invalid_stream")
            thinking_open = True
        elif event.kind == "thinking_delta":
            if completed or not thinking_open:
                raise MemoryValidationError("invalid_stream")
            byte_count += len(event.text.encode("utf-8"))
            if byte_count > max_bytes:
                raise MemoryValidationError("output_limit")
        elif event.kind == "thinking_end":
            if completed or not thinking_open:
                raise MemoryValidationError("invalid_stream")
            thinking_open = False
        elif event.kind == "usage":
            if completed:
                raise MemoryValidationError("invalid_stream")
        elif event.kind == "completed" and not completed and not thinking_open:
            completed = True
        else:
            raise MemoryValidationError("invalid_stream")
    if not completed or not chunks:
        raise MemoryValidationError("incomplete_stream")
    return "".join(chunks)


def _parse_envelope(payload: str, snapshot: MemorySnapshot) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MemoryValidationError("duplicate_field")
            result[key] = value
        return result

    try:
        loaded = json.loads(payload, object_pairs_hook=unique)
    except json.JSONDecodeError as error:
        raise MemoryValidationError("invalid_json") from error
    if not isinstance(loaded, dict) or set(loaded) != {
        "expected_user_digest",
        "expected_project_digest",
        "needs_details",
        "operations",
    }:
        raise MemoryValidationError("invalid_schema")
    if loaded["expected_user_digest"] != snapshot.user_digest or loaded[
        "expected_project_digest"
    ] != snapshot.project_digest:
        raise MemoryValidationError("stale_state")
    needs = loaded["needs_details"]
    operations = loaded["operations"]
    if not isinstance(needs, list) or not isinstance(operations, list):
        raise MemoryValidationError("invalid_schema")
    if not operations and not needs:
        raise MemoryValidationError("invalid_schema")
    seen: set[str] = set()
    for entry_id in needs:
        if not isinstance(entry_id, str):
            raise MemoryValidationError("invalid_id")
        try:
            canonical_uuid(entry_id)
        except ValueError as error:
            raise MemoryValidationError("invalid_id") from error
        if entry_id not in snapshot.active_ids or entry_id in seen:
            raise MemoryValidationError("unknown_id")
        seen.add(entry_id)
    return loaded
