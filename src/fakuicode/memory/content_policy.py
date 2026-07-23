"""Strict parsing and deterministic content safeguards for automatic memory."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import re
import unicodedata
from uuid import uuid4

import yaml

from fakuicode.memory.models import (
    CreateEntry,
    DeleteSuperseded,
    MemoryCategory,
    MemoryEntry,
    MemoryLimits,
    MemoryOperationBatch,
    MemoryScope,
    MemorySourceRef,
    MergeEntries,
    Noop,
    UpdateEntry,
    UserTextEvidence,
    canonical_uuid,
)


ENTRY_FIELDS = {
    "id",
    "scope",
    "category",
    "summary",
    "created_at",
    "updated_at",
    "sources",
}
SOURCE_FIELDS = {"conversation_id", "event_sequence", "source_type"}
SCOPES = {"user", "project"}
CATEGORIES = {"user_preference", "correction", "project_knowledge", "reference"}
SOURCE_TYPES = {"user_turn", "assistant_final", "tool_summary"}
PROJECT_ONLY_CATEGORIES = {"project_knowledge", "reference"}
DEFAULT_MEMORY_LIMITS = MemoryLimits()


class MemoryValidationError(ValueError):
    """Content-free validation failure safe to surface as an enum-like code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _DuplicateFieldError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise _DuplicateFieldError
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
_UniqueKeyLoader.add_constructor(
    "tag:yaml.org,2002:timestamp",
    lambda loader, node: loader.construct_scalar(node),
)


_SENSITIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in (
        r"-----BEGIN (?:[A-Z ]+ )?(?:PRIVATE KEY|CERTIFICATE)-----",
        r"^\s*(?:authorization|proxy-authorization)\s*:\s*\S+",
        r"^\s*(?:cookie|set-cookie)\s*:\s*\S+",
        r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b",
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{16,}\b",
        r"\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|private[_-]?key|connection[_-]?string)\b\s*[:=]\s*[^\s,;]{8,}",
        r"\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)\s*=\s*[^\s]{8,}",
        r"https?://[^\s/:@]+:[^\s/@]+@",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
        r"(?<!\d)\d{17}[\dXx](?!\d)",
    )
)
_CROSS_PROJECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"所有项目",
        r"全部项目",
        r"无论(?:哪|哪个|什么).*项目",
        r"以后.*(?:都|默认)",
        r"(?:all|every)\s+(?:my\s+)?projects?",
        r"across\s+(?:all\s+)?projects?",
    )
)


def contains_sensitive_content(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in _SENSITIVE_PATTERNS)


def serialize_entry(
    entry: MemoryEntry,
    *,
    limits: MemoryLimits = DEFAULT_MEMORY_LIMITS,
) -> bytes:
    """Serialize one entry with fixed field order and stable UTF-8 newlines."""
    _validate_entry(entry, limits)
    frontmatter = {
        "id": entry.id,
        "scope": entry.scope,
        "category": entry.category,
        "summary": entry.summary,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "sources": [
            {
                "conversation_id": source.conversation_id,
                "event_sequence": source.event_sequence,
                "source_type": source.source_type,
            }
            for source in entry.sources
        ],
    }
    yaml_text = yaml.safe_dump(
        frontmatter,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=4096,
        line_break="\n",
    )
    payload = f"---\n{yaml_text}---\n{entry.body.rstrip()}\n".encode("utf-8")
    if len(payload) > limits.entry_max_bytes:
        raise MemoryValidationError("entry_too_large")
    return payload


def parse_entry_bytes(
    payload: bytes,
    *,
    expected_id: str,
    limits: MemoryLimits = DEFAULT_MEMORY_LIMITS,
) -> MemoryEntry:
    if len(payload) > limits.entry_max_bytes:
        raise MemoryValidationError("entry_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MemoryValidationError("invalid_utf8") from error
    normalized = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n") or "\n---\n" not in normalized[4:]:
        raise MemoryValidationError("invalid_frontmatter")
    raw_frontmatter, body = normalized[4:].split("\n---\n", 1)
    try:
        metadata = yaml.load(raw_frontmatter, Loader=_UniqueKeyLoader)
    except _DuplicateFieldError as error:
        raise MemoryValidationError("duplicate_field") from error
    except yaml.YAMLError as error:
        raise MemoryValidationError("invalid_frontmatter") from error
    if not isinstance(metadata, Mapping):
        raise MemoryValidationError("invalid_frontmatter")
    if not all(isinstance(key, str) for key in metadata):
        raise MemoryValidationError("invalid_frontmatter")
    if set(metadata) - ENTRY_FIELDS:
        raise MemoryValidationError("unknown_field")
    if set(metadata) != ENTRY_FIELDS:
        raise MemoryValidationError("missing_field")

    entry_id = _required_string(metadata, "id")
    try:
        canonical_uuid(entry_id)
        canonical_uuid(expected_id, field_name="expected_id")
    except ValueError as error:
        raise MemoryValidationError("invalid_id") from error
    if entry_id != expected_id:
        raise MemoryValidationError("id_mismatch")
    scope = _scope(metadata.get("scope"))
    category = _category(metadata.get("category"))
    summary = _required_string(metadata, "summary")
    created_at = _valid_time(metadata.get("created_at"))
    updated_at = _valid_time(metadata.get("updated_at"))
    sources = _parse_sources(metadata.get("sources"))
    entry = MemoryEntry(
        id=entry_id,
        scope=scope,
        category=category,
        summary=summary,
        body=body[:-1] if body.endswith("\n") else body,
        created_at=created_at,
        updated_at=updated_at,
        sources=sources,
    )
    _validate_entry(entry, limits)
    return entry


def parse_operation_batch(
    payload: str,
    *,
    user_text: str,
    visible_entries: Mapping[str, MemoryEntry],
    limits: MemoryLimits = DEFAULT_MEMORY_LIMITS,
) -> MemoryOperationBatch:
    raw = _load_unique_json(payload)
    _expect_keys(raw, {"expected_user_digest", "expected_project_digest", "operations"})
    user_digest = _json_string(raw, "expected_user_digest", allow_empty=True)
    project_digest = raw.get("expected_project_digest")
    if project_digest is not None and not isinstance(project_digest, str):
        raise MemoryValidationError("invalid_schema")
    raw_operations = raw.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise MemoryValidationError("invalid_schema")

    operations = []
    touched_ids: set[str] = set()
    for raw_operation in raw_operations:
        if not isinstance(raw_operation, Mapping):
            raise MemoryValidationError("invalid_schema")
        operation, referenced = _parse_operation(
            raw_operation,
            user_text=user_text,
            visible_entries=visible_entries,
            limits=limits,
        )
        if touched_ids.intersection(referenced):
            raise MemoryValidationError("conflicting_operation")
        touched_ids.update(referenced)
        operations.append(operation)
    if any(isinstance(item, Noop) for item in operations) and len(operations) != 1:
        raise MemoryValidationError("conflicting_operation")
    return MemoryOperationBatch(user_digest, project_digest, tuple(operations))


def _parse_operation(
    raw: Mapping[str, object],
    *,
    user_text: str,
    visible_entries: Mapping[str, MemoryEntry],
    limits: MemoryLimits,
) -> tuple[object, set[str]]:
    kind = _json_string(raw, "kind")
    if kind == "noop":
        _expect_keys(raw, {"kind"})
        return Noop(), set()
    if kind == "create":
        allowed = {"kind", "scope", "category", "summary", "body", "sources", "evidence"}
        _expect_keys(raw, allowed, required=allowed - {"evidence"})
        scope = _scope(raw.get("scope"))
        category = _category(raw.get("category"))
        evidence = _operation_evidence(raw.get("evidence"), user_text, required=scope == "user")
        sources = _parse_sources(raw.get("sources"))
        summary, body = _operation_text(raw, limits)
        candidate = MemoryEntry(
            str(uuid4()), scope, category, summary, body, _now_placeholder(), _now_placeholder(), sources
        )
        _validate_entry(candidate, limits)
        return CreateEntry(scope, category, summary, body, sources, evidence), set()
    if kind == "update":
        allowed = {"kind", "entry_id", "summary", "body", "sources", "evidence"}
        _expect_keys(raw, allowed, required=allowed - {"evidence"})
        entry_id = _visible_id(raw.get("entry_id"), visible_entries)
        current = visible_entries[entry_id]
        evidence = _operation_evidence(raw.get("evidence"), user_text, required=current.scope == "user")
        summary, body = _operation_text(raw, limits)
        sources = _parse_sources(raw.get("sources"))
        candidate = MemoryEntry(
            current.id,
            current.scope,
            current.category,
            summary,
            body,
            current.created_at,
            current.updated_at,
            sources,
        )
        _validate_entry(candidate, limits)
        return UpdateEntry(entry_id, summary, body, sources, evidence), {entry_id}
    if kind == "merge":
        allowed = {
            "kind", "entry_ids", "scope", "category", "summary", "body", "sources", "evidence"
        }
        _expect_keys(raw, allowed, required=allowed - {"evidence"})
        entry_ids = _visible_ids(raw.get("entry_ids"), visible_entries, minimum=2)
        scope = _scope(raw.get("scope"))
        category = _category(raw.get("category"))
        if any(visible_entries[item].scope != scope for item in entry_ids):
            raise MemoryValidationError("invalid_scope")
        evidence = _operation_evidence(raw.get("evidence"), user_text, required=scope == "user")
        summary, body = _operation_text(raw, limits)
        sources = _parse_sources(raw.get("sources"))
        candidate = MemoryEntry(
            str(uuid4()), scope, category, summary, body, _now_placeholder(), _now_placeholder(), sources
        )
        _validate_entry(candidate, limits)
        return MergeEntries(entry_ids, scope, category, summary, body, sources, evidence), set(entry_ids)
    if kind == "delete":
        _expect_keys(raw, {"kind", "entry_ids"})
        entry_ids = _visible_ids(raw.get("entry_ids"), visible_entries)
        return DeleteSuperseded(entry_ids), set(entry_ids)
    raise MemoryValidationError("unknown_operation")


def _load_unique_json(payload: str) -> Mapping[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateFieldError
            result[key] = value
        return result

    try:
        loaded = json.loads(payload, object_pairs_hook=unique)
    except _DuplicateFieldError as error:
        raise MemoryValidationError("duplicate_field") from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise MemoryValidationError("invalid_json") from error
    if not isinstance(loaded, Mapping):
        raise MemoryValidationError("invalid_schema")
    return loaded


def _expect_keys(
    value: Mapping[str, object],
    allowed: set[str],
    *,
    required: set[str] | None = None,
) -> None:
    if not all(isinstance(key, str) for key in value) or set(value) - allowed:
        raise MemoryValidationError("unknown_field")
    if not (required if required is not None else allowed).issubset(value):
        raise MemoryValidationError("missing_field")


def _parse_sources(raw: object) -> tuple[MemorySourceRef, ...]:
    if not isinstance(raw, list) or not raw:
        raise MemoryValidationError("invalid_source")
    sources: list[MemorySourceRef] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise MemoryValidationError("invalid_source")
        _expect_keys(value, SOURCE_FIELDS)
        conversation_id = _json_string(value, "conversation_id")
        sequence = value.get("event_sequence")
        source_type = value.get("source_type")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
            or source_type not in SOURCE_TYPES
        ):
            raise MemoryValidationError("invalid_source")
        try:
            sources.append(
                MemorySourceRef(conversation_id, sequence, source_type)  # type: ignore[arg-type]
            )
        except ValueError as error:
            raise MemoryValidationError("invalid_source") from error
    return tuple(sources)


def _validate_entry(entry: MemoryEntry, limits: MemoryLimits) -> None:
    try:
        canonical_uuid(entry.id)
    except ValueError as error:
        raise MemoryValidationError("invalid_id") from error
    if not entry.summary.strip() or len(entry.summary.encode("utf-8")) > limits.summary_max_bytes:
        raise MemoryValidationError("summary_too_large")
    lowered_summary = entry.summary.casefold()
    if (
        entry.summary != entry.summary.strip()
        or any(character in "\r\n" or unicodedata.category(character).startswith("C") for character in entry.summary)
        or "<long_term_memory" in lowered_summary
        or "</long_term_memory" in lowered_summary
        or "<system-reminder" in lowered_summary
        or "</system-reminder" in lowered_summary
    ):
        raise MemoryValidationError("invalid_summary")
    if not entry.body.strip() or len(entry.body.encode("utf-8")) > limits.body_max_bytes:
        raise MemoryValidationError("body_too_large")
    if contains_sensitive_content(entry.summary) or contains_sensitive_content(entry.body):
        raise MemoryValidationError("sensitive_content")
    _valid_time(entry.created_at)
    _valid_time(entry.updated_at)


def _operation_text(raw: Mapping[str, object], limits: MemoryLimits) -> tuple[str, str]:
    summary = _json_string(raw, "summary")
    body = _json_string(raw, "body")
    if len(summary.encode("utf-8")) > limits.summary_max_bytes:
        raise MemoryValidationError("summary_too_large")
    if len(body.encode("utf-8")) > limits.body_max_bytes:
        raise MemoryValidationError("body_too_large")
    if contains_sensitive_content(summary) or contains_sensitive_content(body):
        raise MemoryValidationError("sensitive_content")
    return summary, body


def _operation_evidence(raw: object, user_text: str, *, required: bool) -> UserTextEvidence | None:
    if raw is None:
        if required:
            raise MemoryValidationError("invalid_evidence")
        return None
    if not isinstance(raw, Mapping):
        raise MemoryValidationError("invalid_evidence")
    _expect_keys(raw, {"start", "end", "intent", "text"})
    start, end, intent, text = raw.get("start"), raw.get("end"), raw.get("intent"), raw.get("text")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or intent not in {"cross_project", "project_only"}
        or not isinstance(text, str)
        or start < 0
        or end <= start
        or end > len(user_text)
        or user_text[start:end] != text
    ):
        raise MemoryValidationError("invalid_evidence")
    if required and (
        intent != "cross_project"
        or not any(pattern.search(text) is not None for pattern in _CROSS_PROJECT_PATTERNS)
    ):
        raise MemoryValidationError("invalid_evidence")
    return UserTextEvidence(start, end, intent)  # type: ignore[arg-type]


def _visible_id(raw: object, visible_entries: Mapping[str, MemoryEntry]) -> str:
    if not isinstance(raw, str):
        raise MemoryValidationError("invalid_id")
    try:
        canonical_uuid(raw)
    except ValueError as error:
        raise MemoryValidationError("invalid_id") from error
    if raw not in visible_entries:
        raise MemoryValidationError("unknown_id")
    return raw


def _visible_ids(
    raw: object,
    visible_entries: Mapping[str, MemoryEntry],
    *,
    minimum: int = 1,
) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) < minimum:
        raise MemoryValidationError("invalid_id")
    ids = tuple(_visible_id(value, visible_entries) for value in raw)
    if len(ids) != len(set(ids)):
        raise MemoryValidationError("duplicate_id")
    return ids


def _required_string(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise MemoryValidationError("invalid_field")
    return raw


def _json_string(value: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or (not allow_empty and not raw.strip()):
        raise MemoryValidationError("invalid_schema")
    return raw


def _scope(value: object) -> MemoryScope:
    if value not in SCOPES:
        raise MemoryValidationError("invalid_scope")
    return value  # type: ignore[return-value]


def _category(value: object) -> MemoryCategory:
    if value not in CATEGORIES:
        raise MemoryValidationError("invalid_category")
    return value  # type: ignore[return-value]


def _valid_time(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MemoryValidationError("invalid_time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise MemoryValidationError("invalid_time") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise MemoryValidationError("invalid_time")
    return value


def _now_placeholder() -> str:
    return "1970-01-01T00:00:00Z"
