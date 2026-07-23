"""Immutable domain models and resource limits for automatic memory."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, TypeAlias
from uuid import UUID

from fakuicode.models import ProviderConfig


MemoryScope: TypeAlias = Literal["user", "project"]
MemoryCategory: TypeAlias = Literal[
    "user_preference",
    "correction",
    "project_knowledge",
    "reference",
]
MemorySourceType: TypeAlias = Literal["user_turn", "assistant_final", "tool_summary"]
MemoryDiagnosticCode: TypeAlias = Literal[
    "invalid_entry",
    "scope_unavailable",
    "scope_overflow",
    "identity_unavailable",
    "lock_timeout",
    "stale_state",
    "storage_failure",
    "maintenance_skipped",
    "maintenance_failed",
]


def canonical_uuid(value: str, *, field_name: str = "id") -> str:
    """Return a canonical UUID string or raise a content-free validation error."""
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a canonical UUID.") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(f"{field_name} must be a canonical UUID.")
    return canonical


@dataclass(frozen=True)
class MemoryLimits:
    snapshot_max_lines: int = 200
    snapshot_max_bytes: int = 25 * 1024
    boundary_max_lines: int = 10
    boundary_max_bytes: int = 2 * 1024
    user_index_max_lines: int = 60
    user_index_max_bytes: int = 8 * 1024
    project_index_max_lines: int = 130
    project_index_max_bytes: int = 15 * 1024
    entry_max_bytes: int = 16 * 1024
    summary_max_bytes: int = 256
    body_max_bytes: int = 12 * 1024
    maintenance_input_max_bytes: int = 25 * 1024
    candidate_detail_max_count: int = 8
    candidate_detail_max_bytes: int = 25 * 1024
    maintenance_output_max_bytes: int = 32 * 1024
    maintenance_output_token_limit: int = 4_000
    maintenance_max_calls: int = 2
    write_lock_timeout_seconds: float = 1.0
    pending_turn_slots: int = 1

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{item.name} must be positive.")


@dataclass(frozen=True)
class MemorySourceRef:
    conversation_id: str
    event_sequence: int
    source_type: MemorySourceType

    def __post_init__(self) -> None:
        canonical_uuid(self.conversation_id, field_name="conversation_id")
        if isinstance(self.event_sequence, bool) or self.event_sequence < 0:
            raise ValueError("event_sequence must be non-negative.")
        if self.source_type not in {"user_turn", "assistant_final", "tool_summary"}:
            raise ValueError("source_type is invalid.")


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    scope: MemoryScope
    category: MemoryCategory
    summary: str
    body: str
    created_at: str
    updated_at: str
    sources: tuple[MemorySourceRef, ...]

    def __post_init__(self) -> None:
        canonical_uuid(self.id)
        if self.scope not in {"user", "project"}:
            raise ValueError("scope is invalid.")
        if self.category not in {
            "user_preference",
            "correction",
            "project_knowledge",
            "reference",
        }:
            raise ValueError("category is invalid.")
        if self.scope == "user" and self.category in {"project_knowledge", "reference"}:
            raise ValueError("Project knowledge and reference entries are project-only.")
        if not self.sources:
            raise ValueError("sources must not be empty.")


@dataclass(frozen=True)
class MemoryDiagnostic:
    code: MemoryDiagnosticCode
    scope: MemoryScope | None = None


@dataclass(frozen=True)
class MemoryScopeRef:
    scope: MemoryScope
    project_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope == "project":
            if self.project_id is None:
                raise ValueError("project_id is required for project scope.")
            canonical_uuid(self.project_id, field_name="project_id")
        elif self.project_id is not None:
            raise ValueError("project_id is not allowed for user scope.")


@dataclass(frozen=True)
class ScopeSnapshot:
    scope_ref: MemoryScopeRef
    entries: tuple[MemoryEntry, ...]
    index: str
    digest: str
    diagnostics: tuple[MemoryDiagnostic, ...] = ()

    @property
    def active_ids(self) -> frozenset[str]:
        return frozenset(entry.id for entry in self.entries)


@dataclass(frozen=True)
class MemorySnapshot:
    rendered: str
    active_ids: frozenset[str]
    project_id: str | None
    user_digest: str
    project_digest: str | None
    diagnostics: tuple[MemoryDiagnostic, ...]

    def __post_init__(self) -> None:
        for entry_id in self.active_ids:
            canonical_uuid(entry_id)
        if self.project_id is not None:
            canonical_uuid(self.project_id, field_name="project_id")


@dataclass(frozen=True)
class AgentTurnContext:
    memory_snapshot: MemorySnapshot | None = None
    first_request_reminder: str = ""
    settings_generation: int | None = None


@dataclass(frozen=True)
class SafeToolSummary:
    name: str
    success: bool
    summary: str


@dataclass(frozen=True)
class CompletedTurn:
    conversation_id: str
    user_event_sequence: int
    assistant_event_sequence: int
    user_text: str
    final_answer: str
    tool_summaries: tuple[SafeToolSummary, ...]
    profile_config: ProviderConfig
    project_id: str | None
    settings_generation: int


@dataclass(frozen=True)
class UserTextEvidence:
    start: int
    end: int
    intent: Literal["cross_project", "project_only"]

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Evidence offset range must be non-negative and ordered.")


@dataclass(frozen=True)
class Noop:
    kind: Literal["noop"] = "noop"


@dataclass(frozen=True)
class CreateEntry:
    scope: MemoryScope
    category: MemoryCategory
    summary: str
    body: str
    sources: tuple[MemorySourceRef, ...]
    evidence: UserTextEvidence | None = None
    kind: Literal["create"] = "create"


@dataclass(frozen=True)
class UpdateEntry:
    entry_id: str
    summary: str
    body: str
    sources: tuple[MemorySourceRef, ...]
    evidence: UserTextEvidence | None = None
    kind: Literal["update"] = "update"


@dataclass(frozen=True)
class MergeEntries:
    entry_ids: tuple[str, ...]
    scope: MemoryScope
    category: MemoryCategory
    summary: str
    body: str
    sources: tuple[MemorySourceRef, ...]
    evidence: UserTextEvidence | None = None
    kind: Literal["merge"] = "merge"


@dataclass(frozen=True)
class DeleteSuperseded:
    entry_ids: tuple[str, ...]
    kind: Literal["delete"] = "delete"


MemoryOperation: TypeAlias = Noop | CreateEntry | UpdateEntry | MergeEntries | DeleteSuperseded


@dataclass(frozen=True)
class MemoryOperationBatch:
    expected_user_digest: str
    expected_project_digest: str | None
    operations: tuple[MemoryOperation, ...]


@dataclass(frozen=True)
class VisibleScopes:
    user: MemoryScopeRef
    project: MemoryScopeRef | None = None


@dataclass(frozen=True)
class CommitResult:
    success: bool
    code: str
    entry_ids: tuple[str, ...] = ()
