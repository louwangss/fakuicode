from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from fakuicode.memory.models import (
    AgentTurnContext,
    MemoryEntry,
    MemoryLimits,
    MemorySnapshot,
    MemorySourceRef,
    UserTextEvidence,
)


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


def test_memory_limits_match_the_approved_resource_budget() -> None:
    limits = MemoryLimits()

    assert limits.snapshot_max_lines == 200
    assert limits.snapshot_max_bytes == 25 * 1024
    assert limits.entry_max_bytes == 16 * 1024
    assert limits.summary_max_bytes == 256
    assert limits.body_max_bytes == 12 * 1024
    assert limits.maintenance_input_max_bytes == 25 * 1024
    assert limits.candidate_detail_max_count == 8
    assert limits.candidate_detail_max_bytes == 25 * 1024
    assert limits.maintenance_output_max_bytes == 32 * 1024
    assert limits.maintenance_output_token_limit == 4_000
    assert limits.maintenance_max_calls == 2
    assert limits.write_lock_timeout_seconds == 1.0
    assert limits.pending_turn_slots == 1


def test_memory_limits_reject_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        MemoryLimits(snapshot_max_lines=0)

    with pytest.raises(ValueError, match="positive"):
        MemoryLimits(write_lock_timeout_seconds=-1)


def test_entry_requires_a_canonical_uuid_and_valid_scope_category_pair() -> None:
    source = MemorySourceRef(CONVERSATION_ID, 3, "user_turn")
    entry = MemoryEntry(
        id=str(uuid4()),
        scope="project",
        category="project_knowledge",
        summary="The project uses Python.",
        body="Use Python 3.11 or newer.",
        created_at="2026-07-21T00:00:00Z",
        updated_at="2026-07-21T00:00:00Z",
        sources=(source,),
    )

    assert entry.sources == (source,)
    with pytest.raises(FrozenInstanceError):
        entry.summary = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError, match="UUID"):
        MemoryEntry(
            id="../note",
            scope="project",
            category="project_knowledge",
            summary="summary",
            body="body",
            created_at="2026-07-21T00:00:00Z",
            updated_at="2026-07-21T00:00:00Z",
            sources=(),
        )

    with pytest.raises(ValueError, match="project-only"):
        MemoryEntry(
            id=str(uuid4()),
            scope="user",
            category="reference",
            summary="summary",
            body="body",
            created_at="2026-07-21T00:00:00Z",
            updated_at="2026-07-21T00:00:00Z",
            sources=(),
        )


def test_snapshot_active_ids_are_immutable_canonical_uuids() -> None:
    entry_id = str(uuid4())
    snapshot = MemorySnapshot(
        rendered="memory",
        active_ids=frozenset({entry_id}),
        project_id=None,
        user_digest="digest",
        project_digest=None,
        diagnostics=(),
    )

    assert snapshot.active_ids == frozenset({entry_id})
    assert AgentTurnContext(snapshot).memory_snapshot is snapshot

    with pytest.raises(ValueError, match="UUID"):
        MemorySnapshot("", frozenset({"notes/file.md"}), None, "", None, ())


def test_source_and_user_evidence_require_non_negative_ordered_offsets() -> None:
    with pytest.raises(ValueError, match="event_sequence"):
        MemorySourceRef(CONVERSATION_ID, -1, "user_turn")

    with pytest.raises(ValueError, match="conversation_id"):
        MemorySourceRef("secret=value", 1, "user_turn")

    with pytest.raises(ValueError, match="source_type"):
        MemorySourceRef(CONVERSATION_ID, 1, "repository_text")  # type: ignore[arg-type]

    assert UserTextEvidence(0, 4, "cross_project").end == 4
    with pytest.raises(ValueError, match="offset"):
        UserTextEvidence(4, 3, "project_only")
