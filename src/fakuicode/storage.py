"""Durable local conversation storage with ordered timeline events."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import RLock
from time import time_ns
from uuid import UUID, uuid4

from fakuicode.models import TimelineEvent, TimelineEventKind


_DIAGNOSTIC_INTEGER_FIELDS = {
    "estimated_before",
    "estimated_after",
    "threshold",
    "artifact_count",
    "artifact_bytes",
    "duration_ms",
    "consecutive_failures",
}
_DIAGNOSTIC_ENUM_FIELDS = {
    "trigger": {"automatic", "manual", "emergency"},
    "result": {"offloaded", "compacted", "noop", "failed", "blocked", "breaker"},
    "error_category": {
        "none",
        "provider",
        "context_overflow",
        "invalid_summary",
        "artifact_write",
        "hard_limit",
        "cancelled",
        "other",
    },
}
_SUMMARY_TRIGGERS = {"automatic", "manual", "emergency"}
DEFAULT_CONVERSATION_TITLE = "New conversation"


@dataclass(frozen=True)
class ConversationRecord:
    """A locally persisted conversation summary."""

    id: str
    title: str
    workspace: Path
    profile_name: str
    created_at: int
    updated_at: int
    conversation_type: str = "main"
    parent_conversation_id: str | None = None
    skill_name: str | None = None
    status: str = "active"
    agent_name: str | None = None


def default_store_path(home: Path | None = None) -> Path:
    """Return Fakuicode's private, per-user conversation database path."""
    return (home or Path.home()) / ".fakuicode" / "conversations.sqlite3"


class ConversationStore:
    """SQLite-backed conversation store; all writes are transactional."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_conversation(
        self,
        title: str,
        workspace: Path,
        profile_name: str,
        *,
        conversation_type: str = "main",
        parent_conversation_id: str | None = None,
        skill_name: str | None = None,
        agent_name: str | None = None,
        status: str = "active",
        conversation_id: str | None = None,
    ) -> ConversationRecord:
        if conversation_type not in {"main", "skill", "agent"}:
            raise ValueError("Invalid conversation type.")
        if conversation_type == "skill" and (
            not parent_conversation_id or not skill_name or agent_name is not None
        ):
            raise ValueError("Skill conversations require a parent and Skill name.")
        if conversation_type == "agent" and (
            not parent_conversation_id or not agent_name or skill_name is not None
        ):
            raise ValueError("Agent conversations require a parent and Agent name.")
        if conversation_type == "main" and (
            parent_conversation_id is not None
            or skill_name is not None
            or agent_name is not None
        ):
            raise ValueError("Main conversations cannot have child metadata.")
        with self._write_transaction():
            now = self._next_conversation_timestamp()
            record = ConversationRecord(
                str(uuid4()) if conversation_id is None else str(UUID(conversation_id)),
                title.strip() or DEFAULT_CONVERSATION_TITLE,
                workspace.resolve(),
                profile_name,
                now,
                now,
                conversation_type,
                parent_conversation_id,
                skill_name,
                status,
                agent_name,
            )
            self._connection.execute(
                """
                INSERT INTO conversations (
                    id, title, workspace, profile_name, created_at, updated_at,
                    conversation_type, parent_conversation_id, skill_name, status, agent_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id, record.title, str(record.workspace), record.profile_name,
                    record.created_at, record.updated_at, record.conversation_type,
                    record.parent_conversation_id, record.skill_name, record.status,
                    record.agent_name,
                ),
            )
        return record

    def ensure_conversation_title(
        self,
        conversation_id: str,
        candidate: str,
    ) -> ConversationRecord:
        """Replace only the default title with a normalized first user prompt."""

        with self._write_transaction():
            row = self._connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Conversation was not found.")
            record = _record_from_row(row)
            if record.conversation_type != "main":
                return record
            if record.title != DEFAULT_CONVERSATION_TITLE:
                return record
            title = " ".join(candidate.split())
            if not title:
                return record
            self._connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = ?",
                (title, conversation_id, DEFAULT_CONVERSATION_TITLE),
            )
            updated = self._connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            assert updated is not None
            return _record_from_row(updated)

    def backfill_default_conversation_titles(self) -> None:
        """Backfill default titles from first user messages without changing activity order."""

        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT conversations.id, timeline_events.content
                FROM conversations
                JOIN timeline_events
                  ON timeline_events.conversation_id = conversations.id
                WHERE conversations.title = ?
                  AND conversations.conversation_type = 'main'
                  AND timeline_events.kind = 'user'
                  AND (timeline_events.metadata IS NULL OR timeline_events.metadata NOT LIKE '%"skill_invocation"%')
                  AND timeline_events.sequence = (
                      SELECT MIN(first_user.sequence)
                      FROM timeline_events AS first_user
                      WHERE first_user.conversation_id = conversations.id
                        AND first_user.kind = 'user'
                        AND (first_user.metadata IS NULL OR first_user.metadata NOT LIKE '%"skill_invocation"%')
                  )
                """,
                (DEFAULT_CONVERSATION_TITLE,),
            ).fetchall()
            updates: list[tuple[str, str, str]] = []
            for row in rows:
                title = " ".join(str(row["content"]).split())
                if title:
                    updates.append((title, str(row["id"]), DEFAULT_CONVERSATION_TITLE))
            self._connection.executemany(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = ?",
                updates,
            )

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError("Conversation was not found.")
        return _record_from_row(row)

    def list_conversations(
        self,
        *,
        workspace: Path | None = None,
    ) -> list[ConversationRecord]:
        parameters: tuple[object, ...] = ()
        workspace_clause = ""
        if workspace is not None:
            workspace_clause = " AND workspace = ?"
            parameters = (str(workspace.resolve()),)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM conversations "
                "WHERE conversation_type = 'main'"
                f"{workspace_clause} ORDER BY updated_at DESC",
                parameters,
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def visible_message_count(self, conversation_id: str) -> int:
        """Count user/assistant bubbles without loading timeline content."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT kind, metadata
                FROM timeline_events
                WHERE conversation_id = ?
                ORDER BY sequence ASC
                """,
                (conversation_id,),
            ).fetchall()

        count = 0
        assistant_turn_open = False
        for row in rows:
            kind = str(row["kind"])
            if kind == "user":
                count += 1
                assistant_turn_open = False
                continue
            if kind != "assistant":
                continue
            if not assistant_turn_open:
                count += 1
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            assistant_turn_open = bool(
                isinstance(metadata, dict) and metadata.get("tool_calls")
            )
        return count

    def delete_conversation(self, conversation_id: str) -> None:
        with self._write_transaction():
            self._connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def conversation_subtree(
        self,
        conversation_id: str,
    ) -> tuple[ConversationRecord, ...]:
        """Return a root conversation and every recursive child with its workspace."""

        with self._lock:
            rows = self._connection.execute(
                """
                WITH RECURSIVE subtree(id) AS (
                    SELECT id FROM conversations WHERE id = ?
                    UNION
                    SELECT child.id
                    FROM conversations AS child
                    JOIN subtree ON child.parent_conversation_id = subtree.id
                )
                SELECT conversations.*
                FROM conversations
                JOIN subtree ON subtree.id = conversations.id
                ORDER BY CASE WHEN conversations.id = ? THEN 0 ELSE 1 END,
                         conversations.created_at,
                         conversations.id
                """,
                (conversation_id, conversation_id),
            ).fetchall()
        if not rows:
            raise KeyError("Conversation was not found.")
        return tuple(_record_from_row(row) for row in rows)

    def child_conversation_ids(self, conversation_id: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id FROM conversations WHERE parent_conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        return tuple(str(row["id"]) for row in rows)

    def update_conversation_status(self, conversation_id: str, status: str) -> None:
        if status not in {"active", "completed", "error", "cancelled"}:
            raise ValueError("Invalid conversation status.")
        with self._write_transaction():
            self._connection.execute(
                "UPDATE conversations SET status = ?, updated_at = ? WHERE id = ?",
                (status, self._next_conversation_timestamp(), conversation_id),
            )

    def append_event(
        self,
        conversation_id: str,
        kind: TimelineEventKind,
        content: str,
        *,
        call_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TimelineEvent:
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM timeline_events WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            encoded_metadata = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) if metadata is not None else None
            self._connection.execute(
                """
                INSERT INTO timeline_events (conversation_id, sequence, kind, content, call_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, sequence, kind, content, call_id, encoded_metadata),
            )
            self._connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (self._next_conversation_timestamp(), conversation_id),
            )
        return TimelineEvent(sequence, kind, content, call_id, metadata)

    def load_events(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
    ) -> list[TimelineEvent]:
        if after_sequence < 0 or (
            through_sequence is not None and through_sequence < after_sequence
        ):
            raise ValueError("Invalid timeline sequence boundary.")
        upper_clause = " AND sequence <= ?" if through_sequence is not None else ""
        parameters: tuple[object, ...] = (
            (conversation_id, after_sequence, through_sequence)
            if through_sequence is not None
            else (conversation_id, after_sequence)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT sequence, kind, content, call_id, metadata
                FROM timeline_events
                WHERE conversation_id = ? AND sequence > ?{upper_clause}
                ORDER BY sequence ASC
                """,
                parameters,
            ).fetchall()
        return [_timeline_event_from_row(row) for row in rows]

    def load_events_by_sequences(
        self,
        conversation_id: str,
        sequences: tuple[int, ...],
    ) -> list[TimelineEvent]:
        """Load a small explicit event set without scanning covered timeline content."""

        unique = tuple(sorted(set(sequences)))
        if any(
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            for sequence in unique
        ):
            raise ValueError("Timeline sequences must be positive integers.")
        if not unique:
            return []
        placeholders = ",".join("?" for _ in unique)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT sequence, kind, content, call_id, metadata
                FROM timeline_events
                WHERE conversation_id = ? AND sequence IN ({placeholders})
                ORDER BY sequence ASC
                """,
                (conversation_id, *unique),
            ).fetchall()
        return [_timeline_event_from_row(row) for row in rows]

    def append_clear_boundary(self, conversation_id: str) -> TimelineEvent:
        """Advance active model context without deleting the durable timeline."""

        return self.append_event(
            conversation_id,
            "system",
            "",
            metadata={"context_boundary": "clear"},
        )

    def append_skill_activation(
        self,
        conversation_id: str,
        skill_name: str,
        snapshot: dict[str, object],
    ) -> TimelineEvent:
        return self.append_event(
            conversation_id,
            "skill_activation",
            skill_name,
            metadata=dict(snapshot),
        )

    def load_active_skill_events(self, conversation_id: str) -> list[TimelineEvent]:
        boundary = self.latest_clear_sequence(conversation_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT sequence, kind, content, call_id, metadata
                FROM timeline_events
                WHERE conversation_id = ? AND sequence > ? AND kind = 'skill_activation'
                ORDER BY sequence ASC
                """,
                (conversation_id, boundary),
            ).fetchall()
        latest: dict[str, TimelineEvent] = {}
        order: list[str] = []
        for row in rows:
            event = _timeline_event_from_row(row)
            if event.content not in latest:
                order.append(event.content)
            latest[event.content] = event
        return [latest[name] for name in order]

    def latest_clear_sequence(self, conversation_id: str) -> int:
        encoded_boundary = json.dumps(
            {"context_boundary": "clear"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS boundary
                FROM timeline_events
                WHERE conversation_id = ? AND kind = 'system' AND metadata = ?
                """,
                (conversation_id, encoded_boundary),
            ).fetchone()
        return int(row["boundary"])

    def append_context_summary(
        self,
        conversation_id: str,
        content: str,
        *,
        through_sequence: int,
        preserved_user_sequences: tuple[int, ...],
        trigger: str,
        estimated_before: int,
        estimated_after: int,
        format_version: int,
    ) -> TimelineEvent:
        if (
            through_sequence < 0
            or any(sequence < 0 or sequence > through_sequence for sequence in preserved_user_sequences)
            or trigger not in _SUMMARY_TRIGGERS
            or estimated_before < 0
            or estimated_after < 0
            or format_version <= 0
        ):
            raise ValueError("Invalid context summary metadata.")
        return self.append_event(
            conversation_id,
            "summary",
            content,
            metadata={
                "through_sequence": through_sequence,
                "preserved_user_sequences": list(preserved_user_sequences),
                "trigger": trigger,
                "estimated_before": estimated_before,
                "estimated_after": estimated_after,
                "format_version": format_version,
            },
        )

    def load_latest_context_summary(self, conversation_id: str) -> TimelineEvent | None:
        boundary = self.latest_clear_sequence(conversation_id)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT sequence, kind, content, call_id, metadata
                FROM timeline_events
                WHERE conversation_id = ? AND sequence > ? AND kind = 'summary'
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (conversation_id, boundary),
            ).fetchone()
        return _timeline_event_from_row(row) if row is not None else None

    def append_context_diagnostic(
        self,
        conversation_id: str,
        metadata: dict[str, object],
    ) -> TimelineEvent:
        _validate_context_diagnostic(metadata)
        return self.append_event(
            conversation_id,
            "context_diagnostic",
            "",
            metadata=dict(metadata),
        )

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
        with self._write_transaction():
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    profile_name TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    conversation_type TEXT NOT NULL DEFAULT 'main',
                    parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    skill_name TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    agent_name TEXT
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline_events (
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    call_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (conversation_id, sequence)
                )
                """
            )
            columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            migrations = {
                "conversation_type": "ALTER TABLE conversations ADD COLUMN conversation_type TEXT NOT NULL DEFAULT 'main'",
                "parent_conversation_id": "ALTER TABLE conversations ADD COLUMN parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE",
                "skill_name": "ALTER TABLE conversations ADD COLUMN skill_name TEXT",
                "status": "ALTER TABLE conversations ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
                "agent_name": "ALTER TABLE conversations ADD COLUMN agent_name TEXT",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    self._connection.execute(statement)

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        """Serialize read-modify-write operations across SQLite connections."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _next_conversation_timestamp(self) -> int:
        row = self._connection.execute(
            "SELECT MAX(CASE WHEN created_at > updated_at THEN created_at ELSE updated_at END) AS latest FROM conversations"
        ).fetchone()
        latest = int(row["latest"]) if row is not None and row["latest"] is not None else 0
        return max(time_ns(), latest + 1)


def _record_from_row(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        row["id"],
        row["title"],
        Path(row["workspace"]),
        row["profile_name"],
        int(row["created_at"]),
        int(row["updated_at"]),
        row["conversation_type"],
        row["parent_conversation_id"],
        row["skill_name"],
        row["status"],
        row["agent_name"],
    )


def _timeline_event_from_row(row: sqlite3.Row) -> TimelineEvent:
    return TimelineEvent(
        int(row["sequence"]),
        row["kind"],
        row["content"],
        row["call_id"],
        json.loads(row["metadata"]) if row["metadata"] is not None else None,
    )


def _validate_context_diagnostic(metadata: dict[str, object]) -> None:
    allowed = _DIAGNOSTIC_INTEGER_FIELDS | set(_DIAGNOSTIC_ENUM_FIELDS)
    if set(metadata) - allowed:
        raise ValueError("Context diagnostic contains a non-whitelisted field.")
    for name in _DIAGNOSTIC_INTEGER_FIELDS & set(metadata):
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Context diagnostic contains an invalid numeric value.")
    for name, values in _DIAGNOSTIC_ENUM_FIELDS.items():
        if name in metadata and metadata[name] not in values:
            raise ValueError("Context diagnostic contains an invalid category.")
