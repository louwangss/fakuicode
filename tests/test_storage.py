from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest


def test_store_serializes_event_sequences_across_independent_connections(
    tmp_path: Path,
) -> None:
    from fakuicode.storage import ConversationStore

    database = tmp_path / "concurrent.sqlite3"
    owner = ConversationStore(database)
    conversation = owner.create_conversation("Concurrent", tmp_path, "default")
    stores = [ConversationStore(database) for _ in range(8)]
    start = Barrier(len(stores))

    def append_batch(worker: int, store: ConversationStore) -> None:
        start.wait()
        for item in range(50):
            store.append_event(
                conversation.id,
                "user",
                f"worker-{worker}-event-{item}",
            )

    try:
        with ThreadPoolExecutor(max_workers=len(stores)) as executor:
            futures = [
                executor.submit(append_batch, worker, store)
                for worker, store in enumerate(stores)
            ]
            for future in futures:
                future.result()

        events = owner.load_events(conversation.id)
        assert [event.sequence for event in events] == list(range(1, 401))
        assert {event.content for event in events} == {
            f"worker-{worker}-event-{item}"
            for worker in range(8)
            for item in range(50)
        }
    finally:
        for store in stores:
            store.close()
        owner.close()


def test_store_migrates_existing_database_and_hides_skill_children(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, workspace TEXT NOT NULL,
            profile_name TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
        );
        CREATE TABLE timeline_events (
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
            call_id TEXT, metadata TEXT, PRIMARY KEY (conversation_id, sequence)
        );
        INSERT INTO conversations VALUES ('legacy', 'Legacy', '.', 'default', 1, 1);
        """
    )
    connection.close()

    store = ConversationStore(database)
    parent = store.get_conversation("legacy")
    child = store.create_conversation(
        "Hidden Skill",
        tmp_path,
        "default",
        conversation_type="skill",
        parent_conversation_id=parent.id,
        skill_name="test",
    )

    assert [item.id for item in store.list_conversations()] == [parent.id]
    assert store.get_conversation(child.id).conversation_type == "skill"
    store.delete_conversation(parent.id)
    with pytest.raises(KeyError):
        store.get_conversation(child.id)


def test_store_persists_agent_children_without_listing_them_as_main_sessions(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "store.sqlite3")
    parent = store.create_conversation("Main", tmp_path, "default")

    child = store.create_conversation(
        "Agent: explore",
        tmp_path,
        "default",
        conversation_type="agent",
        parent_conversation_id=parent.id,
        agent_name="explore",
    )

    restored = store.get_conversation(child.id)
    assert restored.conversation_type == "agent"
    assert restored.parent_conversation_id == parent.id
    assert restored.agent_name == "explore"
    assert [item.id for item in store.list_conversations()] == [parent.id]


def test_store_persists_skill_activation_events_after_current_clear_boundary(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "store.sqlite3")
    conversation = store.create_conversation("Main", tmp_path, "default")
    store.append_skill_activation(conversation.id, "one", {"fingerprint": "a"})
    store.append_clear_boundary(conversation.id)
    store.append_skill_activation(conversation.id, "two", {"fingerprint": "b"})

    events = store.load_active_skill_events(conversation.id)

    assert [event.content for event in events] == ["two"]


def test_store_restores_ordered_timeline_events_after_reopening(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    database = tmp_path / "fakuicode.sqlite3"
    store = ConversationStore(database)
    conversation = store.create_conversation("First task", tmp_path, "default")
    store.append_event(conversation.id, "user", "hello")
    store.append_event(conversation.id, "tool_result", "done", call_id="call-1", metadata={"summary": "read file"})
    store.close()

    reopened = ConversationStore(database)
    restored = reopened.get_conversation(conversation.id)
    events = reopened.load_events(conversation.id)

    assert restored.title == "First task"
    assert restored.profile_name == "default"
    assert [(event.sequence, event.kind, event.content) for event in events] == [
        (1, "user", "hello"),
        (2, "tool_result", "done"),
    ]
    assert events[1].metadata == {"summary": "read file"}


def test_store_lists_recent_conversations_and_deletes_only_the_requested_one(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    first = store.create_conversation("First", tmp_path, "default")
    second = store.create_conversation("Second", tmp_path, "fallback")
    store.append_event(first.id, "user", "older")
    store.append_event(second.id, "user", "newer")

    assert [item.id for item in store.list_conversations()] == [second.id, first.id]

    store.delete_conversation(first.id)

    assert [item.id for item in store.list_conversations()] == [second.id]
    assert store.load_events(second.id)[0].content == "newer"


def test_store_titles_a_default_conversation_from_its_first_user_prompt_without_reordering(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    conversation = store.create_conversation("New conversation", tmp_path, "default")
    original_updated_at = conversation.updated_at

    titled = store.ensure_conversation_title(
        conversation.id,
        "  Diagnose   the disappearing\ninput text  ",
    )
    unchanged = store.ensure_conversation_title(conversation.id, "A later prompt")

    assert titled.title == "Diagnose the disappearing input text"
    assert titled.updated_at == original_updated_at
    assert unchanged.title == titled.title


def test_store_can_backfill_a_default_title_from_the_first_saved_user_message(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    conversation = store.create_conversation("New conversation", tmp_path, "default")
    store.append_event(conversation.id, "user", "First saved question")
    store.append_event(conversation.id, "user", "Later question")

    store.backfill_default_conversation_titles()
    titled = store.get_conversation(conversation.id)

    assert titled.title == "First saved question"


def test_store_skips_skill_invocations_when_backfilling_a_title(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    conversation = store.create_conversation("New conversation", tmp_path, "default")
    store.append_event(
        conversation.id,
        "user",
        "/review",
        metadata={"skill_invocation": {"name": "review", "arguments": ""}},
    )
    store.append_event(conversation.id, "user", "First ordinary question")

    store.backfill_default_conversation_titles()

    assert store.get_conversation(conversation.id).title == "First ordinary question"


def test_default_store_path_uses_the_fakuicode_home_directory(tmp_path: Path) -> None:
    from fakuicode.storage import default_store_path

    assert default_store_path(tmp_path) == tmp_path / ".fakuicode" / "conversations.sqlite3"


def test_store_loads_events_with_inclusive_sequence_boundaries(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    conversation = store.create_conversation("Boundaries", tmp_path, "default")
    for content in ("one", "two", "three", "four"):
        store.append_event(conversation.id, "user", content)

    events = store.load_events(conversation.id, after_sequence=1, through_sequence=3)

    assert [(event.sequence, event.content) for event in events] == [(2, "two"), (3, "three")]


def test_store_returns_only_the_latest_summary_after_the_last_clear_boundary(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    conversation = store.create_conversation("Summary", tmp_path, "default")
    user = store.append_event(conversation.id, "user", "original user text")
    old_summary = store.append_context_summary(
        conversation.id,
        "old summary",
        through_sequence=user.sequence,
        preserved_user_sequences=(user.sequence,),
        trigger="automatic",
        estimated_before=100,
        estimated_after=50,
        format_version=1,
    )
    boundary = store.append_clear_boundary(conversation.id)
    later_user = store.append_event(conversation.id, "user", "later user text")
    latest = store.append_context_summary(
        conversation.id,
        "latest summary",
        through_sequence=later_user.sequence,
        preserved_user_sequences=(later_user.sequence,),
        trigger="manual",
        estimated_before=90,
        estimated_after=40,
        format_version=1,
    )

    assert store.latest_clear_sequence(conversation.id) == boundary.sequence
    assert store.load_latest_context_summary(conversation.id) == latest
    assert old_summary.content == "old summary"
    assert [event.content for event in store.load_events(conversation.id) if event.kind == "user"] == [
        "original user text",
        "later user text",
    ]
    assert latest.metadata == {
        "through_sequence": later_user.sequence,
        "preserved_user_sequences": [later_user.sequence],
        "trigger": "manual",
        "estimated_before": 90,
        "estimated_after": 40,
        "format_version": 1,
    }


def test_context_diagnostics_accept_only_non_content_whitelisted_fields(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "fakuicode.sqlite3")
    conversation = store.create_conversation("Diagnostics", tmp_path, "default")
    diagnostic = store.append_context_diagnostic(
        conversation.id,
        {
            "trigger": "automatic",
            "result": "compacted",
            "estimated_before": 116_000,
            "estimated_after": 12_000,
            "artifact_count": 2,
            "artifact_bytes": 48_000,
            "duration_ms": 125,
            "consecutive_failures": 0,
            "error_category": "none",
        },
    )

    assert diagnostic.kind == "context_diagnostic"
    assert diagnostic.content == ""
    assert diagnostic.metadata is not None
    assert "secret-marker" not in str(diagnostic.metadata)

    with pytest.raises(ValueError, match="diagnostic"):
        store.append_context_diagnostic(
            conversation.id,
            {"trigger": "automatic", "raw_response": "secret-marker"},
        )
    with pytest.raises(ValueError, match="diagnostic"):
        store.append_context_diagnostic(
            conversation.id,
            {"trigger": "secret-marker"},
        )


def test_visible_message_count_matches_tui_bubbles_without_derived_events(
    tmp_path: Path,
) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "history.sqlite3")
    conversation = store.create_conversation("count", tmp_path, "default")
    assert store.visible_message_count(conversation.id) == 0

    store.append_event(conversation.id, "user", "question")
    store.append_event(
        conversation.id,
        "assistant",
        "I will inspect",
        metadata={"tool_calls": [{"id": "call-1", "name": "read_file", "arguments": {}}]},
    )
    store.append_event(conversation.id, "tool_call", "read_file", call_id="call-1")
    store.append_event(conversation.id, "tool_result", "large result", call_id="call-1")
    store.append_event(conversation.id, "assistant", "final answer")
    store.append_event(conversation.id, "summary", "derived summary")
    store.append_context_diagnostic(
        conversation.id,
        {"trigger": "manual", "result": "noop"},
    )
    store.append_clear_boundary(conversation.id)

    assert store.visible_message_count(conversation.id) == 2
