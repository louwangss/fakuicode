from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest


def test_artifact_write_preserves_complete_content_and_returns_workspace_relative_path(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    output = "head\n" + ("x" * 20_000) + "\ntail-marker"
    store = ContextArtifactStore(tmp_path, "conversation-1")

    reference = store.write_tool_result(source_sequence=7, output=output, success=True)

    artifact = tmp_path / reference.read_path
    assert artifact.read_text(encoding="utf-8") == output
    assert reference.source_sequence == 7
    assert reference.byte_size == len(output.encode("utf-8"))
    assert reference.success is True
    assert reference.read_path.startswith(".fakuicode/context-artifacts/conversation-1/")
    assert "tail-marker" in artifact.read_text(encoding="utf-8")


def test_artifact_filename_never_uses_an_untrusted_provider_call_id(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    reference = store.write_tool_result(
        source_sequence=3,
        output="complete",
        success=False,
        provider_call_id="../../escape/secret",
    )

    assert "escape" not in reference.read_path
    assert "secret" not in reference.read_path
    assert (tmp_path / reference.read_path).is_file()


def test_writing_the_same_timeline_result_is_idempotent(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")

    first = store.write_tool_result(source_sequence=4, output="same", success=True)
    second = store.write_tool_result(source_sequence=4, output="same", success=True)

    assert first == second
    assert first.newly_created is True
    assert second.newly_created is False
    assert len(list((tmp_path / ".fakuicode/context-artifacts/conversation-1").glob("*.txt"))) == 1


def test_failed_atomic_replace_returns_no_reference_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(store, "_atomic_replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        store.write_tool_result(source_sequence=5, output="must remain available to caller", success=True)

    conversation_dir = tmp_path / ".fakuicode/context-artifacts/conversation-1"
    assert list(conversation_dir.glob("*.tmp")) == []
    assert list(conversation_dir.glob("*.txt")) == []


@pytest.mark.parametrize("conversation_id", ["", "../escape", "a/b", "a\\b", ".", "x" * 129])
def test_artifact_store_rejects_unsafe_conversation_ids(tmp_path: Path, conversation_id: str) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    with pytest.raises(ValueError):
        ContextArtifactStore(tmp_path, conversation_id)


def test_artifact_store_rejects_a_symlinked_artifact_root_that_escapes_the_workspace(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / ".fakuicode").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(ValueError, match="escapes"):
        ContextArtifactStore(workspace, "conversation-1")


def test_resolving_an_artifact_reference_rechecks_conversation_path_and_integrity(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    reference = store.write_tool_result(source_sequence=8, output="complete", success=True)

    assert store.resolve_read_path(reference) == tmp_path / reference.read_path
    with pytest.raises(ValueError):
        store.resolve_read_path(replace(reference, read_path="../escape.txt"))

    (tmp_path / reference.read_path).write_text("tampered", encoding="utf-8")
    with pytest.raises(OSError, match="integrity"):
        store.resolve_read_path(reference)


def test_staged_conversation_deletion_can_be_restored_or_purged(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    reference = store.write_tool_result(source_sequence=9, output="complete", success=True)
    original = tmp_path / reference.read_path

    staged = store.stage_conversation_deletion()
    assert staged is not None and staged.is_dir()
    assert not original.exists()
    store.restore_staged_deletion(staged)
    assert original.is_file()

    staged = store.stage_conversation_deletion()
    assert staged is not None
    store.purge_staged_deletion(staged)
    assert not staged.exists()


def test_startup_cleanup_only_removes_validated_tombstones(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    store.write_tool_result(source_sequence=10, output="complete", success=True)
    staged = store.stage_conversation_deletion()
    other_store = ContextArtifactStore(tmp_path, "conversation-2")
    other_store.write_tool_result(source_sequence=11, output="other", success=True)
    other_staged = other_store.stage_conversation_deletion()
    unrelated = store.root / "keep-me"
    unrelated.mkdir()

    assert store.cleanup_stale_tombstones() == 2
    assert staged is not None and not staged.exists()
    assert other_staged is not None and not other_staged.exists()
    assert unrelated.is_dir()


def test_startup_cleanup_restores_tombstone_when_database_record_is_retained(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    reference = store.write_tool_result(source_sequence=10, output="complete", success=True)
    staged = store.stage_conversation_deletion()

    assert staged is not None
    assert store.cleanup_stale_tombstones(
        retained_conversation_ids={"conversation-1"},
    ) == 1
    assert not staged.exists()
    assert (tmp_path / reference.read_path).read_text(encoding="utf-8") == "complete"
