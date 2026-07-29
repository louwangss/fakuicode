from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from io import BytesIO

import pytest


class _ChunkOnlyBytesIO(BytesIO):
    def read(self, size: int = -1) -> bytes:
        assert size > 0, "artifact capture must never read an unbounded stream"
        return super().read(size)


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


def test_writing_the_same_timeline_result_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    output = "same" * 10_000

    first = store.write_tool_result(source_sequence=4, output=output, success=True)

    def fail_unbounded_read(_path: Path) -> bytes:
        raise AssertionError("idempotence checks must not read the complete artifact")

    monkeypatch.setattr(Path, "read_bytes", fail_unbounded_read)
    second = store.write_tool_result(source_sequence=4, output=output, success=True)

    assert first == second
    assert first.newly_created is True
    assert second.newly_created is False
    assert len(list((tmp_path / ".fakuicode/context-artifacts/conversation-1").glob("*.txt"))) == 1


def test_command_streams_are_captured_incrementally_in_the_existing_output_format(
    tmp_path: Path,
) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    reference = store.write_command_result_streams(
        stdout=_ChunkOnlyBytesIO(b"head\n" + b"x" * 50_000 + b"\xfftail"),
        stderr=_ChunkOnlyBytesIO(b"warning\n"),
        exit_code=3,
        success=False,
    )

    artifact = tmp_path / reference.read_path
    output = artifact.read_text(encoding="utf-8")
    assert output.startswith("stdout:\nhead\n")
    assert "\ufffdtail\nstderr:\nwarning\n" in output
    assert output.endswith("\nexit_code: 3")
    assert reference.source_sequence == 0
    assert reference.byte_size == len(output.encode("utf-8"))
    assert reference.content_sha256 in artifact.name
    assert reference.success is False
    assert list(artifact.parent.glob(".staging-*.tmp")) == []
    assert list(artifact.parent.glob(".staging-*.lock")) == []


def test_command_staging_holds_a_companion_lock_until_the_artifact_is_claimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore
    from fakuicode.locking import FileLockPolicy, FileLockTimeoutError, KernelFileLock

    store = ContextArtifactStore(tmp_path, "conversation-1")

    def assert_locked_then_replace(source: Path, target: Path) -> None:
        token = source.name.removeprefix(".staging-").removesuffix(".tmp")
        lock_path = source.parent / f".staging-{token}.lock"
        assert lock_path.is_file()
        contender = KernelFileLock(
            lock_path,
            policy=FileLockPolicy(timeout_seconds=0),
        )
        with pytest.raises(FileLockTimeoutError):
            contender.acquire()
        ContextArtifactStore._atomic_replace(source, target)

    monkeypatch.setattr(store, "_atomic_replace", assert_locked_then_replace)

    store.write_command_result_streams(
        stdout=BytesIO(b"complete"),
        stderr=BytesIO(),
        exit_code=0,
        success=True,
    )

    assert list(store.conversation_dir.glob(".staging-*.tmp")) == []
    assert list(store.conversation_dir.glob(".staging-*.lock")) == []


def test_command_stream_capture_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    first = store.write_command_result_streams(
        stdout=BytesIO(b"same"),
        stderr=BytesIO(),
        exit_code=0,
        success=True,
    )
    second = store.write_command_result_streams(
        stdout=BytesIO(b"same"),
        stderr=BytesIO(),
        exit_code=0,
        success=True,
    )

    assert first == second
    assert first.newly_created is True
    assert second.newly_created is False
    assert len(list(store.conversation_dir.glob("command-*.txt"))) == 1


def test_failed_command_stream_claim_removes_the_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("injected command claim failure")

    monkeypatch.setattr(store, "_atomic_replace", fail_replace)

    with pytest.raises(OSError, match="injected command claim failure"):
        store.write_command_result_streams(
            stdout=BytesIO(b"complete"),
            stderr=BytesIO(),
            exit_code=0,
            success=True,
        )

    assert list(store.conversation_dir.glob(".staging-*.tmp")) == []
    assert list(store.conversation_dir.glob(".staging-*.lock")) == []
    assert list(store.conversation_dir.glob("command-*.txt")) == []


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


def test_resolving_an_artifact_streams_the_integrity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    store = ContextArtifactStore(tmp_path, "conversation-1")
    reference = store.write_tool_result(
        source_sequence=8,
        output="complete" * 10_000,
        success=True,
    )

    def fail_unbounded_read(_path: Path) -> bytes:
        raise AssertionError("artifact integrity checks must not read the complete file")

    monkeypatch.setattr(Path, "read_bytes", fail_unbounded_read)

    assert store.resolve_read_path(reference) == tmp_path / reference.read_path


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


def test_startup_cleanup_removes_only_unlocked_ephemeral_artifacts(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore

    active = ContextArtifactStore(tmp_path, f"ephemeral-{'a' * 32}")
    active_lease = active.acquire_ephemeral_lease()
    active.write_tool_result(source_sequence=1, output="active", success=True)
    orphaned = ContextArtifactStore(tmp_path, f"ephemeral-{'b' * 32}")
    orphaned_lease = orphaned.acquire_ephemeral_lease()
    orphaned.write_tool_result(source_sequence=1, output="orphaned", success=True)
    orphaned_lease.release()
    empty_unleased = ContextArtifactStore(tmp_path, f"ephemeral-{'c' * 32}")
    empty_unleased.conversation_dir.mkdir(parents=True)
    populated_unleased = ContextArtifactStore(tmp_path, f"ephemeral-{'d' * 32}")
    populated_unleased.write_tool_result(
        source_sequence=1,
        output="preserve legacy evidence",
        success=True,
    )

    try:
        cleaner = ContextArtifactStore(tmp_path, "conversation-1")
        assert cleaner.cleanup_orphaned_ephemeral_artifacts() == 2
        assert active.conversation_dir.is_dir()
        assert not orphaned.conversation_dir.exists()
        assert not empty_unleased.conversation_dir.exists()
        assert populated_unleased.conversation_dir.is_dir()
    finally:
        active_lease.release()


def test_startup_cleanup_removes_only_unlocked_staging_files(tmp_path: Path) -> None:
    from fakuicode.context_artifacts import ContextArtifactStore
    from fakuicode.locking import FileLockPolicy, KernelFileLock

    store = ContextArtifactStore(tmp_path, "conversation-1")
    store.conversation_dir.mkdir(parents=True)
    active_token = "a" * 32
    active_temporary = store.conversation_dir / f".staging-{active_token}.tmp"
    active_lock_path = store.conversation_dir / f".staging-{active_token}.lock"
    active_temporary.write_bytes(b"active")
    active_lock = KernelFileLock(
        active_lock_path,
        policy=FileLockPolicy(timeout_seconds=0),
    )
    active_lock.acquire()
    orphaned_token = "b" * 32
    orphaned_temporary = store.conversation_dir / f".staging-{orphaned_token}.tmp"
    orphaned_lock_path = store.conversation_dir / f".staging-{orphaned_token}.lock"
    orphaned_temporary.write_bytes(b"orphaned")
    orphaned_lock = KernelFileLock(
        orphaned_lock_path,
        policy=FileLockPolicy(timeout_seconds=0),
    )
    orphaned_lock.acquire()
    orphaned_lock.release()

    try:
        assert store.cleanup_orphaned_staging_files() == 1
        assert active_temporary.is_file()
        assert active_lock_path.is_file()
        assert not orphaned_temporary.exists()
        assert not orphaned_lock_path.exists()
    finally:
        active_lock.release()
