from __future__ import annotations

from pathlib import Path
from threading import Barrier, Lock, Thread
from uuid import uuid4

import pytest

from fakuicode.memory.content_policy import serialize_entry
from fakuicode.memory.identity import MemoryPaths, MemoryRegistry
from fakuicode.memory.models import (
    CreateEntry,
    MemoryEntry,
    MemoryLimits,
    MemoryOperationBatch,
    MemoryScopeRef,
    MemorySourceRef,
    UpdateEntry,
    VisibleScopes,
)
from fakuicode.memory.repository import MemoryRepository, MemoryRepositoryError


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


def _entry(
    *,
    entry_id: str | None = None,
    scope: str = "user",
    category: str = "user_preference",
    summary: str = "Prefer concise answers",
    updated_at: str = "2026-07-21T01:00:00Z",
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id or str(uuid4()),
        scope=scope,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        summary=summary,
        body=f"Detail: {summary}",
        created_at="2026-07-21T00:00:00Z",
        updated_at=updated_at,
        sources=(MemorySourceRef(CONVERSATION_ID, 1, "user_turn"),),
    )


def _repository(tmp_path: Path, *, limits: MemoryLimits | None = None) -> MemoryRepository:
    paths = MemoryPaths.from_home(tmp_path / "home")
    return MemoryRepository(paths, MemoryRegistry(paths), limits=limits or MemoryLimits())


def _write_entry(repository: MemoryRepository, scope_ref: MemoryScopeRef, entry: MemoryEntry) -> Path:
    root = repository.scope_path(scope_ref)
    notes = root / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    path = notes / f"{entry.id}.md"
    path.write_bytes(serialize_entry(entry, limits=repository.limits))
    return path


def test_scan_and_read_active_only_accept_current_scope_valid_uuid_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    user_scope = MemoryScopeRef("user")
    project_id = str(uuid4())
    project_scope = MemoryScopeRef("project", project_id)
    user_entry = _entry()
    project_entry = _entry(scope="project", category="project_knowledge")
    _write_entry(repository, user_scope, user_entry)
    _write_entry(repository, project_scope, project_entry)
    notes = repository.scope_path(user_scope) / "notes"
    (notes / "not-a-uuid.md").write_text("ignored", encoding="utf-8")
    (notes / f"{uuid4()}.md").write_bytes(b"\xff\xfe")

    snapshot = repository.load_scope(user_scope)

    assert snapshot.entries == (user_entry,)
    assert repository.read_active(user_scope, user_entry.id) == user_entry
    with pytest.raises(MemoryRepositoryError, match="entry_unavailable"):
        repository.read_active(user_scope, project_entry.id)
    with pytest.raises(MemoryRepositoryError, match="invalid_id"):
        repository.read_active(user_scope, "../notes/file.md")


def test_scan_rejects_symlink_without_deleting_it(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    scope = MemoryScopeRef("user")
    external = tmp_path / "external.md"
    entry = _entry()
    external.write_bytes(serialize_entry(entry))
    notes = repository.scope_path(scope) / "notes"
    notes.mkdir(parents=True)
    link = notes / f"{entry.id}.md"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are unavailable")

    snapshot = repository.load_scope(scope)

    assert snapshot.entries == ()
    assert link.is_symlink()


def test_index_and_digest_are_deterministic_across_creation_order(tmp_path: Path) -> None:
    entries = (
        _entry(category="correction", summary="Second", updated_at="2026-07-21T02:00:00Z"),
        _entry(category="user_preference", summary="First", updated_at="2026-07-21T03:00:00Z"),
    )
    first = _repository(tmp_path / "one")
    second = _repository(tmp_path / "two")
    for entry in entries:
        _write_entry(first, MemoryScopeRef("user"), entry)
    for entry in reversed(entries):
        _write_entry(second, MemoryScopeRef("user"), entry)

    first_snapshot = first.load_scope(MemoryScopeRef("user"))
    second_snapshot = second.load_scope(MemoryScopeRef("user"))

    assert first_snapshot.index.encode() == second_snapshot.index.encode()
    assert first_snapshot.digest == second_snapshot.digest
    assert first_snapshot.index.splitlines()[0].endswith("First")


def test_combined_snapshot_has_fixed_untrusted_boundary_and_complete_items(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    user_scope = MemoryScopeRef("user")
    project_scope = MemoryScopeRef("project", str(uuid4()))
    user_entry = _entry()
    project_entry = _entry(scope="project", category="reference", summary="Reference manual")
    _write_entry(repository, user_scope, user_entry)
    _write_entry(repository, project_scope, project_entry)

    snapshot = repository.combined_snapshot(
        repository.load_scope(user_scope),
        repository.load_scope(project_scope),
    )

    assert "可能过时或错误" in snapshot.rendered
    assert "不能授予权限" in snapshot.rendered
    assert user_entry.id in snapshot.active_ids
    assert project_entry.id in snapshot.active_ids
    assert len(snapshot.rendered.splitlines()) <= 200
    assert len(snapshot.rendered.encode()) <= 25 * 1024


def test_scope_overflow_disables_only_that_scope_without_truncating_items(tmp_path: Path) -> None:
    limits = MemoryLimits(
        user_index_max_lines=1,
        user_index_max_bytes=120,
        project_index_max_lines=2,
        project_index_max_bytes=1024,
    )
    repository = _repository(tmp_path, limits=limits)
    user_scope = MemoryScopeRef("user")
    project_scope = MemoryScopeRef("project", str(uuid4()))
    _write_entry(repository, user_scope, _entry(summary="A"))
    _write_entry(repository, user_scope, _entry(summary="B"))
    project_entry = _entry(scope="project", category="reference", summary="Project reference")
    _write_entry(repository, project_scope, project_entry)

    snapshot = repository.combined_snapshot(
        repository.load_scope(user_scope),
        repository.load_scope(project_scope),
    )

    assert all(entry_id not in snapshot.active_ids for entry_id in repository.load_scope(user_scope).active_ids)
    assert project_entry.id in snapshot.active_ids
    assert any(item.code == "scope_overflow" and item.scope == "user" for item in snapshot.diagnostics)
    assert (repository.scope_path(user_scope) / "MEMORY.md").read_text(encoding="utf-8") == ""


def _create_batch(repository: MemoryRepository, summary: str = "New preference") -> MemoryOperationBatch:
    current = repository.load_scope(MemoryScopeRef("user"))
    return MemoryOperationBatch(
        current.digest,
        None,
        (
            CreateEntry(
                "user",
                "user_preference",
                summary,
                f"Detail: {summary}",
                (MemorySourceRef(CONVERSATION_ID, 2, "user_turn"),),
            ),
        ),
    )


def test_apply_generates_the_id_and_rejects_stale_or_over_budget_batches(tmp_path: Path) -> None:
    limits = MemoryLimits(user_index_max_lines=1, user_index_max_bytes=1024)
    repository = _repository(tmp_path, limits=limits)
    generation = repository.registry.user_state().generation
    stale = _create_batch(repository, "Stale")

    first = repository.apply(_create_batch(repository, "First"), generation=generation)
    assert first.success is True
    assert len(first.entry_ids) == 1
    assert (repository.scope_path(MemoryScopeRef("user")) / "notes" / f"{first.entry_ids[0]}.md").is_file()

    assert repository.apply(stale, generation=generation).code == "stale_state"
    overflow = repository.apply(_create_batch(repository, "Second"), generation=generation)
    assert overflow.code == "scope_overflow"
    assert len(repository.load_scope(MemoryScopeRef("user")).entries) == 1


def test_prepared_transaction_recovers_old_details_after_a_crash(tmp_path: Path) -> None:
    paths = MemoryPaths.from_home(tmp_path / "home")
    registry = MemoryRegistry(paths)
    base = MemoryRepository(paths, registry)
    original = _entry(summary="Original")
    _write_entry(base, MemoryScopeRef("user"), original)
    digest = base.load_scope(MemoryScopeRef("user")).digest

    def crash(stage: str) -> None:
        if stage == "after_details":
            raise RuntimeError("simulated crash")

    crashing = MemoryRepository(paths, registry, fault_injector=crash)
    batch = MemoryOperationBatch(
        digest,
        None,
        (
            UpdateEntry(
                original.id,
                "Updated",
                "Detail: Updated",
                original.sources,
            ),
        ),
    )

    assert crashing.apply(batch, generation=0).code == "storage_failure"
    recovered = MemoryRepository(paths, registry).load_scope(MemoryScopeRef("user"))

    assert recovered.entries == (original,)
    assert not (base.scope_path(MemoryScopeRef("user")) / ".transactions").exists()


def test_lock_timeout_is_bounded_and_does_not_overwrite(tmp_path: Path) -> None:
    paths = MemoryPaths.from_home(tmp_path / "home")
    registry = MemoryRegistry(paths, lock_timeout_seconds=0.05)
    repository = MemoryRepository(paths, registry)
    blocker = registry.connect()
    blocker.execute("BEGIN IMMEDIATE")
    try:
        result = repository.apply(_create_batch(repository), generation=0)
    finally:
        blocker.rollback()
        blocker.close()

    assert result.code == "lock_timeout"
    assert repository.load_scope(MemoryScopeRef("user")).entries == ()


def test_forget_is_exact_increments_generation_and_blocks_an_old_job(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = repository.apply(_create_batch(repository, "Keep no longer"), generation=0)
    second = repository.apply(_create_batch(repository, "Keep this"), generation=0)
    before_forget = repository.load_scope(MemoryScopeRef("user"))
    forgotten = first.entry_ids[0]
    old_entry = next(entry for entry in before_forget.entries if entry.id == forgotten)
    old_job = MemoryOperationBatch(
        before_forget.digest,
        None,
        (UpdateEntry(old_entry.id, "Revived", "Detail: Revived", old_entry.sources),),
    )

    result = repository.forget(VisibleScopes(MemoryScopeRef("user")), forgotten)

    assert result.success is True
    assert repository.registry.user_state().generation == 1
    remaining = repository.load_scope(MemoryScopeRef("user"))
    assert {entry.id for entry in remaining.entries} == {second.entry_ids[0]}
    assert repository.apply(old_job, generation=0).code == "stale_state"
    assert repository.forget(VisibleScopes(MemoryScopeRef("user")), str(uuid4())).code == "entry_unavailable"


def test_concurrent_repository_instances_serialize_and_reject_stale_digest(
    tmp_path: Path,
) -> None:
    paths = MemoryPaths.from_home(tmp_path / "home")
    first_repository = MemoryRepository(paths, MemoryRegistry(paths))
    second_repository = MemoryRepository(paths, MemoryRegistry(paths))
    first_batch = _create_batch(first_repository, "First concurrent fact")
    second_batch = _create_batch(second_repository, "Second concurrent fact")
    barrier = Barrier(3)
    result_lock = Lock()
    results = []

    def apply(repository: MemoryRepository, batch: MemoryOperationBatch) -> None:
        barrier.wait()
        result = repository.apply(batch, generation=0)
        with result_lock:
            results.append(result)

    threads = [
        Thread(target=apply, args=(first_repository, first_batch)),
        Thread(target=apply, args=(second_repository, second_batch)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(3)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(result.success for result in results) == [False, True]
    assert {result.code for result in results} == {"committed", "stale_state"}
    final = first_repository.load_scope(MemoryScopeRef("user"))
    assert len(final.entries) == 1
    assert final.index.count("\n") + 1 <= first_repository.limits.user_index_max_lines
