from __future__ import annotations

import sqlite3
import subprocess
import os
from pathlib import Path

import pytest

from fakuicode.memory.identity import MemoryPaths, MemoryRegistry, ProjectIdentityResolver


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _init_repository(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "-c", "user.name=Test User", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "initial")


def test_registry_defaults_to_enabled_and_persists_generation_and_notice(tmp_path: Path) -> None:
    paths = MemoryPaths.from_home(tmp_path)
    registry = MemoryRegistry(paths)

    initial = registry.user_state()
    assert initial.enabled is True
    assert initial.notice_shown is False
    assert initial.generation == 0

    disabled = registry.set_enabled(False)
    assert disabled.enabled is False
    assert disabled.generation == 1
    assert MemoryRegistry(paths).user_state() == disabled

    shown = registry.mark_notice_shown()
    assert shown.notice_shown is True
    assert shown.generation == 1


def test_registry_schema_contains_no_conversation_or_memory_body_columns(tmp_path: Path) -> None:
    paths = MemoryPaths.from_home(tmp_path)
    MemoryRegistry(paths)

    with sqlite3.connect(paths.registry) as connection:
        columns = {
            row[1]
            for table in ("user_state", "projects", "scope_status")
            for row in connection.execute(f"PRAGMA table_info({table})")
        }

    forbidden = {"conversation", "message", "content", "body", "tool_output", "memory_text"}
    assert columns.isdisjoint(forbidden)


def test_registry_rejects_a_symlinked_memory_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    private = home / ".fakuicode"
    target = tmp_path / "outside"
    private.mkdir(parents=True)
    target.mkdir()
    try:
        os.symlink(target, private / "memory", target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable in this environment.")

    with pytest.raises(RuntimeError, match="unsafe_memory_directory"):
        MemoryRegistry(MemoryPaths.from_home(home))


def test_non_git_identity_is_stable_isolated_and_changes_after_move(tmp_path: Path) -> None:
    registry = MemoryRegistry(MemoryPaths.from_home(tmp_path / "home"))
    resolver = ProjectIdentityResolver(registry)
    first = tmp_path / "a" / "project"
    second = tmp_path / "b" / "project"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    first_identity = resolver.resolve(first)
    assert first_identity is not None
    assert resolver.resolve(first).project_id == first_identity.project_id  # type: ignore[union-attr]
    assert resolver.resolve(second).project_id != first_identity.project_id  # type: ignore[union-attr]
    assert first_identity.project_id not in {"project", str(first.resolve())}

    moved = tmp_path / "moved"
    first.rename(moved)
    assert resolver.resolve(moved).project_id != first_identity.project_id  # type: ignore[union-attr]


def test_regular_git_identity_is_stable_and_independent_clones_do_not_share(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repository(source)
    first_clone = tmp_path / "clone-a"
    second_clone = tmp_path / "clone-b"
    _git(tmp_path, "clone", str(source), str(first_clone))
    _git(tmp_path, "clone", str(source), str(second_clone))
    resolver = ProjectIdentityResolver(MemoryRegistry(MemoryPaths.from_home(tmp_path / "home")))

    source_identity = resolver.resolve(source)
    first_identity = resolver.resolve(first_clone)
    second_identity = resolver.resolve(second_clone)

    assert source_identity is not None
    assert source_identity.identity_kind == "git_common_dir"
    assert first_identity is not None and second_identity is not None
    assert len({source_identity.project_id, first_identity.project_id, second_identity.project_id}) == 3


def test_linked_worktree_shares_identity_only_with_a_valid_reverse_registration(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    _init_repository(repository)
    _git(repository, "worktree", "add", "-b", "test-worktree", str(worktree))
    resolver = ProjectIdentityResolver(MemoryRegistry(MemoryPaths.from_home(tmp_path / "home")))

    main_identity = resolver.resolve(repository)
    linked_identity = resolver.resolve(worktree)

    assert main_identity is not None and linked_identity is not None
    assert linked_identity.project_id == main_identity.project_id

    git_dir = Path(_git(worktree, "rev-parse", "--absolute-git-dir"))
    (git_dir / "gitdir").write_text(str(tmp_path / "forged"), encoding="utf-8")
    assert resolver.resolve(worktree) is None


def test_git_failure_does_not_fall_back_to_a_weak_project_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = MemoryRegistry(MemoryPaths.from_home(tmp_path / "home"))

    def broken_git(_: Path, __: tuple[str, ...]) -> bytes:
        raise OSError("git unavailable")

    resolver = ProjectIdentityResolver(registry, git_runner=broken_git)

    assert resolver.resolve(workspace) is None
