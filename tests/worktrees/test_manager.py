from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
from uuid import UUID

import pytest

from fakuicode.worktrees.git import GitRunner
from fakuicode.worktrees.manager import (
    WorktreeManager,
    WorktreeRecoveryConflictError,
    WorktreeUnavailableError,
)
from fakuicode.worktrees.models import WorktreeIdentity


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Worktree Tests")
    _git(repo, "config", "user.email", "worktrees@example.test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


def _unlock_process_lease(lease) -> None:
    handle = lease._lock_handle
    assert handle is not None
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    lease._lock_handle = None


def _expire(lease) -> datetime:
    state = json.loads(lease.state_path.read_text(encoding="utf-8"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    state["last_used_at"] = (cutoff - timedelta(seconds=1)).isoformat()
    lease.state_path.write_text(json.dumps(state), encoding="utf-8")
    return cutoff


def test_manager_discovers_repo_root_and_maps_a_project_subdirectory(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    project = repo / "packages" / "cli"
    project.mkdir(parents=True)

    manager = WorktreeManager(project)

    assert manager.repo_root == repo.resolve()
    assert manager.project_workspace == project.resolve()
    assert manager.project_relative == Path("packages/cli")


def test_manager_preserves_user_excludes_and_guarantees_managed_roots_are_ignored(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("# user rule\n/local-only.txt\n", encoding="utf-8")

    manager = WorktreeManager(repo)
    manager.ensure_managed_roots_ignored()

    content = exclude.read_text(encoding="utf-8")
    assert content.startswith("# user rule\n/local-only.txt\n")
    assert content.count("# BEGIN fakuicode managed worktrees v1") == 1
    assert "/.fakuicode/worktrees/" in content
    assert "/.fakuicode/worktree-state/" in content
    assert _git(
        repo,
        "check-ignore",
        "--no-index",
        ".fakuicode/worktrees/probe",
    )
    assert _git(
        repo,
        "check-ignore",
        "--no-index",
        ".fakuicode/worktree-state/probe.json",
    )


def test_manager_rejects_tracked_files_under_a_managed_root(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    tracked = repo / ".fakuicode" / "worktrees" / "keep.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "-f", ".fakuicode/worktrees/keep.txt")
    _git(repo, "commit", "-m", "tracked manager file")

    with pytest.raises(WorktreeUnavailableError):
        WorktreeManager(repo).ensure_managed_roots_ignored()


def test_manager_rejects_reversed_managed_exclude_markers(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text(
        "# END fakuicode managed worktrees v1\n"
        "# BEGIN fakuicode managed worktrees v1\n",
        encoding="utf-8",
    )

    with pytest.raises(WorktreeUnavailableError):
        WorktreeManager(repo).ensure_managed_roots_ignored()


def test_create_and_release_an_unchanged_worktree(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("12345678-1234-5678-9abc-123456789abc")
    )
    manager = WorktreeManager(repo)

    lease = manager.create(identity)

    assert lease.worktree_root == (
        repo
        / ".fakuicode"
        / "worktrees"
        / "fork"
        / "12345678-1234-5678-9abc-123456789abc"
    ).resolve()
    assert lease.execution_workspace == lease.worktree_root
    assert lease.branch == identity.branch
    assert lease.base_sha == _git(repo, "rev-parse", "HEAD")
    assert (lease.worktree_root / "README.md").read_text(encoding="utf-8") == "base\n"

    state_path = (
        repo
        / ".fakuicode"
        / "worktree-state"
        / "12345678-1234-5678-9abc-123456789abc.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "active"
    assert state["branch"] == identity.branch
    assert state["base_sha"] == lease.base_sha

    report = manager.release(lease)

    assert report.status == "removed"
    assert report.removed is True
    assert not lease.worktree_root.exists()
    assert _git(repo, "branch", "--list", identity.branch) == ""
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "removed"


def test_existing_managed_worktree_recovers_without_running_git(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    )
    first_manager = WorktreeManager(repo)
    first = first_manager.create(identity)
    _unlock_process_lease(first)

    second_manager = WorktreeManager(repo)

    class NoGit:
        def run(self, *args, **kwargs):
            raise AssertionError("existing-directory recovery must not run git")

    second_manager.git = NoGit()
    recovered = second_manager.create(identity)

    assert recovered.worktree_root == first.worktree_root
    assert recovered.branch == first.branch
    assert recovered.base_sha == first.base_sha

    second_manager.git = GitRunner()
    assert second_manager.release(recovered).removed is True


def test_existing_directory_without_matching_sidecar_is_never_adopted(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    )
    target = repo / ".fakuicode" / "worktrees" / identity.relative_path
    target.mkdir(parents=True)

    with pytest.raises(Exception, match="恢复|状态|托管"):
        WorktreeManager(repo).create(identity)

    assert target.is_dir()


def test_recovery_rejects_a_sidecar_inventory_path_escape(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("10101010-2020-3030-4040-505050505050")
    )
    lease = WorktreeManager(repo).create(identity)
    _unlock_process_lease(lease)
    state = json.loads(lease.state_path.read_text(encoding="utf-8"))
    state["initialization"]["copies"] = [
        {"path": "../../outside", "sha256": "0" * 64, "size": 0}
    ]
    lease.state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(WorktreeRecoveryConflictError) as captured:
        WorktreeManager(repo).create(identity)

    assert captured.value.code == "worktree_recovery_conflict"
    assert lease.worktree_root.is_dir()


def test_include_copy_and_dependency_link_are_initialized_and_removed_safely(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    (repo / ".gitignore").write_text(".env\nnode_modules/\n", encoding="utf-8")
    (repo / ".worktreeinclude").write_text(".env\n", encoding="utf-8")
    (repo / ".worktreelinks").write_text("node_modules\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", ".worktreeinclude", ".worktreelinks")
    _git(repo, "commit", "-m", "worktree manifests")
    (repo / ".env").write_text("LOCAL_VALUE=test\n", encoding="utf-8")
    dependency = repo / "node_modules"
    dependency.mkdir()
    (dependency / "package.txt").write_text("shared\n", encoding="utf-8")
    identity = WorktreeIdentity.for_fork(
        UUID("11111111-2222-3333-4444-555555555555")
    )
    manager = WorktreeManager(repo)

    lease = manager.create(identity)

    copied = lease.worktree_root / ".env"
    linked = lease.worktree_root / "node_modules"
    assert copied.read_text(encoding="utf-8") == "LOCAL_VALUE=test\n"
    assert linked.resolve() == dependency.resolve()
    assert (linked / "package.txt").read_text(encoding="utf-8") == "shared\n"
    state = json.loads(lease.state_path.read_text(encoding="utf-8"))
    assert state["initialization"]["copies"][0]["path"] == ".env"
    assert state["initialization"]["links"][0]["path"] == "node_modules"

    report = manager.release(lease)

    assert report.removed is True
    assert dependency.is_dir()
    assert (dependency / "package.txt").exists()


def test_changed_include_copy_causes_fail_closed_retention(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    (repo / ".worktreeinclude").write_text(".env\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", ".worktreeinclude")
    _git(repo, "commit", "-m", "worktree include")
    (repo / ".env").write_text("before\n", encoding="utf-8")
    identity = WorktreeIdentity.for_fork(
        UUID("99999999-8888-7777-6666-555555555555")
    )
    manager = WorktreeManager(repo)
    lease = manager.create(identity)
    (lease.worktree_root / ".env").write_text("after\n", encoding="utf-8")

    report = manager.release(lease)

    assert report.status == "retained"
    assert lease.worktree_root.is_dir()
    assert _git(repo, "branch", "--list", identity.branch)


def test_invalid_initialization_rolls_back_only_the_new_worktree(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    (repo / ".worktreelinks").write_text("missing-dependency\n", encoding="utf-8")
    _git(repo, "add", ".worktreelinks")
    _git(repo, "commit", "-m", "invalid worktree links")
    identity = WorktreeIdentity.for_fork(
        UUID("45454545-6767-8989-abab-cdcdcdcdcdcd")
    )
    manager = WorktreeManager(repo)

    with pytest.raises(Exception, match="Worktree"):
        manager.create(identity)

    target = repo / ".fakuicode" / "worktrees" / identity.relative_path
    state_path = (
        repo / ".fakuicode" / "worktree-state" / f"{identity.session_id}.json"
    )
    assert not target.exists()
    assert _git(repo, "branch", "--list", identity.branch) == ""
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "removed"
    assert (repo / "README.md").read_text(encoding="utf-8") == "base\n"


def test_a_new_commit_is_retained_even_when_the_working_tree_is_clean(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb")
    )
    manager = WorktreeManager(repo)
    lease = manager.create(identity)
    (lease.worktree_root / "README.md").write_text("child\n", encoding="utf-8")
    _git(lease.worktree_root, "add", "README.md")
    _git(lease.worktree_root, "commit", "-m", "child change")

    report = manager.release(lease)

    assert report.status == "retained"
    assert lease.worktree_root.is_dir()


def test_sweeper_removes_an_expired_pristine_abandoned_worktree(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("12121212-3434-5656-7878-909090909090")
    )
    first = WorktreeManager(repo)
    lease = first.create(identity)
    cutoff = _expire(lease)
    _unlock_process_lease(lease)

    reports = WorktreeManager(repo).sweep_stale(cutoff)

    assert len(reports) == 1
    assert reports[0].removed is True
    assert not lease.worktree_root.exists()
    assert _git(repo, "branch", "--list", identity.branch) == ""


def test_sweeper_skips_a_live_cross_process_lease(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("abababab-cdcd-efef-1212-343434343434")
    )
    lease = WorktreeManager(repo).create(identity)
    cutoff = _expire(lease)

    reports = WorktreeManager(repo).sweep_stale(cutoff)

    assert reports == ()
    assert lease.worktree_root.is_dir()
    assert WorktreeManager(repo).release(lease).removed is True


def test_sweeper_retains_expired_unpushed_commits(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    identity = WorktreeIdentity.for_fork(
        UUID("cdcdcdcd-efef-1212-3434-565656565656")
    )
    manager = WorktreeManager(repo)
    lease = manager.create(identity)
    (lease.worktree_root / "README.md").write_text("child\n", encoding="utf-8")
    _git(lease.worktree_root, "add", "README.md")
    _git(lease.worktree_root, "commit", "-m", "child")
    assert manager.release(lease).status == "retained"
    cutoff = _expire(lease)

    reports = manager.sweep_stale(cutoff)

    assert len(reports) == 1
    assert reports[0].status == "retained"
    assert lease.worktree_root.is_dir()


def test_sweeper_removes_pushed_worktree_but_preserves_its_branch(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "HEAD")
    identity = WorktreeIdentity.for_fork(
        UUID("efefefef-1212-3434-5656-787878787878")
    )
    manager = WorktreeManager(repo)
    lease = manager.create(identity)
    (lease.worktree_root / "README.md").write_text("published\n", encoding="utf-8")
    _git(lease.worktree_root, "add", "README.md")
    _git(lease.worktree_root, "commit", "-m", "published")
    _git(lease.worktree_root, "push", "-u", "origin", identity.branch)
    assert manager.release(lease).status == "retained"
    cutoff = _expire(lease)

    reports = manager.sweep_stale(cutoff)

    assert len(reports) == 1
    assert reports[0].removed is True
    assert not lease.worktree_root.exists()
    assert _git(repo, "branch", "--list", identity.branch)
    state = json.loads(lease.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "branch_preserved"
