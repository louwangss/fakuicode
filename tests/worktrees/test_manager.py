from __future__ import annotations

import json
from pathlib import Path
import subprocess
from uuid import UUID

from fakuicode.worktrees.manager import WorktreeManager
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
