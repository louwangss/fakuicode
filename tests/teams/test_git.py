from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from fakuicode.teams.git import TeamGitCoordinator
from fakuicode.teams.models import BackendType, TaskStatus, TeamMember
from fakuicode.teams.service import TeamService
from fakuicode.teams.storage import TeamStore
from fakuicode.worktrees.manager import WorktreeManager


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
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
    _git(repo, "config", "user.name", "Team Tests")
    _git(repo, "config", "user.email", "team@example.test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


def _setup(tmp_path: Path) -> tuple[Path, TeamService, TeamGitCoordinator, TeamMember]:
    repo = _repository(tmp_path)
    service = TeamService(
        TeamStore(tmp_path / "teams"),
        lead_conversation_id="lead-conversation",
        repository_fingerprint=str(repo.resolve()).casefold(),
        target_branch=_git(repo, "branch", "--show-current"),
        target_sha=_git(repo, "rev-parse", "HEAD"),
        lead_profile="default",
    )
    team = service.create_team("alpha")
    alice = TeamMember.create(
        name="alice",
        role="实现",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="alice-conversation",
    )
    service.store.add_member(team.team_id, alice)
    coordinator = TeamGitCoordinator(service, WorktreeManager(repo))
    return repo, service, coordinator, alice


def test_next_task_worktree_starts_from_latest_integration_head(tmp_path: Path) -> None:
    _, service, coordinator, alice = _setup(tmp_path)
    first = service.create_task(service.actor(), title="first", description="")
    service.claim_task(service.actor(), first.task_id, alice.member_id)
    first_lease = coordinator.prepare_task(service.actor(), first.task_id)
    (first_lease.execution_workspace / "first.txt").write_text("first\n", encoding="utf-8")
    _git(first_lease.execution_workspace, "add", "first.txt")
    _git(first_lease.execution_workspace, "commit", "-m", "first")
    first_head = _git(first_lease.execution_workspace, "rev-parse", "HEAD")

    coordinator.record_completion(
        service.actor_for_member(alice.member_id),
        first.task_id,
        first_head,
        verification_summary="unit tests passed",
    )
    report = coordinator.integrate_task(service.actor(), first.task_id)

    assert report["status"] == "completed"
    assert service.store.get_task(service.actor().team_id, first.task_id).status is (
        TaskStatus.COMPLETED
    )

    second = service.create_task(service.actor(), title="second", description="")
    service.claim_task(service.actor(), second.task_id, alice.member_id)
    second_lease = coordinator.prepare_task(service.actor(), second.task_id)

    assert second_lease.base_sha == report["integration_sha"]
    assert (second_lease.execution_workspace / "first.txt").read_text(
        encoding="utf-8"
    ) == "first\n"


def test_conflicting_task_aborts_merge_and_preserves_integration_worktree(
    tmp_path: Path,
) -> None:
    _, service, coordinator, alice = _setup(tmp_path)
    first = service.create_task(service.actor(), title="first", description="")
    second = service.create_task(service.actor(), title="second", description="")
    service.claim_task(service.actor(), first.task_id, alice.member_id)
    first_lease = coordinator.prepare_task(service.actor(), first.task_id)
    bob = TeamMember.create(
        name="bob",
        role="实现",
        profile="default",
        backend=BackendType.IN_PROCESS,
        requires_plan_approval=False,
        conversation_id="bob-conversation",
    )
    service.store.add_member(service.actor().team_id, bob)
    service.claim_task(service.actor(), second.task_id, bob.member_id)
    second_lease = coordinator.prepare_task(service.actor(), second.task_id)

    for lease, content in ((first_lease, "first\n"), (second_lease, "second\n")):
        (lease.execution_workspace / "README.md").write_text(content, encoding="utf-8")
        _git(lease.execution_workspace, "add", "README.md")
        _git(lease.execution_workspace, "commit", "-m", content.strip())
    first_head = _git(first_lease.execution_workspace, "rev-parse", "HEAD")
    second_head = _git(second_lease.execution_workspace, "rev-parse", "HEAD")
    coordinator.record_completion(
        service.actor_for_member(alice.member_id),
        first.task_id,
        first_head,
        verification_summary="passed",
    )
    coordinator.record_completion(
        service.actor_for_member(bob.member_id),
        second.task_id,
        second_head,
        verification_summary="passed",
    )
    coordinator.integrate_task(service.actor(), first.task_id)

    conflict = coordinator.integrate_task(service.actor(), second.task_id)

    assert conflict["status"] == "integration_failed"
    assert service.store.get_task(service.actor().team_id, second.task_id).status is (
        TaskStatus.INTEGRATION_FAILED
    )
    integration = coordinator.integration_lease(service.actor())
    assert _git(integration.execution_workspace, "status", "--porcelain") == ""
    assert (integration.execution_workspace / "README.md").read_text(
        encoding="utf-8"
    ) == "first\n"


def test_finalization_requires_matching_token_clean_target_and_fast_forward(
    tmp_path: Path,
) -> None:
    repo, service, coordinator, alice = _setup(tmp_path)
    task = service.create_task(service.actor(), title="first", description="")
    service.claim_task(service.actor(), task.task_id, alice.member_id)
    lease = coordinator.prepare_task(service.actor(), task.task_id)
    (lease.execution_workspace / "first.txt").write_text("first\n", encoding="utf-8")
    _git(lease.execution_workspace, "add", "first.txt")
    _git(lease.execution_workspace, "commit", "-m", "first")
    head = _git(lease.execution_workspace, "rev-parse", "HEAD")
    coordinator.record_completion(
        service.actor_for_member(alice.member_id),
        task.task_id,
        head,
        verification_summary="passed",
    )
    coordinator.integrate_task(service.actor(), task.task_id)
    prepared = coordinator.prepare_finalization(service.actor())

    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError):
        coordinator.finalize(service.actor(), prepared["confirmation_token"])
    (repo / "dirty.txt").unlink()

    result = coordinator.finalize(
        service.actor(),
        prepared["confirmation_token"],
    )

    assert result["status"] == "finalized"
    assert _git(repo, "rev-parse", "HEAD") == prepared["integration_sha"]
