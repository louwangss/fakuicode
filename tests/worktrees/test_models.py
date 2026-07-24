from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from fakuicode.worktrees.models import WorktreeIdentity, WorktreeLimits


def test_role_identity_builds_fixed_safe_path_and_branch() -> None:
    session_id = UUID("12345678-1234-5678-9abc-123456789abc")

    identity = WorktreeIdentity.for_role(session_id, "code-reviewer")

    assert identity.relative_path == Path(
        "role-code-reviewer/12345678-1234-5678-9abc-123456789abc"
    )
    assert (
        identity.branch
        == "worktree/role-code-reviewer/12345678-1234-5678-9abc-123456789abc"
    )


def test_fork_identity_never_uses_a_model_supplied_name() -> None:
    session_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    identity = WorktreeIdentity.for_fork(session_id)

    assert identity.relative_path == Path(
        "fork/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    assert identity.branch == "worktree/fork/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.mark.parametrize(
    "role",
    ("", ".", "..", "CodeReviewer", "code_reviewer", "a/b", "a" * 33),
)
def test_role_identity_rejects_values_outside_the_existing_role_contract(role: str) -> None:
    with pytest.raises(ValueError):
        WorktreeIdentity.for_role(
            UUID("12345678-1234-5678-9abc-123456789abc"),
            role,
        )


def test_worktree_limits_have_the_confirmed_bounded_defaults() -> None:
    limits = WorktreeLimits()

    assert limits.manifest_bytes == 64 * 1024
    assert limits.manifest_line_bytes == 1024
    assert limits.max_links == 64
    assert limits.max_copy_files == 512
    assert limits.max_copy_file_bytes == 8 * 1024 * 1024
    assert limits.max_copy_total_bytes == 64 * 1024 * 1024
    assert limits.metadata_timeout_seconds == 30
    assert limits.lifecycle_timeout_seconds == 120
