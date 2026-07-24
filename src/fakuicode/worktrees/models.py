"""Immutable contracts for managed child-agent Worktrees."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import IO, Literal
from uuid import UUID


_ROLE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?\Z")

WorktreeKind = Literal["role", "fork"]
WorktreePublicStatus = Literal["active", "retained", "removed", "unavailable"]
PathAccess = Literal["read_only", "read_write"]


@dataclass(frozen=True)
class WorktreeLimits:
    manifest_bytes: int = 64 * 1024
    manifest_line_bytes: int = 1024
    max_links: int = 64
    max_copy_files: int = 512
    max_copy_file_bytes: int = 8 * 1024 * 1024
    max_copy_total_bytes: int = 64 * 1024 * 1024
    metadata_timeout_seconds: float = 30
    lifecycle_timeout_seconds: float = 120


@dataclass(frozen=True)
class WorktreeIdentity:
    kind: WorktreeKind
    session_id: UUID
    role: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "role":
            if self.role is None or _ROLE_NAME.fullmatch(self.role) is None:
                raise ValueError("Worktree role must satisfy the existing agent-name contract.")
        elif self.kind == "fork":
            if self.role is not None:
                raise ValueError("Fork Worktree identities cannot carry a role.")
        else:
            raise ValueError("Unknown Worktree identity kind.")

    @classmethod
    def for_role(cls, session_id: UUID, role: str) -> "WorktreeIdentity":
        return cls("role", session_id, role)

    @classmethod
    def for_fork(cls, session_id: UUID) -> "WorktreeIdentity":
        return cls("fork", session_id)

    @property
    def owner_segment(self) -> str:
        return f"role-{self.role}" if self.kind == "role" else "fork"

    @property
    def relative_path(self) -> Path:
        return Path(self.owner_segment) / str(self.session_id)

    @property
    def branch(self) -> str:
        return f"worktree/{self.owner_segment}/{self.session_id}"


@dataclass(frozen=True)
class PathMapping:
    alias: Path
    target: Path
    access: PathAccess


@dataclass
class WorktreeLease:
    identity: WorktreeIdentity
    project_workspace: Path
    repo_root: Path
    worktree_root: Path
    execution_workspace: Path
    branch: str
    base_sha: str
    state_path: Path
    lease_token: str
    _lock_handle: IO[bytes] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ChildExecutionContext:
    project_workspace: Path
    repo_root: Path
    worktree_root: Path
    execution_workspace: Path
    branch: str
    base_sha: str
    mappings: tuple[PathMapping, ...]
    lease: WorktreeLease


@dataclass(frozen=True)
class WorktreeReleaseReport:
    status: WorktreePublicStatus
    removed: bool
    branch: str
    workspace: Path
    reason: str | None = None
