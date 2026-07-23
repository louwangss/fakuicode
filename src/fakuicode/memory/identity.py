"""Private registry and conservative local project identity resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import unicodedata
from uuid import uuid4


SCHEMA_VERSION = 1
GitRunner = Callable[[Path, tuple[str, ...]], bytes]


@dataclass(frozen=True)
class MemoryPaths:
    root: Path
    registry: Path
    user_scope: Path
    projects_root: Path

    @classmethod
    def from_home(cls, home: Path) -> "MemoryPaths":
        root = home.resolve() / ".fakuicode" / "memory"
        return cls(root, root / "registry.sqlite3", root / "user", root / "projects")

    def project_scope(self, project_id: str) -> Path:
        from fakuicode.memory.models import canonical_uuid

        canonical_uuid(project_id, field_name="project_id")
        return self.projects_root / project_id


@dataclass(frozen=True)
class MemoryUserState:
    enabled: bool
    notice_shown: bool
    generation: int
    last_update_code: str
    last_update_at: str | None


@dataclass(frozen=True)
class ProjectIdentity:
    project_id: str
    identity_kind: str
    display_label: str


class MemoryRegistry:
    """SQLite metadata store containing no conversation or memory正文."""

    def __init__(self, paths: MemoryPaths, *, lock_timeout_seconds: float = 1.0) -> None:
        self.paths = paths
        self.lock_timeout_seconds = lock_timeout_seconds
        for directory in (
            paths.root.parent,
            paths.root,
            paths.user_scope,
            paths.projects_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.is_dir() or _is_link_or_reparse(directory):
                raise RuntimeError("unsafe_memory_directory")
        if paths.registry.exists() and (
            not paths.registry.is_file() or _is_link_or_reparse(paths.registry)
        ):
            raise RuntimeError("unsafe_memory_registry")
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.registry, timeout=self.lock_timeout_seconds)
        connection.execute(f"PRAGMA busy_timeout = {int(self.lock_timeout_seconds * 1000)}")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    notice_shown INTEGER NOT NULL CHECK (notice_shown IN (0, 1)),
                    generation INTEGER NOT NULL CHECK (generation >= 0),
                    last_update_code TEXT NOT NULL,
                    last_update_at TEXT
                );
                CREATE TABLE IF NOT EXISTS projects (
                    identity_kind TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    project_id TEXT NOT NULL UNIQUE,
                    display_label TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    PRIMARY KEY (identity_kind, identity_key)
                );
                CREATE TABLE IF NOT EXISTS scope_status (
                    scope_key TEXT PRIMARY KEY,
                    status_code TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            row = connection.execute("SELECT version FROM schema_info").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] != SCHEMA_VERSION:
                raise RuntimeError("unsupported_memory_registry")
            connection.execute(
                """
                INSERT OR IGNORE INTO user_state(
                    singleton, enabled, notice_shown, generation, last_update_code, last_update_at
                ) VALUES (1, 1, 0, 0, 'never', NULL)
                """
            )

    def user_state(self, *, connection: sqlite3.Connection | None = None) -> MemoryUserState:
        owned = connection is None
        active = connection or self.connect()
        try:
            row = active.execute(
                """SELECT enabled, notice_shown, generation, last_update_code, last_update_at
                   FROM user_state WHERE singleton = 1"""
            ).fetchone()
            if row is None:
                raise RuntimeError("missing_memory_user_state")
            return MemoryUserState(bool(row[0]), bool(row[1]), row[2], row[3], row[4])
        finally:
            if owned:
                active.close()

    def set_enabled(self, enabled: bool) -> MemoryUserState:
        with self.connect() as connection:
            connection.execute(
                """UPDATE user_state
                   SET enabled = ?, generation = generation + 1
                   WHERE singleton = 1""",
                (int(enabled),),
            )
        return self.user_state()

    def increment_generation(self, *, connection: sqlite3.Connection | None = None) -> int:
        owned = connection is None
        active = connection or self.connect()
        try:
            active.execute(
                "UPDATE user_state SET generation = generation + 1 WHERE singleton = 1"
            )
            if owned:
                active.commit()
            return self.user_state(connection=active).generation
        finally:
            if owned:
                active.close()

    def mark_notice_shown(self) -> MemoryUserState:
        with self.connect() as connection:
            connection.execute("UPDATE user_state SET notice_shown = 1 WHERE singleton = 1")
        return self.user_state()

    def update_last_status(self, code: str) -> None:
        safe_code = _safe_label(code, fallback="unknown")[:64]
        with self.connect() as connection:
            connection.execute(
                """UPDATE user_state SET last_update_code = ?, last_update_at = ?
                   WHERE singleton = 1""",
                (safe_code, _utc_now()),
            )

    def project_for_key(self, identity_kind: str, identity_key: str, label: str) -> ProjectIdentity:
        now = _utc_now()
        safe_label = _safe_label(label, fallback="local project")[:80]
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT project_id, display_label FROM projects
                   WHERE identity_kind = ? AND identity_key = ?""",
                (identity_kind, identity_key),
            ).fetchone()
            if row is None:
                project_id = str(uuid4())
                connection.execute(
                    """INSERT INTO projects(
                           identity_kind, identity_key, project_id, display_label,
                           first_seen_at, last_seen_at
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (identity_kind, identity_key, project_id, safe_label, now, now),
                )
                display_label = safe_label
            else:
                project_id, display_label = row
                connection.execute(
                    """UPDATE projects SET last_seen_at = ?
                       WHERE identity_kind = ? AND identity_key = ?""",
                    (now, identity_kind, identity_key),
                )
        return ProjectIdentity(project_id, identity_kind, display_label)

    def project_count(self, *, excluding: str | None = None) -> int:
        with self.connect() as connection:
            if excluding is None:
                row = connection.execute("SELECT COUNT(*) FROM projects").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM projects WHERE project_id <> ?", (excluding,)
                ).fetchone()
        return int(row[0]) if row else 0


class _NotGitRepository(Exception):
    pass


class ProjectIdentityResolver:
    """Resolve only verified Git metadata or an explicit non-Git real path."""

    def __init__(self, registry: MemoryRegistry, *, git_runner: GitRunner | None = None) -> None:
        self.registry = registry
        self._git_runner = git_runner or self._run_git

    def resolve(self, workspace: Path) -> ProjectIdentity | None:
        try:
            resolved_workspace = workspace.resolve(strict=True)
            if not resolved_workspace.is_dir():
                return None
            try:
                identity_key = self._verified_git_key(resolved_workspace)
                kind = "git_common_dir"
            except _NotGitRepository:
                identity_key = _path_identity_key(resolved_workspace)
                kind = "workspace_path"
            return self.registry.project_for_key(kind, identity_key, resolved_workspace.name)
        except Exception:
            return None

    def _verified_git_key(self, workspace: Path) -> str:
        inside = self._git(workspace, "rev-parse", "--is-inside-work-tree")
        if inside != "true":
            raise RuntimeError("invalid_git_state")
        top = self._git_path(workspace, "--show-toplevel")
        git_dir = self._git_path(workspace, "--absolute-git-dir")
        common_dir = self._git_path(workspace, "--path-format=absolute", "--git-common-dir")
        try:
            workspace.relative_to(top)
        except ValueError as error:
            raise RuntimeError("invalid_git_root") from error
        dot_git = top / ".git"
        if dot_git.is_dir():
            if _is_link_or_reparse(dot_git) or git_dir != dot_git.resolve() or common_dir != git_dir:
                raise RuntimeError("invalid_regular_git")
        elif dot_git.is_file():
            self._validate_linked_worktree(dot_git, git_dir, common_dir)
        else:
            raise RuntimeError("missing_git_control")
        return _path_identity_key(common_dir)

    def _validate_linked_worktree(self, dot_git: Path, git_dir: Path, common_dir: Path) -> None:
        if not _safe_regular_file(dot_git) or _is_link_or_reparse(git_dir) or not git_dir.is_dir():
            raise RuntimeError("unsafe_worktree_control")
        if git_dir.parent.name != "worktrees" or git_dir.parent.parent.resolve() != common_dir:
            raise RuntimeError("invalid_worktree_slot")
        if _is_link_or_reparse(common_dir) or not common_dir.is_dir():
            raise RuntimeError("unsafe_common_dir")
        declared = _read_control_file(dot_git)
        if not declared.startswith("gitdir: "):
            raise RuntimeError("invalid_git_pointer")
        if _resolve_declared_path(declared[8:], dot_git.parent) != git_dir:
            raise RuntimeError("invalid_git_pointer")
        reverse = git_dir / "gitdir"
        commondir = git_dir / "commondir"
        if not _safe_regular_file(reverse) or not _safe_regular_file(commondir):
            raise RuntimeError("missing_worktree_reverse_pointer")
        if _resolve_declared_path(_read_control_file(reverse), git_dir) != dot_git.resolve():
            raise RuntimeError("invalid_worktree_reverse_pointer")
        if _resolve_declared_path(_read_control_file(commondir), git_dir) != common_dir:
            raise RuntimeError("invalid_worktree_common_pointer")

    def _git_path(self, workspace: Path, *arguments: str) -> Path:
        value = self._git(workspace, "rev-parse", *arguments)
        path = Path(value)
        if not path.is_absolute():
            path = workspace / path
        return path.resolve(strict=True)

    def _git(self, workspace: Path, *arguments: str) -> str:
        raw = self._git_runner(workspace, tuple(arguments))
        try:
            value = raw.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as error:
            raise RuntimeError("invalid_git_encoding") from error
        if not value or "\x00" in value:
            raise RuntimeError("invalid_git_output")
        return value

    @staticmethod
    def _run_git(workspace: Path, arguments: tuple[str, ...]) -> bytes:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=workspace,
                capture_output=True,
                check=False,
                timeout=2.0,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("git_unavailable") from error
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").casefold()
            if "not a git repository" in stderr:
                raise _NotGitRepository
            raise RuntimeError("git_failed")
        return result.stdout


def _read_control_file(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(4097)
    if len(raw) > 4096:
        raise RuntimeError("git_control_too_large")
    return raw.decode("utf-8", errors="strict").strip()


def _resolve_declared_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def _safe_regular_file(path: Path) -> bool:
    return path.is_file() and not _is_link_or_reparse(path)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    return path.is_symlink() or bool(attributes & 0x400)


def _path_identity_key(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve(strict=True)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_label(value: str, *, fallback: str) -> str:
    cleaned = "".join(
        "�" if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"} else character
        for character in value
    ).strip()
    return cleaned or fallback


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
