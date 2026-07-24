"""Safe lifecycle management for child-agent Git Worktrees."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from threading import RLock
import tempfile
from typing import Any
from uuid import uuid4

from fakuicode.worktrees.git import GitCommandError, GitRunner
from fakuicode.worktrees.models import (
    WorktreeIdentity,
    WorktreeLease,
    WorktreeLimits,
    WorktreeReleaseReport,
)


_EXCLUDE_BEGIN = "# BEGIN fakuicode managed worktrees v1"
_EXCLUDE_END = "# END fakuicode managed worktrees v1"
_EXCLUDE_BLOCK = (
    f"{_EXCLUDE_BEGIN}\n"
    "/.fakuicode/worktrees/\n"
    "/.fakuicode/worktree-state/\n"
    f"{_EXCLUDE_END}\n"
)
_SHA = __import__("re").compile(r"[0-9a-f]{40,64}\Z")


class WorktreeError(RuntimeError):
    code = "worktree_setup_failed"


class WorktreeUnavailableError(WorktreeError):
    code = "worktree_unavailable"


class WorktreeRecoveryConflictError(WorktreeError):
    code = "worktree_recovery_conflict"


class WorktreeManager:
    def __init__(
        self,
        project_workspace: Path,
        *,
        limits: WorktreeLimits = WorktreeLimits(),
        git_runner: GitRunner | None = None,
    ) -> None:
        self.project_workspace = project_workspace.resolve()
        self.limits = limits
        self.git = git_runner or GitRunner()
        self._lock = RLock()
        self._active: dict[str, WorktreeLease] = {}
        self.repo_root, self.git_common_dir = self._discover_repository()
        try:
            self.project_relative = self.project_workspace.relative_to(self.repo_root)
        except ValueError as error:
            raise WorktreeUnavailableError("当前工作目录不在 Git 仓库内。") from error
        self.worktree_dir = self.repo_root / ".fakuicode" / "worktrees"
        self.state_dir = self.repo_root / ".fakuicode" / "worktree-state"

    def _discover_repository(self) -> tuple[Path, Path]:
        timeout = self.limits.metadata_timeout_seconds
        try:
            inside = self.git.run(
                self.project_workspace,
                ("rev-parse", "--is-inside-work-tree"),
                timeout=timeout,
            ).stdout
            bare = self.git.run(
                self.project_workspace,
                ("rev-parse", "--is-bare-repository"),
                timeout=timeout,
            ).stdout
            root_text = self.git.run(
                self.project_workspace,
                ("rev-parse", "--show-toplevel"),
                timeout=timeout,
            ).stdout
            common_text = self.git.run(
                self.project_workspace,
                ("rev-parse", "--git-common-dir"),
                timeout=timeout,
            ).stdout
        except GitCommandError as error:
            raise WorktreeUnavailableError("无法识别当前 Git 仓库。") from error
        if inside != "true" or bare == "true" or not root_text or not common_text:
            raise WorktreeUnavailableError("Worktree 隔离只支持非 bare Git 工作树。")
        root = Path(root_text).resolve()
        common = Path(common_text)
        if not common.is_absolute():
            common = self.project_workspace / common
        return root, common.resolve()

    def ensure_managed_roots_ignored(self) -> None:
        info = self.git_common_dir / "info"
        exclude = info / "exclude"
        if _is_reparse(exclude) or _has_reparse_ancestor(exclude, self.git_common_dir):
            raise WorktreeUnavailableError("Git exclude 路径不安全。")
        try:
            info.mkdir(parents=True, exist_ok=True)
            original = exclude.read_bytes() if exclude.exists() else b""
            text = original.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise WorktreeUnavailableError("无法读取 Git 本地忽略规则。") from error
        begin_count = text.count(_EXCLUDE_BEGIN)
        end_count = text.count(_EXCLUDE_END)
        if (begin_count, end_count) not in {(0, 0), (1, 1)}:
            raise WorktreeUnavailableError("fakuiCode Git ignore 管理区块不完整。")
        if begin_count == 1:
            start = text.index(_EXCLUDE_BEGIN)
            finish = text.index(_EXCLUDE_END, start) + len(_EXCLUDE_END)
            existing = text[start:finish].replace("\r\n", "\n") + "\n"
            if existing != _EXCLUDE_BLOCK:
                raise WorktreeUnavailableError("fakuiCode Git ignore 管理区块已被修改。")
        else:
            separator = "" if not text or text.endswith(("\n", "\r")) else "\n"
            updated = f"{text}{separator}{_EXCLUDE_BLOCK}".encode("utf-8")
            self._replace_if_unchanged(exclude, original, updated)
        for probe in (
            ".fakuicode/worktrees/probe",
            ".fakuicode/worktree-state/probe.json",
        ):
            tracked = self.git.run(
                self.repo_root,
                ("ls-files", "--", probe),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
            ignored = self.git.run(
                self.repo_root,
                ("check-ignore", "--no-index", probe),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            )
            if tracked or ignored.returncode != 0:
                raise WorktreeUnavailableError("无法保证 Worktree 管理目录不被 Git 追踪。")

    def _replace_if_unchanged(self, path: Path, original: bytes, updated: bytes) -> None:
        descriptor = -1
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            current = path.read_bytes() if path.exists() else b""
            if sha256(current).digest() != sha256(original).digest():
                raise WorktreeUnavailableError("Git 本地忽略规则在写入期间发生变化。")
            os.replace(temporary, path)
        except WorktreeUnavailableError:
            raise
        except OSError as error:
            raise WorktreeUnavailableError("无法安全更新 Git 本地忽略规则。") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def create(self, identity: WorktreeIdentity) -> WorktreeLease:
        key = str(identity.session_id)
        with self._lock:
            if key in self._active:
                raise WorktreeError("Worktree 会话已经存在。")
        self.ensure_managed_roots_ignored()
        worktree_root = (self.worktree_dir / identity.relative_path).resolve()
        if not _is_strict_descendant(worktree_root, self.worktree_dir.resolve()):
            raise WorktreeUnavailableError("Worktree 路径越过管理目录。")
        if worktree_root.exists():
            raise WorktreeRecoveryConflictError("已存在的 Worktree 需要执行严格恢复校验。")
        self.git.run(
            self.repo_root,
            ("check-ref-format", "--branch", identity.branch),
            timeout=self.limits.metadata_timeout_seconds,
        )
        base_sha = self.git.run(
            self.repo_root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=self.limits.metadata_timeout_seconds,
        ).stdout
        if _SHA.fullmatch(base_sha) is None:
            raise WorktreeError("Git HEAD 不是有效提交。")
        state_path = self.state_dir / f"{identity.session_id}.json"
        lease_token = str(uuid4())
        state = self._state_document(
            identity,
            worktree_root,
            base_sha,
            lease_token,
            status="creating",
        )
        self._write_state(state_path, state)
        created = False
        try:
            worktree_root.parent.mkdir(parents=True, exist_ok=True)
            self.git.run(
                self.repo_root,
                ("worktree", "add", "-b", identity.branch, str(worktree_root), base_sha),
                timeout=self.limits.lifecycle_timeout_seconds,
            )
            created = True
            execution_workspace = (worktree_root / self.project_relative).resolve()
            if not _is_within_or_equal(execution_workspace, worktree_root):
                raise WorktreeError("子 Agent 执行目录越过 Worktree。")
            if not execution_workspace.is_dir():
                raise WorktreeError("基线提交中不存在当前项目子目录。")
            self.git.run(
                self.repo_root,
                ("worktree", "lock", "--reason", f"fakuicode:{lease_token}", str(worktree_root)),
                timeout=self.limits.metadata_timeout_seconds,
            )
            lock_handle = _acquire_lease_lock(state_path.with_suffix(".lock"))
            lease = WorktreeLease(
                identity=identity,
                project_workspace=self.project_workspace,
                repo_root=self.repo_root,
                worktree_root=worktree_root,
                execution_workspace=execution_workspace,
                branch=identity.branch,
                base_sha=base_sha,
                state_path=state_path,
                lease_token=lease_token,
                _lock_handle=lock_handle,
            )
            state["status"] = "active"
            self._write_state(state_path, state)
            with self._lock:
                self._active[key] = lease
            return lease
        except Exception as error:
            state["status"] = "orphaned"
            self._write_state(state_path, state)
            if created and not isinstance(error, GitCommandError):
                self._rollback_created(identity, worktree_root, base_sha)
            if isinstance(error, WorktreeError):
                raise
            raise WorktreeError("Worktree 创建失败。") from error

    def release(self, lease: WorktreeLease) -> WorktreeReleaseReport:
        clean = self._is_pristine(lease)
        if not clean:
            self._update_status(lease.state_path, "retained")
            _release_lease_lock(lease)
            with self._lock:
                self._active.pop(str(lease.identity.session_id), None)
            return WorktreeReleaseReport(
                "retained",
                False,
                lease.branch,
                lease.execution_workspace,
                "Worktree 包含修改或新增提交。",
            )
        self._update_status(lease.state_path, "removing")
        try:
            self.git.run(
                self.repo_root,
                ("worktree", "unlock", str(lease.worktree_root)),
                timeout=self.limits.metadata_timeout_seconds,
            )
            self.git.run(
                self.repo_root,
                ("worktree", "remove", str(lease.worktree_root)),
                timeout=self.limits.lifecycle_timeout_seconds,
            )
            self.git.run(
                self.repo_root,
                ("update-ref", "-d", f"refs/heads/{lease.branch}", lease.base_sha),
                timeout=self.limits.metadata_timeout_seconds,
            )
        except GitCommandError as error:
            self._update_status(lease.state_path, "orphaned")
            _release_lease_lock(lease)
            raise WorktreeError("Worktree 安全删除失败。") from error
        _release_lease_lock(lease)
        self._update_status(lease.state_path, "removed")
        with self._lock:
            self._active.pop(str(lease.identity.session_id), None)
        return WorktreeReleaseReport(
            "removed",
            True,
            lease.branch,
            lease.execution_workspace,
        )

    def _is_pristine(self, lease: WorktreeLease) -> bool:
        try:
            status_text = self.git.run(
                lease.worktree_root,
                ("status", "--porcelain=v1", "--untracked-files=all"),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
            head = self.git.run(
                lease.worktree_root,
                ("rev-parse", "--verify", "HEAD^{commit}"),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
        except GitCommandError:
            return False
        return not status_text and head == lease.base_sha

    def _rollback_created(
        self,
        identity: WorktreeIdentity,
        worktree_root: Path,
        base_sha: str,
    ) -> None:
        try:
            self.git.run(
                self.repo_root,
                ("worktree", "unlock", str(worktree_root)),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            )
            self.git.run(
                self.repo_root,
                ("worktree", "remove", str(worktree_root)),
                timeout=self.limits.lifecycle_timeout_seconds,
            )
            self.git.run(
                self.repo_root,
                ("update-ref", "-d", f"refs/heads/{identity.branch}", base_sha),
                timeout=self.limits.metadata_timeout_seconds,
            )
        except GitCommandError:
            return

    def _state_document(
        self,
        identity: WorktreeIdentity,
        worktree_root: Path,
        base_sha: str,
        lease_token: str,
        *,
        status: str,
    ) -> dict[str, Any]:
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        return {
            "version": 1,
            "status": status,
            "identity": {
                "kind": identity.kind,
                "session_id": str(identity.session_id),
                "role": identity.role,
            },
            "project_workspace": str(self.project_workspace),
            "repo_root": str(self.repo_root),
            "git_common_dir": str(self.git_common_dir),
            "worktree_root": str(worktree_root),
            "execution_workspace": str(worktree_root / self.project_relative),
            "branch": identity.branch,
            "base_sha": base_sha,
            "lease_token": lease_token,
            "created_at": now,
            "last_used_at": now,
            "initialization": {"copies": [], "links": []},
        }

    @staticmethod
    def _write_state(path: Path, document: dict[str, Any]) -> None:
        payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass
            raise WorktreeError("无法持久化 Worktree 状态。") from error

    @classmethod
    def _update_status(cls, path: Path, status: str) -> None:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorktreeError("Worktree 状态文件不可用。") from error
        document["status"] = status
        document["last_used_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        cls._write_state(path, document)


def _is_strict_descendant(candidate: Path, root: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts) and all(part not in {".", ".."} for part in relative.parts)


def _is_within_or_equal(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if stat.S_ISLNK(details.st_mode):
        return True
    attributes = getattr(details, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _has_reparse_ancestor(path: Path, stop: Path) -> bool:
    current = path.parent
    stop = stop.resolve()
    while True:
        if current.exists() and _is_reparse(current):
            return True
        if current == stop:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _acquire_lease_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if handle.read(1) != b"\0":
        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ImportError) as error:
        handle.close()
        raise WorktreeUnavailableError("Worktree 正被其他进程使用。") from error
    return handle


def _release_lease_lock(lease: WorktreeLease) -> None:
    handle = lease._lock_handle
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
        lease._lock_handle = None
