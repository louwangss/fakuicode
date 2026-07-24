"""Safe lifecycle management for child-agent Git Worktrees."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
import tempfile
from typing import Any
from uuid import UUID, uuid4

from fakuicode.worktrees.git import GitCommandError, GitRunner
from fakuicode.worktrees.initialization import (
    WorktreeInitializer,
)
from fakuicode.worktrees.models import (
    WorktreeIdentity,
    WorktreeLease,
    WorktreeLimits,
    WorktreeReleaseReport,
    PathMapping,
)


_EXCLUDE_BEGIN = "# BEGIN fakuicode managed worktrees v1"
_EXCLUDE_END = "# END fakuicode managed worktrees v1"
_EXCLUDE_BLOCK = (
    f"{_EXCLUDE_BEGIN}\n"
    "/.fakuicode/worktrees/\n"
    "/.fakuicode/worktree-state/\n"
    f"{_EXCLUDE_END}\n"
)
_SHA = re.compile(r"[0-9a-f]{40,64}\Z")
_MAX_STATE_BYTES = 4 * 1024 * 1024


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
        if _is_reparse(common) or _has_reparse_ancestor(common, root):
            raise WorktreeUnavailableError("Git common-dir 包含链接或 reparse point。")
        return root, common.resolve()

    def ensure_managed_roots_ignored(self) -> None:
        info = self.git_common_dir / "info"
        if _is_reparse(info) or _has_reparse_ancestor(info, self.git_common_dir):
            raise WorktreeUnavailableError("Git info 路径不安全。")
        lock_handle = _acquire_lease_lock(info / "fakuicode-worktrees.lock")
        try:
            self._ensure_managed_roots_ignored()
        finally:
            _release_lock_handle(lock_handle)

    def _ensure_managed_roots_ignored(self) -> None:
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
            try:
                finish = text.index(_EXCLUDE_END, start) + len(_EXCLUDE_END)
            except ValueError as error:
                raise WorktreeUnavailableError(
                    "fakuiCode Git ignore 管理区块顺序无效。"
                ) from error
            existing = text[start:finish].replace("\r\n", "\n") + "\n"
            if existing != _EXCLUDE_BLOCK:
                raise WorktreeUnavailableError("fakuiCode Git ignore 管理区块已被修改。")
        else:
            separator = "" if not text or text.endswith(("\n", "\r")) else "\n"
            updated = f"{text}{separator}{_EXCLUDE_BLOCK}".encode("utf-8")
            self._replace_if_unchanged(exclude, original, updated)
        tracked_roots = self.git.run(
            self.repo_root,
            (
                "ls-files",
                "--",
                ".fakuicode/worktrees",
                ":(glob).fakuicode/worktrees/**",
                ".fakuicode/worktree-state",
                ":(glob).fakuicode/worktree-state/**",
            ),
            timeout=self.limits.metadata_timeout_seconds,
        ).stdout
        if tracked_roots:
            raise WorktreeUnavailableError("Worktree 管理目录已包含被 Git 追踪的文件。")
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
        lexical_root = self.worktree_dir / identity.relative_path
        if (
            _has_reparse_ancestor(lexical_root, self.repo_root)
            or _is_reparse(self.worktree_dir)
            or _is_reparse(self.state_dir)
        ):
            raise WorktreeUnavailableError("Worktree 管理路径包含链接或 reparse point。")
        worktree_root = lexical_root.resolve()
        if not _is_strict_descendant(worktree_root, self.worktree_dir.resolve()):
            raise WorktreeUnavailableError("Worktree 路径越过管理目录。")
        state_path = self.state_dir / f"{identity.session_id}.json"
        if worktree_root.exists():
            lease = self._recover_existing(identity, worktree_root, state_path)
            with self._lock:
                self._active[key] = lease
            return lease
        self.ensure_managed_roots_ignored()
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
        lease_token = str(uuid4())
        state = self._state_document(
            identity,
            worktree_root,
            base_sha,
            lease_token,
            status="creating",
        )
        creation_lock = _acquire_lease_lock(state_path.with_suffix(".lock"))
        created = False
        state_owned = False
        initializer: WorktreeInitializer | None = None
        try:
            if worktree_root.exists() or state_path.exists():
                raise WorktreeRecoveryConflictError(
                    "Worktree 身份在创建期间被其他进程占用。"
                )
            self._write_state(state_path, state)
            state_owned = True
            worktree_root.parent.mkdir(parents=True, exist_ok=True)
            self.git.run(
                self.repo_root,
                ("worktree", "add", "-b", identity.branch, str(worktree_root), base_sha),
                timeout=self.limits.lifecycle_timeout_seconds,
            )
            created = True
            self.git.run(
                self.repo_root,
                ("worktree", "lock", "--reason", f"fakuicode:{lease_token}", str(worktree_root)),
                timeout=self.limits.metadata_timeout_seconds,
            )
            initializer = WorktreeInitializer(
                repo_root=self.repo_root,
                worktree_root=worktree_root,
                git=self.git,
                limits=self.limits,
            )
            inventory, mappings = initializer.initialize()
            state["initialization"] = inventory
            hooks = self.git.run(
                worktree_root,
                ("rev-parse", "--git-path", "hooks"),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            )
            state["git_hooks"] = {
                "available": hooks.returncode == 0,
                "path": hooks.stdout if hooks.returncode == 0 else None,
            }
            self._write_state(state_path, state)
            execution_workspace = (worktree_root / self.project_relative).resolve()
            if not _is_within_or_equal(execution_workspace, worktree_root):
                raise WorktreeError("子 Agent 执行目录越过 Worktree。")
            if not execution_workspace.is_dir():
                raise WorktreeError("基线提交中不存在当前项目子目录。")
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
                mappings=mappings,
                _lock_handle=creation_lock,
            )
            creation_lock = None
            state["status"] = "active"
            self._write_state(state_path, state)
            with self._lock:
                self._active[key] = lease
            return lease
        except Exception as error:
            if initializer is not None:
                state["initialization"] = initializer.inventory
            rolled_back = False
            if created and not (
                isinstance(error, GitCommandError) and error.timed_out
            ):
                rolled_back = self._rollback_created(
                    identity,
                    worktree_root,
                    base_sha,
                    state,
                )
            if state_owned:
                state["status"] = "removed" if rolled_back else "orphaned"
                self._write_state(state_path, state)
            if isinstance(error, WorktreeError):
                raise
            raise WorktreeError("Worktree 创建失败。") from error
        finally:
            if creation_lock is not None:
                _release_lock_handle(creation_lock)

    def release(self, lease: WorktreeLease) -> WorktreeReleaseReport:
        try:
            clean = self._is_pristine(lease)
        except WorktreeError:
            self._deactivate(lease)
            raise
        if not clean:
            self._update_status_and_deactivate(lease, "retained")
            return WorktreeReleaseReport(
                "retained",
                False,
                lease.branch,
                lease.execution_workspace,
                "Worktree 包含修改或新增提交。",
            )
        self._update_status(lease.state_path, "removing")
        try:
            state = self._load_state(lease.state_path)
            initializer = WorktreeInitializer(
                repo_root=self.repo_root,
                worktree_root=lease.worktree_root,
                git=self.git,
                limits=self.limits,
            )
            if not initializer.cleanup(state.get("initialization", {})):
                self._update_status_and_deactivate(lease, "retained")
                return WorktreeReleaseReport(
                    "retained",
                    False,
                    lease.branch,
                    lease.execution_workspace,
                    "初始化资产无法安全清理。",
                )
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
            self._update_status_and_deactivate(lease, "orphaned")
            raise WorktreeError("Worktree 安全删除失败。") from error
        self._deactivate(lease)
        self._update_status(lease.state_path, "removed")
        return WorktreeReleaseReport(
            "removed",
            True,
            lease.branch,
            lease.execution_workspace,
        )

    def _deactivate(self, lease: WorktreeLease) -> None:
        _release_lease_lock(lease)
        with self._lock:
            self._active.pop(str(lease.identity.session_id), None)

    def _update_status_and_deactivate(
        self,
        lease: WorktreeLease,
        status: str,
    ) -> None:
        try:
            self._update_status(lease.state_path, status)
        finally:
            self._deactivate(lease)

    def touch(self, lease: WorktreeLease) -> None:
        """Refresh liveness without changing ownership or lifecycle state."""

        try:
            state = self._load_state(lease.state_path)
        except WorktreeError:
            return
        if state.get("lease_token") != lease.lease_token:
            return
        state["last_used_at"] = _utc_now().isoformat()
        try:
            self._write_state(lease.state_path, state)
        except WorktreeError:
            return

    def sweep_stale(self, cutoff: datetime) -> tuple[WorktreeReleaseReport, ...]:
        """Remove only expired, provably owned Worktrees that no process is using."""

        if cutoff.tzinfo is None:
            raise ValueError("Worktree 清理截止时间必须包含时区。")
        if not self.state_dir.is_dir() or _is_reparse(self.state_dir):
            return ()
        reports: list[WorktreeReleaseReport] = []
        for state_path in sorted(self.state_dir.glob("*.json")):
            lease: WorktreeLease | None = None
            try:
                state = self._load_state(state_path)
                last_used = _parse_utc_time(state.get("last_used_at"))
                if last_used is None or last_used > cutoff:
                    continue
                identity = _identity_from_state(state)
                if state_path.name != f"{identity.session_id}.json":
                    continue
                with self._lock:
                    if str(identity.session_id) in self._active:
                        continue
                worktree_root = self.worktree_dir / identity.relative_path
                lease = self._recover_existing(identity, worktree_root, state_path)
                reports.append(self._release_stale(lease))
                lease = None
            except WorktreeError:
                continue
            finally:
                if lease is not None:
                    _release_lease_lock(lease)
        return tuple(reports)

    def _release_stale(self, lease: WorktreeLease) -> WorktreeReleaseReport:
        initializer = WorktreeInitializer(
            repo_root=self.repo_root,
            worktree_root=lease.worktree_root,
            git=self.git,
            limits=self.limits,
        )
        try:
            state = self._load_state(lease.state_path)
            if not initializer.audit(state.get("initialization", {})):
                self._update_status(lease.state_path, "retained")
                _release_lease_lock(lease)
                return _retained_report(lease, "过期 Worktree 包含文件修改。")
            head = self.git.run(
                lease.worktree_root,
                ("rev-parse", "--verify", "HEAD^{commit}"),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
            ancestor = self.git.run(
                lease.worktree_root,
                ("merge-base", "--is-ancestor", lease.base_sha, head),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            )
            if ancestor.returncode != 0:
                self._update_status(lease.state_path, "retained")
                _release_lease_lock(lease)
                return _retained_report(lease, "Worktree 基线不是当前 HEAD 的祖先。")
            if head == lease.base_sha:
                return self.release(lease)
            upstream = self.git.run(
                lease.worktree_root,
                ("rev-parse", "--verify", "@{upstream}^{commit}"),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            )
            unpublished = self.git.run(
                lease.worktree_root,
                (
                    "rev-list",
                    "--max-count=1",
                    f"{lease.base_sha}..HEAD",
                    "--not",
                    "--remotes",
                ),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
            if (
                upstream.returncode != 0
                or _SHA.fullmatch(upstream.stdout) is None
                or unpublished
                or self.git.run(
                    lease.worktree_root,
                    ("merge-base", "--is-ancestor", head, upstream.stdout),
                    timeout=self.limits.metadata_timeout_seconds,
                    check=False,
                ).returncode
                != 0
            ):
                self._update_status(lease.state_path, "retained")
                _release_lease_lock(lease)
                return _retained_report(lease, "Worktree 包含未推送提交。")
            self._update_status(lease.state_path, "removing")
            if not initializer.cleanup(state.get("initialization", {})):
                self._update_status(lease.state_path, "retained")
                _release_lease_lock(lease)
                return _retained_report(lease, "初始化资产无法安全清理。")
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
            self._update_status(lease.state_path, "branch_preserved")
            _release_lease_lock(lease)
            return WorktreeReleaseReport(
                "removed",
                True,
                lease.branch,
                lease.execution_workspace,
                "Worktree 目录已删除，已推送分支被保留。",
            )
        except GitCommandError as error:
            self._update_status(lease.state_path, "orphaned")
            _release_lease_lock(lease)
            raise WorktreeError("过期 Worktree 安全清理失败。") from error

    def _is_pristine(self, lease: WorktreeLease) -> bool:
        try:
            state = self._load_state(lease.state_path)
            initializer = WorktreeInitializer(
                repo_root=self.repo_root,
                worktree_root=lease.worktree_root,
                git=self.git,
                limits=self.limits,
            )
            if not initializer.audit(state.get("initialization", {})):
                return False
            head = self.git.run(
                lease.worktree_root,
                ("rev-parse", "--verify", "HEAD^{commit}"),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
        except GitCommandError:
            return False
        return head == lease.base_sha

    def _rollback_created(
        self,
        identity: WorktreeIdentity,
        worktree_root: Path,
        base_sha: str,
        state: dict[str, Any],
    ) -> bool:
        try:
            initializer = WorktreeInitializer(
                repo_root=self.repo_root,
                worktree_root=worktree_root,
                git=self.git,
                limits=self.limits,
            )
            if not initializer.cleanup(state.get("initialization", {})):
                return False
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
            return True
        except GitCommandError:
            return False

    def _recover_existing(
        self,
        identity: WorktreeIdentity,
        worktree_root: Path,
        state_path: Path,
    ) -> WorktreeLease:
        try:
            state = self._load_state(state_path)
            expected_identity = {
                "kind": identity.kind,
                "session_id": str(identity.session_id),
                "role": identity.role,
            }
            if state.get("version") != 1 or state.get("identity") != expected_identity:
                raise WorktreeRecoveryConflictError("Worktree 状态身份不匹配。")
            if state.get("repo_fingerprint") != _path_fingerprint(self.repo_root):
                raise WorktreeRecoveryConflictError("Worktree 仓库指纹不匹配。")
            if state.get("common_dir_fingerprint") != _path_fingerprint(
                self.git_common_dir
            ):
                raise WorktreeRecoveryConflictError("Worktree common-dir 指纹不匹配。")
            if state.get("status") not in {"active", "retained"}:
                raise WorktreeRecoveryConflictError("Worktree 状态不允许恢复。")
            expected_paths = {
                "project_workspace": self.project_workspace,
                "repo_root": self.repo_root,
                "git_common_dir": self.git_common_dir,
                "worktree_root": worktree_root,
                "execution_workspace": worktree_root / self.project_relative,
            }
            for field, expected in expected_paths.items():
                raw = state.get(field)
                if not isinstance(raw, str) or _path_key(Path(raw)) != _path_key(expected):
                    raise WorktreeRecoveryConflictError("Worktree 状态路径不匹配。")
            if state.get("branch") != identity.branch:
                raise WorktreeRecoveryConflictError("Worktree 分支不匹配。")
            base_sha = state.get("base_sha")
            token = state.get("lease_token")
            if not isinstance(base_sha, str) or _SHA.fullmatch(base_sha) is None:
                raise WorktreeRecoveryConflictError("Worktree 基线无效。")
            if not isinstance(token, str) or not token:
                raise WorktreeRecoveryConflictError("Worktree 租约无效。")
            git_file = worktree_root / ".git"
            if _is_reparse(git_file) or not git_file.is_file():
                raise WorktreeRecoveryConflictError("Worktree Git 控制文件无效。")
            control = _read_control_file(git_file)
            if not control.startswith("gitdir: "):
                raise WorktreeRecoveryConflictError("Worktree gitdir 指针无效。")
            gitdir = Path(control.removeprefix("gitdir: ").strip())
            if not gitdir.is_absolute():
                gitdir = git_file.parent / gitdir
            gitdir = gitdir.resolve(strict=True)
            metadata_root = (self.git_common_dir / "worktrees").resolve()
            if not _is_strict_descendant(gitdir, metadata_root) or _is_reparse(gitdir):
                raise WorktreeRecoveryConflictError("Worktree gitdir 越界。")
            backlink = _read_control_file(gitdir / "gitdir")
            if _path_key(Path(backlink)) != _path_key(git_file):
                raise WorktreeRecoveryConflictError("Worktree gitdir 回指不匹配。")
            common_text = _read_control_file(gitdir / "commondir")
            common = Path(common_text)
            if not common.is_absolute():
                common = gitdir / common
            if _path_key(common.resolve(strict=True)) != _path_key(self.git_common_dir):
                raise WorktreeRecoveryConflictError("Worktree common-dir 不匹配。")
            head = _read_control_file(gitdir / "HEAD")
            expected_ref = f"refs/heads/{identity.branch}"
            if head != f"ref: {expected_ref}":
                raise WorktreeRecoveryConflictError("Worktree HEAD 分支不匹配。")
            if _read_ref(self.git_common_dir, expected_ref) is None:
                raise WorktreeRecoveryConflictError("Worktree HEAD 引用不存在。")
            reason = _read_control_file(gitdir / "locked")
            if reason != f"fakuicode:{token}":
                raise WorktreeRecoveryConflictError("Worktree Git 锁不属于 fakuiCode。")
            inventory = state.get("initialization")
            initializer = WorktreeInitializer(
                repo_root=self.repo_root,
                worktree_root=worktree_root,
                git=self.git,
                limits=self.limits,
            )
            if not initializer.inventory_contract_valid(inventory):
                raise WorktreeRecoveryConflictError("Worktree 初始化记录无效。")
            assert isinstance(inventory, dict)
            mappings = tuple(
                PathMapping(
                    worktree_root / record["path"],
                    Path(record["target"]),
                    "read_write",
                )
                for record in inventory.get("links", [])
                if isinstance(record, dict)
                and isinstance(record.get("path"), str)
                and isinstance(record.get("target"), str)
            )
            lock_handle = _acquire_lease_lock(state_path.with_suffix(".lock"))
            return WorktreeLease(
                identity=identity,
                project_workspace=self.project_workspace,
                repo_root=self.repo_root,
                worktree_root=worktree_root,
                execution_workspace=(worktree_root / self.project_relative).resolve(),
                branch=identity.branch,
                base_sha=base_sha,
                state_path=state_path,
                lease_token=token,
                mappings=mappings,
                _lock_handle=lock_handle,
            )
        except WorktreeRecoveryConflictError:
            raise
        except WorktreeError as error:
            raise WorktreeRecoveryConflictError(
                "Worktree 快速恢复校验失败。"
            ) from error
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            raise WorktreeRecoveryConflictError("Worktree 快速恢复校验失败。") from error

    def _state_document(
        self,
        identity: WorktreeIdentity,
        worktree_root: Path,
        base_sha: str,
        lease_token: str,
        *,
        status: str,
    ) -> dict[str, Any]:
        now = _utc_now().isoformat()
        return {
            "version": 1,
            "status": status,
            "identity": {
                "kind": identity.kind,
                "session_id": str(identity.session_id),
                "role": identity.role,
            },
            "repo_fingerprint": _path_fingerprint(self.repo_root),
            "common_dir_fingerprint": _path_fingerprint(self.git_common_dir),
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
        if (
            _is_reparse(path)
            or _is_reparse(path.parent)
            or _is_reparse(path.parent.parent)
        ):
            raise WorktreeError("Worktree 状态路径不安全。")
        payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        temporary: Path | None = None
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
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise WorktreeError("无法持久化 Worktree 状态。") from error

    @classmethod
    def _update_status(cls, path: Path, status: str) -> None:
        document = cls._load_state(path)
        document["status"] = status
        document["last_used_at"] = _utc_now().isoformat()
        cls._write_state(path, document)

    @staticmethod
    def _load_state(path: Path) -> dict[str, Any]:
        if _is_reparse(path) or not path.is_file():
            raise WorktreeError("Worktree 状态文件不可用。")
        try:
            with path.open("rb") as handle:
                payload = handle.read(_MAX_STATE_BYTES + 1)
            if len(payload) > _MAX_STATE_BYTES:
                raise WorktreeError("Worktree 状态文件超过大小限制。")
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorktreeError("Worktree 状态文件不可用。") from error
        if not isinstance(document, dict):
            raise WorktreeError("Worktree 状态文件不可用。")
        return document


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    normalized = parsed.astimezone(timezone.utc)
    if normalized > _utc_now():
        return None
    return normalized


def _identity_from_state(state: dict[str, Any]) -> WorktreeIdentity:
    raw = state.get("identity")
    if not isinstance(raw, dict) or set(raw) != {"kind", "session_id", "role"}:
        raise WorktreeRecoveryConflictError("Worktree 状态身份无效。")
    try:
        session_id = UUID(str(raw["session_id"]))
        if raw["kind"] == "role":
            return WorktreeIdentity.for_role(session_id, str(raw["role"]))
        if raw["kind"] == "fork" and raw["role"] is None:
            return WorktreeIdentity.for_fork(session_id)
    except (TypeError, ValueError) as error:
        raise WorktreeRecoveryConflictError("Worktree 状态身份无效。") from error
    raise WorktreeRecoveryConflictError("Worktree 状态身份无效。")


def _retained_report(lease: WorktreeLease, reason: str) -> WorktreeReleaseReport:
    return WorktreeReleaseReport(
        "retained",
        False,
        lease.branch,
        lease.execution_workspace,
        reason,
    )


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


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _path_fingerprint(path: Path) -> dict[str, object]:
    try:
        details = path.stat()
    except OSError as error:
        raise WorktreeUnavailableError("无法读取 Git 仓库指纹。") from error
    return {
        "path_sha256": sha256(_path_key(path).encode("utf-8")).hexdigest(),
        "device": int(details.st_dev),
        "inode": int(details.st_ino),
    }


def _read_ref(common_dir: Path, reference: str) -> str | None:
    parts = reference.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    loose = common_dir.joinpath(*parts)
    if (
        loose.is_file()
        and not _is_reparse(loose)
        and not _has_reparse_ancestor(loose, common_dir)
    ):
        value = loose.read_text(encoding="ascii").strip()
        return value if _SHA.fullmatch(value) else None
    packed = common_dir / "packed-refs"
    if not packed.is_file() or _is_reparse(packed):
        return None
    for line in packed.read_text(encoding="ascii").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        try:
            value, name = line.split(" ", 1)
        except ValueError:
            return None
        if name == reference:
            return value if _SHA.fullmatch(value) else None
    return None


def _read_control_file(path: Path) -> str:
    if _is_reparse(path) or not path.is_file():
        raise WorktreeRecoveryConflictError("Worktree Git 控制文件无效。")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise WorktreeRecoveryConflictError(
            "Worktree Git 控制文件不可读取。"
        ) from error
    if len(content) > 16 * 1024 or "\0" in content:
        raise WorktreeRecoveryConflictError("Worktree Git 控制文件无效。")
    return content.strip()


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
    _release_lock_handle(handle)
    lease._lock_handle = None


def _release_lock_handle(handle: Any) -> None:
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
