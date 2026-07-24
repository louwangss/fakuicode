"""Bounded initialization of ignored runtime files and shared dependency links."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
from typing import Any

from fakuicode.worktrees.git import GitCommandError, GitRunner
from fakuicode.worktrees.models import PathMapping, WorktreeLimits


class WorktreeInitializationError(RuntimeError):
    pass


class WorktreeInitializer:
    def __init__(
        self,
        *,
        repo_root: Path,
        worktree_root: Path,
        git: GitRunner,
        limits: WorktreeLimits,
    ) -> None:
        self.repo_root = repo_root
        self.worktree_root = worktree_root
        self.git = git
        self.limits = limits
        self.inventory: dict[str, list[dict[str, Any]]] = {
            "copies": [],
            "links": [],
        }

    def initialize(self) -> tuple[dict[str, list[dict[str, Any]]], tuple[PathMapping, ...]]:
        self.inventory = {"copies": [], "links": []}
        copies = self._copy_included_files()
        links = self._link_dependency_directories()
        mappings = tuple(
            PathMapping(
                self.worktree_root / record["path"],
                Path(record["target"]),
                "read_write",
            )
            for record in links
        )
        return self.inventory, mappings

    def cleanup(self, inventory: dict[str, Any]) -> bool:
        copies = inventory.get("copies")
        links = inventory.get("links")
        if not isinstance(copies, list) or not isinstance(links, list):
            return False
        for record in copies:
            if not self._copy_matches(record):
                return False
        for record in links:
            if not self._link_matches(record):
                return False
        try:
            for record in copies:
                (self.worktree_root / record["path"]).unlink()
            for record in links:
                _remove_directory_link(self.worktree_root / record["path"])
            self._remove_empty_parents(
                [self.worktree_root / record["path"] for record in (*copies, *links)]
            )
        except OSError:
            return False
        return True

    def audit(self, inventory: dict[str, Any]) -> bool:
        """Return True only when tracked, untracked and initialized state is pristine."""

        copies = inventory.get("copies")
        links = inventory.get("links")
        if not isinstance(copies, list) or not isinstance(links, list):
            return False
        if any(not self._copy_matches(record) for record in copies):
            return False
        if any(not self._link_matches(record) for record in links):
            return False
        try:
            unstaged = self.git.run(
                self.worktree_root,
                ("diff", "--quiet", "--exit-code"),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            ).returncode
            staged = self.git.run(
                self.worktree_root,
                ("diff", "--cached", "--quiet", "--exit-code"),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            ).returncode
            ordinary = self._git_paths(
                ("ls-files", "--others", "--exclude-standard", "-z", "--"),
                cwd=self.worktree_root,
            )
            ignored = self._git_paths(
                ("ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--"),
                cwd=self.worktree_root,
            )
        except GitCommandError:
            return False
        if unstaged != 0 or staged != 0 or ordinary:
            return False
        copy_paths = {record["path"] for record in copies if isinstance(record, dict)}
        link_paths = {record["path"] for record in links if isinstance(record, dict)}
        unknown_ignored = {
            item
            for item in ignored
            if item not in copy_paths
            and not any(item == link or item.startswith(f"{link}/") for link in link_paths)
        }
        return not unknown_ignored

    def inventory_contract_valid(self, inventory: object) -> bool:
        """Validate untrusted sidecar records without requiring assets to be unchanged."""

        if not isinstance(inventory, dict) or set(inventory) != {"copies", "links"}:
            return False
        copies = inventory.get("copies")
        links = inventory.get("links")
        if not isinstance(copies, list) or not isinstance(links, list):
            return False
        if len(copies) > self.limits.max_copy_files or len(links) > self.limits.max_links:
            return False
        total = 0
        for record in copies:
            if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
                return False
            try:
                pure = _safe_repo_relative(record["path"], literal=False)
            except (KeyError, TypeError, WorktreeInitializationError):
                return False
            size = record.get("size")
            digest = record.get("sha256")
            if (
                _protected(pure)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or size > self.limits.max_copy_file_bytes
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                return False
            total += size
        if total > self.limits.max_copy_total_bytes:
            return False
        link_paths: list[PurePosixPath] = []
        for record in links:
            if not isinstance(record, dict) or set(record) != {"path", "target"}:
                return False
            try:
                pure = _safe_repo_relative(record["path"])
                target = Path(record["target"])
            except (KeyError, TypeError, WorktreeInitializationError):
                return False
            if (
                _protected(pure)
                or not target.is_absolute()
                or not _is_relative_to(target, self.repo_root)
            ):
                return False
            link_paths.append(pure)
        return not any(
            _overlaps(left, right)
            for index, left in enumerate(link_paths)
            for right in link_paths[index + 1 :]
        )

    def _copy_included_files(self) -> list[dict[str, Any]]:
        manifest = self.worktree_root / ".worktreeinclude"
        if not manifest.exists():
            return []
        self._read_manifest(manifest)
        standard = set(
            self._git_paths(
                ("ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--"),
                cwd=self.repo_root,
            )
        )
        matched = set(
            self._git_paths(
                (
                    "ls-files",
                    "--others",
                    "--ignored",
                    f"--exclude-from={manifest}",
                    "-z",
                    "--",
                ),
                cwd=self.repo_root,
            )
        )
        selected = sorted(standard & matched)
        candidates: list[tuple[str, Path, Path, os.stat_result]] = []
        total = 0
        for relative in selected:
            pure = _safe_repo_relative(relative, literal=False)
            if _protected(pure):
                raise WorktreeInitializationError("worktreeinclude 命中了受保护路径。")
            source = self.repo_root.joinpath(*pure.parts)
            target = self.worktree_root.joinpath(*pure.parts)
            if _is_reparse(source) or _has_reparse_ancestor(source, self.repo_root):
                raise WorktreeInitializationError("worktreeinclude 来源包含链接或 reparse point。")
            if _has_reparse_ancestor(target, self.worktree_root):
                raise WorktreeInitializationError("worktreeinclude 目标路径包含链接。")
            try:
                details = source.stat()
            except OSError as error:
                raise WorktreeInitializationError("worktreeinclude 来源不可读取。") from error
            if not stat.S_ISREG(details.st_mode):
                raise WorktreeInitializationError("worktreeinclude 只允许普通文件。")
            if details.st_size > self.limits.max_copy_file_bytes:
                raise WorktreeInitializationError("worktreeinclude 单文件超过限制。")
            total += details.st_size
            if total > self.limits.max_copy_total_bytes:
                raise WorktreeInitializationError("worktreeinclude 总大小超过限制。")
            if target.exists() or target.is_symlink():
                raise WorktreeInitializationError("worktreeinclude 目标已经存在。")
            candidates.append((relative, source, target, details))
        if len(candidates) > self.limits.max_copy_files:
            raise WorktreeInitializationError("worktreeinclude 文件数量超过限制。")
        records: list[dict[str, Any]] = []
        for relative, source, target, before in candidates:
            digest, size = _copy_atomic_without_overwrite(source, target, before)
            record = {"path": relative, "sha256": digest, "size": size}
            records.append(record)
            self.inventory["copies"].append(record)
        return records

    def _link_dependency_directories(self) -> list[dict[str, Any]]:
        manifest = self.worktree_root / ".worktreelinks"
        if not manifest.exists():
            return []
        lines = self._read_manifest(manifest)
        effective = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(effective) > self.limits.max_links:
            raise WorktreeInitializationError("worktreelinks 条目数量超过限制。")
        paths = [_safe_repo_relative(line) for line in effective]
        for index, candidate in enumerate(paths):
            if _protected(candidate):
                raise WorktreeInitializationError("worktreelinks 引用了受保护路径。")
            for other in paths[index + 1 :]:
                if _overlaps(candidate, other):
                    raise WorktreeInitializationError("worktreelinks 目录不能互相包含。")
        records: list[dict[str, Any]] = []
        for pure in paths:
            relative = pure.as_posix()
            source = self.repo_root.joinpath(*pure.parts)
            target = self.worktree_root.joinpath(*pure.parts)
            if _is_reparse(source) or _has_reparse_ancestor(source, self.repo_root):
                raise WorktreeInitializationError("worktreelinks 来源必须是真实目录。")
            if _has_reparse_ancestor(target, self.worktree_root):
                raise WorktreeInitializationError("worktreelinks 目标路径包含链接。")
            if not source.is_dir():
                raise WorktreeInitializationError("worktreelinks 来源目录不存在。")
            if target.exists() or target.is_symlink():
                raise WorktreeInitializationError("worktreelinks 目标已经存在。")
            tracked = self.git.run(
                self.repo_root,
                ("ls-files", "--", relative),
                timeout=self.limits.metadata_timeout_seconds,
            ).stdout
            ignored = self.git.run(
                self.repo_root,
                ("check-ignore", "--no-index", relative),
                timeout=self.limits.metadata_timeout_seconds,
                check=False,
            )
            if tracked or ignored.returncode != 0:
                raise WorktreeInitializationError("worktreelinks 只允许被忽略的未跟踪目录。")
            target.parent.mkdir(parents=True, exist_ok=True)
            _create_directory_link(target, source)
            record = {"path": relative, "target": str(source.resolve())}
            records.append(record)
            self.inventory["links"].append(record)
        return records

    def _read_manifest(self, path: Path) -> list[str]:
        if _is_reparse(path) or not path.is_file():
            raise WorktreeInitializationError("Worktree 清单必须是普通文件。")
        try:
            with path.open("rb") as handle:
                raw = handle.read(self.limits.manifest_bytes + 1)
        except OSError as error:
            raise WorktreeInitializationError("无法读取 Worktree 清单。") from error
        if len(raw) > self.limits.manifest_bytes:
            raise WorktreeInitializationError("Worktree 清单超过大小限制。")
        for line in raw.splitlines():
            if len(line) > self.limits.manifest_line_bytes:
                raise WorktreeInitializationError("Worktree 清单单行超过限制。")
        try:
            return raw.decode("utf-8-sig").splitlines()
        except UnicodeDecodeError as error:
            raise WorktreeInitializationError("Worktree 清单必须是 UTF-8。") from error

    def _git_paths(self, args: tuple[str, ...], *, cwd: Path) -> tuple[str, ...]:
        output = self.git.run(
            cwd,
            args,
            timeout=self.limits.metadata_timeout_seconds,
        ).stdout
        if "\ufffd" in output:
            raise WorktreeInitializationError("Git 路径无法按 UTF-8 解码。")
        return tuple(item for item in output.split("\0") if item)

    def _copy_matches(self, record: Any) -> bool:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            return False
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(relative, str) or not isinstance(digest, str) or not isinstance(size, int):
            return False
        try:
            pure = _safe_repo_relative(relative, literal=False)
        except WorktreeInitializationError:
            return False
        path = self.worktree_root.joinpath(*pure.parts)
        if _is_reparse(path) or not path.is_file():
            return False
        try:
            return path.stat().st_size == size and _hash_file(path) == digest
        except OSError:
            return False

    def _link_matches(self, record: Any) -> bool:
        if not isinstance(record, dict) or set(record) != {"path", "target"}:
            return False
        relative = record.get("path")
        target = record.get("target")
        if not isinstance(relative, str) or not isinstance(target, str):
            return False
        try:
            pure = _safe_repo_relative(relative)
            expected = Path(target).resolve(strict=True)
            actual = self.worktree_root.joinpath(*pure.parts)
            return _is_reparse(actual) and actual.resolve(strict=True) == expected
        except (OSError, WorktreeInitializationError):
            return False

    def _remove_empty_parents(self, paths: list[Path]) -> None:
        parents = sorted(
            {
                parent
                for path in paths
                for parent in path.parents
                if parent != self.worktree_root
                and _is_relative_to(parent, self.worktree_root)
            },
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for parent in parents:
            try:
                parent.rmdir()
            except OSError:
                pass


def _safe_repo_relative(value: str, *, literal: bool = True) -> PurePosixPath:
    if not value or "\0" in value or "\\" in value:
        raise WorktreeInitializationError("Worktree 清单路径无效。")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise WorktreeInitializationError("Worktree 清单路径必须位于仓库内。")
    if literal and any(any(char in part for char in "*?[]") for part in pure.parts):
        raise WorktreeInitializationError("worktreelinks 只接受字面目录。")
    if os.name == "nt":
        forbidden = '<>:"|'
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        for part in pure.parts:
            if any(char in part for char in forbidden) or part.endswith((" ", ".")):
                raise WorktreeInitializationError("Worktree 清单包含 Windows 非法路径。")
            if part.split(".", 1)[0].upper() in reserved:
                raise WorktreeInitializationError("Worktree 清单包含 Windows 保留名。")
    return pure


def _protected(path: PurePosixPath) -> bool:
    value = path.as_posix()
    return (
        value == ".git"
        or value.startswith(".git/")
        or value.startswith(".fakuicode/worktrees/")
        or value.startswith(".fakuicode/worktree-state/")
        or value.startswith(".fakuicode/context-artifacts/")
        or value in {
            ".fakuicode/permissions.yaml",
            ".fakuicode/permissions.local.yaml",
        }
    )


def _overlaps(left: PurePosixPath, right: PurePosixPath) -> bool:
    return (
        left.parts == right.parts[: len(left.parts)]
        or right.parts == left.parts[: len(right.parts)]
    )


def _copy_atomic_without_overwrite(
    source: Path,
    target: Path,
    before: os.stat_result,
) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(name)
    digest = sha256()
    size = 0
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            descriptor = -1
            while chunk := reader.read(64 * 1024):
                digest.update(chunk)
                size += len(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or size != before.st_size:
            raise WorktreeInitializationError("复制期间来源文件发生变化。")
        os.link(temporary, target)
        temporary.unlink()
        return digest.hexdigest(), size
    except FileExistsError as error:
        raise WorktreeInitializationError("Worktree 初始化拒绝覆盖已有文件。") from error
    except OSError as error:
        raise WorktreeInitializationError("无法安全复制 Worktree 初始化文件。") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _create_directory_link(link: Path, target: Path) -> None:
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    import ctypes

    link.mkdir()
    target_text = str(target.resolve())
    substitute = (
        "\\??\\UNC\\" + target_text[2:]
        if target_text.startswith("\\\\")
        else "\\??\\" + target_text
    )
    substitute_bytes = substitute.encode("utf-16-le")
    print_bytes = target_text.encode("utf-16-le")
    path_buffer = substitute_bytes + b"\0\0" + print_bytes + b"\0\0"
    data = struct.pack(
        "<LHHHHHH",
        0xA0000003,
        8 + len(path_buffer),
        0,
        0,
        len(substitute_bytes),
        len(substitute_bytes) + 2,
        len(print_bytes),
    ) + path_buffer
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    )
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.DeviceIoControl.argtypes = (
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
    )
    kernel32.DeviceIoControl.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.CreateFileW(
        str(link),
        0x40000000,
        0,
        None,
        3,
        0x00200000 | 0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        link.rmdir()
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        returned = ctypes.c_ulong()
        ok = kernel32.DeviceIoControl(
            handle,
            0x000900A4,
            data,
            len(data),
            None,
            0,
            ctypes.byref(returned),
            None,
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
    except OSError:
        link.rmdir()
        raise
    finally:
        kernel32.CloseHandle(handle)


def _remove_directory_link(path: Path) -> None:
    if os.name == "nt":
        path.rmdir()
    else:
        path.unlink()


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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
