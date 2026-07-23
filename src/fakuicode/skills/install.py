"""Safe public Skill source resolution and GitHub package acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from threading import Event
from types import MappingProxyType
from typing import Callable, Mapping, Protocol
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
import yaml

from fakuicode.errors import RequestCancelled
from fakuicode.instructions.models import DEFAULT_INSTRUCTION_LIMITS
from fakuicode.skills.models import SkillDefinition, SkillSnapshot, SkillSource
from fakuicode.skills.parser import SkillParseError, fingerprint_upstream, parse_skill_package


_GITHUB_HOST = "github.com"
_SKILLS_HOSTS = {"skills.sh", "www.skills.sh"}
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_PATH_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TREE_MODES = {"040000", "40000"}
_REGULAR_MODES = {"100644", "100755"}
_UNSAFE_MODES = {"120000", "160000"}
_RESERVED_RECEIPT = ".fakuicode/install.yaml"
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class SkillInstallError(ValueError):
    """A public Skill source cannot be installed safely."""


@dataclass(frozen=True)
class SkillInstallSource:
    requested_url: str
    owner: str
    repo: str
    requested_skill: str | None
    ref: str | None = None
    skill_path: str | None = None

    @property
    def canonical_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


@dataclass(frozen=True)
class RemoteSkillPackage:
    source: SkillInstallSource
    name: str
    revision: str
    skill_path: str
    files: Mapping[str, bytes]

    @property
    def total_bytes(self) -> int:
        return sum(len(content) for content in self.files.values())


class SkillInstallScope(StrEnum):
    PROJECT = "project"
    USER = "user"


class SkillInstallPreset(StrEnum):
    INSTRUCTION = "instruction"
    READ_ONLY = "read-only"
    CODING = "coding"


_PRESET_TOOLS: Mapping[SkillInstallPreset, tuple[str, ...]] = MappingProxyType(
    {
        SkillInstallPreset.INSTRUCTION: (),
        SkillInstallPreset.READ_ONLY: ("read_file", "find_files", "search_code"),
        SkillInstallPreset.CODING: (
            "read_file",
            "find_files",
            "search_code",
            "write_file",
            "edit_file",
            "run_command",
        ),
    }
)


@dataclass(frozen=True)
class SkillInstallRequest:
    source: str
    skill: str | None = None
    scope: SkillInstallScope = SkillInstallScope.PROJECT
    preset: SkillInstallPreset | None = None
    replace: bool = False


@dataclass(frozen=True)
class SkillInstallDecision:
    approved: bool
    preset: SkillInstallPreset


@dataclass(frozen=True)
class SkillInstallPreview:
    name: str
    description: str
    license: str | None
    requested_url: str
    source_url: str
    revision: str
    skill_path: str
    target_path: Path
    scope: SkillInstallScope
    preset: SkillInstallPreset
    visible_tools: tuple[str, ...]
    files: tuple[str, ...]
    total_bytes: int
    contains_scripts: bool
    dedicated_tools: tuple[str, ...]
    replacing: bool
    shadows: tuple[str, ...]
    upstream_allowed_tools: str | None = None

    @property
    def file_count(self) -> int:
        return len(self.files)


@dataclass(frozen=True)
class SkillInstallResult:
    success: bool
    name: str
    output: str
    target_path: Path | None = None


InstallConfirmation = Callable[[SkillInstallPreview], SkillInstallDecision]
RefreshSkills = Callable[[], SkillSnapshot | None]


class SkillPackageFetcher(Protocol):
    def fetch(
        self,
        source: SkillInstallSource,
        *,
        cancel_event: Event | None = None,
    ) -> RemoteSkillPackage: ...


class SkillInstaller:
    """Stage, confirm and atomically install one validated remote Skill package."""

    def __init__(
        self,
        workspace: Path,
        user_root: Path,
        *,
        fetcher: SkillPackageFetcher | None = None,
        refresh: RefreshSkills | None = None,
        builtin_root: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.project_root = self.workspace / ".fakuicode" / "skills"
        self.user_root = user_root
        self.builtin_root = builtin_root
        self.fetcher = fetcher or GitHubSkillFetcher()
        self.refresh = refresh

    def close(self) -> None:
        close = getattr(self.fetcher, "close", None)
        if callable(close):
            close()

    def install(
        self,
        request: SkillInstallRequest,
        *,
        confirm: InstallConfirmation,
        cancel_event: Event | None = None,
    ) -> SkillInstallResult:
        source = parse_install_source(request.source, skill=request.skill)
        package = self.fetcher.fetch(source, cancel_event=cancel_event)
        name = _validated_skill_name(package.name)
        target_root = self.project_root if request.scope is SkillInstallScope.PROJECT else self.user_root
        _assert_safe_install_root(target_root)
        target = target_root / name
        replacing = target.exists()
        if replacing and not request.replace:
            raise SkillInstallError(
                f"Skill '{name}' already exists in the target scope; use --replace to replace it."
            )
        if replacing and (not target.is_dir() or target.is_symlink()):
            raise SkillInstallError("Existing Skill target is not a safe directory.")
        target_identity = _directory_identity(target) if replacing else None
        staging_base = _nearest_existing_directory(
            self.workspace if request.scope is SkillInstallScope.PROJECT else self.user_root
        )
        with tempfile.TemporaryDirectory(prefix=".fakuicode-skill-", dir=staging_base) as temporary:
            staged = Path(temporary) / name
            staged.mkdir()
            self._write_upstream(staged, package)
            try:
                upstream = parse_skill_package(staged, _scope_source(request.scope))
            except SkillParseError as error:
                raise SkillInstallError(f"Downloaded Skill is invalid: {error}") from error
            if upstream.name != name:
                raise SkillInstallError("Downloaded Skill name does not match its source directory.")
            initial_preset = request.preset or _default_preset(name)
            preview = self._preview(
                request.scope,
                package,
                upstream,
                target,
                initial_preset,
                replacing,
            )
            decision = confirm(preview)
            if not isinstance(decision, SkillInstallDecision) or not isinstance(
                decision.preset, SkillInstallPreset
            ):
                raise SkillInstallError("Skill installation confirmation was invalid.")
            if not decision.approved:
                return SkillInstallResult(False, name, f"Skill '{name}' installation cancelled.")
            _check_cancelled(cancel_event)
            self._write_receipt(staged, package, upstream, decision.preset)
            try:
                effective = parse_skill_package(staged, _scope_source(request.scope))
            except SkillParseError as error:
                raise SkillInstallError(f"Effective Skill package is invalid: {error}") from error
            if effective.visible_tools != _PRESET_TOOLS[decision.preset]:
                raise SkillInstallError("Effective Skill tool preset is invalid.")
            _check_cancelled(cancel_event)
            self._commit(staged, target, name, target_identity)
        return SkillInstallResult(True, name, f"Skill '{name}' installed.", target)

    @staticmethod
    def _write_upstream(staged: Path, package: RemoteSkillPackage) -> None:
        if len(package.files) + 1 > DEFAULT_INSTRUCTION_LIMITS.max_file_targets:
            raise SkillInstallError("Skill package contains too many files.")
        names: set[str] = set()
        for relative, content in package.files.items():
            normalized = PurePosixPath(relative).as_posix()
            folded = normalized.casefold()
            if (
                not _safe_tree_path(relative)
                or folded == _RESERVED_RECEIPT.casefold()
                or folded in names
            ):
                raise SkillInstallError(f"Skill package file path is unsafe: {relative}")
            if not isinstance(content, bytes) or len(content) > DEFAULT_INSTRUCTION_LIMITS.max_source_bytes:
                raise SkillInstallError(f"Skill file is too large or invalid: {relative}")
            names.add(folded)
            destination = staged.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    def _preview(
        self,
        scope: SkillInstallScope,
        package: RemoteSkillPackage,
        upstream: SkillDefinition,
        target: Path,
        preset: SkillInstallPreset,
        replacing: bool,
    ) -> SkillInstallPreview:
        shadows: list[str] = []
        if scope is SkillInstallScope.PROJECT:
            if (self.user_root / upstream.name).exists():
                shadows.append("user")
            if self.builtin_root is not None and (self.builtin_root / upstream.name).exists():
                shadows.append("builtin")
        elif self.builtin_root is not None and (self.builtin_root / upstream.name).exists():
            shadows.append("builtin")
        return SkillInstallPreview(
            upstream.name,
            upstream.description,
            upstream.license,
            package.source.requested_url,
            package.source.canonical_url,
            package.revision,
            package.skill_path,
            target,
            scope,
            preset,
            _PRESET_TOOLS[preset],
            tuple(sorted(package.files)),
            package.total_bytes,
            any(PurePosixPath(path).parts[0] == "scripts" for path in package.files),
            tuple(tool.name for tool in upstream.tools),
            replacing,
            tuple(shadows),
            upstream.allowed_tools,
        )

    @staticmethod
    def _write_receipt(
        staged: Path,
        package: RemoteSkillPackage,
        upstream: SkillDefinition,
        preset: SkillInstallPreset,
    ) -> None:
        upstream_fingerprint = fingerprint_upstream(staged)
        extension = {
            "invocation": upstream.invocation.value,
            "visible-tools": list(_PRESET_TOOLS[preset]),
            "execution": upstream.execution.value,
            "history-turns": upstream.history_turns,
            "profile": upstream.profile,
        }
        receipt = {
            "schema-version": 1,
            "requested-url": package.source.requested_url,
            "source-url": package.source.canonical_url,
            "revision": package.revision,
            "skill-path": package.skill_path,
            "upstream-fingerprint": upstream_fingerprint,
            "fakuicode": extension,
        }
        receipt_root = staged / ".fakuicode"
        receipt_root.mkdir(exist_ok=True)
        (receipt_root / "install.yaml").write_text(
            yaml.safe_dump(receipt, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _commit(
        self,
        staged: Path,
        target: Path,
        name: str,
        expected_identity: tuple[int, int, int] | None,
    ) -> None:
        target_root = target.parent
        _assert_safe_install_root(target_root)
        target_root.mkdir(parents=True, exist_ok=True)
        _assert_safe_install_root(target_root)
        backup: Path | None = None
        installed = False
        if target.exists() != (expected_identity is not None):
            raise SkillInstallError("Skill installation target changed after preview.")
        if target.exists():
            if not target.is_dir() or _is_reparse(target) or target.resolve() != target.absolute():
                raise SkillInstallError("Existing Skill target is not a safe directory.")
            if _directory_identity(target) != expected_identity:
                raise SkillInstallError("Skill installation target changed after preview.")
            backup = target_root / f".{name}.backup-{uuid4().hex}"
            target.replace(backup)
        try:
            staged.replace(target)
            installed = True
            snapshot = self.refresh() if self.refresh is not None else None
            if snapshot is not None:
                current = snapshot.skills.get(name)
                if current is None or current.package_path.resolve() != target.resolve():
                    raise SkillInstallError("Installed Skill was rejected during hot refresh.")
        except Exception as error:
            if installed and target.exists():
                shutil.rmtree(target)
            if backup is not None and backup.exists():
                backup.replace(target)
            if self.refresh is not None:
                try:
                    self.refresh()
                except Exception:
                    pass
            raise SkillInstallError("Skill installation failed and was rolled back.") from error
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def parse_install_source(url: str, *, skill: str | None = None) -> SkillInstallSource:
    """Normalize one supported HTTPS Skill URL without issuing a request."""

    if not isinstance(url, str) or not url.strip():
        raise SkillInstallError("Skill source URL is required.")
    raw = url.strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as error:
        raise SkillInstallError("Skill source URL is invalid.") from error
    host = (parsed.hostname or "").casefold()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SkillInstallError("Only canonical HTTPS Skill URLs are supported.")
    parts = tuple(part for part in parsed.path.split("/") if part)
    requested_skill = _validated_skill_name(skill) if skill is not None else None

    if host in _SKILLS_HOSTS:
        if len(parts) != 3:
            raise SkillInstallError("skills.sh URLs must identify exactly one GitHub-backed Skill.")
        owner, repo, url_skill = parts
        _validate_repository(owner, repo)
        url_skill = _validated_skill_name(url_skill)
        if requested_skill is not None and requested_skill != url_skill:
            raise SkillInstallError("The requested Skill name conflicts with the skills.sh URL.")
        return SkillInstallSource(raw, owner, repo, url_skill)

    if host != _GITHUB_HOST or len(parts) < 2:
        raise SkillInstallError("Only public skills.sh and GitHub sources are supported.")
    owner, repo = parts[:2]
    if repo.endswith(".git"):
        repo = repo[:-4]
    _validate_repository(owner, repo)
    if len(parts) == 2:
        return SkillInstallSource(raw, owner, repo, requested_skill)
    if len(parts) < 5 or parts[2] != "tree":
        raise SkillInstallError("GitHub URLs must name a repository or a tree directory.")
    ref = parts[3]
    path_parts = parts[4:]
    if not _PATH_PART.fullmatch(ref) or any(not _safe_local_part(part) for part in path_parts):
        raise SkillInstallError("GitHub ref or Skill path is invalid.")
    skill_path = "/".join(path_parts)
    inferred = _validated_skill_name(path_parts[-1])
    if requested_skill is not None and requested_skill != inferred:
        raise SkillInstallError("The requested Skill name conflicts with the GitHub tree path.")
    return SkillInstallSource(raw, owner, repo, requested_skill or inferred, ref, skill_path)


class GitHubSkillFetcher:
    """Fetch one bounded public Skill subtree pinned to a GitHub commit."""

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self.client = client or httpx.Client(
            follow_redirects=False,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "fakuicode-skill-installer",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def fetch(
        self,
        source: SkillInstallSource,
        *,
        cancel_event: Event | None = None,
    ) -> RemoteSkillPackage:
        self._check_cancelled(cancel_event)
        ref = source.ref
        if ref is None:
            repository = self._get_json(
                f"https://api.github.com/repos/{source.owner}/{source.repo}",
                cancel_event=cancel_event,
            )
            ref = repository.get("default_branch") if isinstance(repository, Mapping) else None
            if not isinstance(ref, str) or not ref:
                raise SkillInstallError("GitHub repository did not provide a default branch.")
        commit = self._get_json(
            f"https://api.github.com/repos/{source.owner}/{source.repo}/commits/{quote(ref, safe='')}",
            cancel_event=cancel_event,
        )
        revision, tree_sha = _commit_identity(commit)
        tree_document = self._get_json(
            f"https://api.github.com/repos/{source.owner}/{source.repo}/git/trees/{tree_sha}",
            params={"recursive": "1"},
            cancel_event=cancel_event,
        )
        if not isinstance(tree_document, Mapping) or tree_document.get("truncated") is not False:
            raise SkillInstallError("GitHub repository tree is incomplete or invalid.")
        entries = tree_document.get("tree")
        if not isinstance(entries, list):
            raise SkillInstallError("GitHub repository tree is invalid.")
        normalized = tuple(_tree_entry(item) for item in entries)
        skill_path, fallback_name = _select_skill_path(source, normalized)
        selected = _selected_files(skill_path, normalized)
        files: dict[str, bytes] = {}
        for remote_path, relative_path, declared_size in selected:
            self._check_cancelled(cancel_event)
            files[relative_path] = self._download_file(
                source,
                revision,
                remote_path,
                declared_size,
                cancel_event=cancel_event,
            )
        name = _downloaded_skill_name(files.get("SKILL.md"), fallback_name)
        if source.requested_skill is not None and name != source.requested_skill:
            raise SkillInstallError("Downloaded Skill name does not match the requested Skill.")
        return RemoteSkillPackage(
            source,
            name,
            revision,
            skill_path or ".",
            MappingProxyType(files),
        )

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cancel_event: Event | None,
    ) -> object:
        self._check_cancelled(cancel_event)
        try:
            response = self.client.get(url, params=params)
        except httpx.HTTPError as error:
            raise SkillInstallError("GitHub request failed.") from error
        self._validate_response(response)
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as error:
            raise SkillInstallError("GitHub returned invalid JSON.") from error

    def _download_file(
        self,
        source: SkillInstallSource,
        revision: str,
        remote_path: str,
        declared_size: int,
        *,
        cancel_event: Event | None,
    ) -> bytes:
        limit = DEFAULT_INSTRUCTION_LIMITS.max_source_bytes
        if declared_size > limit:
            raise SkillInstallError(f"Skill file is too large: {remote_path}")
        encoded_path = "/".join(quote(part, safe="") for part in PurePosixPath(remote_path).parts)
        url = (
            f"https://raw.githubusercontent.com/{source.owner}/{source.repo}/"
            f"{revision}/{encoded_path}"
        )
        content = bytearray()
        try:
            with self.client.stream("GET", url) as response:
                self._validate_response(response)
                for chunk in response.iter_bytes():
                    self._check_cancelled(cancel_event)
                    content.extend(chunk)
                    if len(content) > limit:
                        raise SkillInstallError(f"Skill file is too large: {remote_path}")
        except SkillInstallError:
            raise
        except httpx.HTTPError as error:
            raise SkillInstallError("GitHub file download failed.") from error
        return bytes(content)

    @staticmethod
    def _validate_response(response: httpx.Response) -> None:
        if response.is_redirect:
            raise SkillInstallError("GitHub redirected to an unsupported download host.")
        if response.status_code in {403, 429}:
            reset = response.headers.get("x-ratelimit-reset") or response.headers.get("retry-after")
            suffix = f" Retry after {reset}." if reset else ""
            raise SkillInstallError("GitHub rate limit was reached." + suffix)
        if response.status_code != 200:
            raise SkillInstallError(f"GitHub request failed with status {response.status_code}.")

    @staticmethod
    def _check_cancelled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelled()


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    type: str
    mode: str
    size: int


def _tree_entry(value: object) -> _TreeEntry:
    if not isinstance(value, Mapping):
        raise SkillInstallError("GitHub repository tree entry is invalid.")
    path = value.get("path")
    kind = value.get("type")
    mode = value.get("mode")
    size = value.get("size", 0)
    if (
        not isinstance(path, str)
        or not _safe_tree_path(path)
        or not isinstance(kind, str)
        or not isinstance(mode, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise SkillInstallError("GitHub repository tree entry is invalid.")
    return _TreeEntry(path, kind, mode, size)


def _commit_identity(value: object) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise SkillInstallError("GitHub commit response is invalid.")
    revision = value.get("sha")
    commit = value.get("commit")
    tree = commit.get("tree") if isinstance(commit, Mapping) else None
    tree_sha = tree.get("sha") if isinstance(tree, Mapping) else None
    if (
        not isinstance(revision, str)
        or _COMMIT.fullmatch(revision) is None
        or not isinstance(tree_sha, str)
        or _COMMIT.fullmatch(tree_sha) is None
    ):
        raise SkillInstallError("GitHub commit response is invalid.")
    return revision, tree_sha


def _select_skill_path(
    source: SkillInstallSource,
    entries: tuple[_TreeEntry, ...],
) -> tuple[str, str]:
    candidates = sorted(
        (entry.path[: -len("/SKILL.md")] if entry.path != "SKILL.md" else "")
        for entry in entries
        if entry.type == "blob"
        and entry.mode in _REGULAR_MODES
        and (entry.path == "SKILL.md" or entry.path.endswith("/SKILL.md"))
    )
    if source.skill_path is not None:
        candidates = [path for path in candidates if path == source.skill_path]
    if source.requested_skill is not None:
        nested = [path for path in candidates if path and PurePosixPath(path).name == source.requested_skill]
        candidates = nested or [path for path in candidates if not path]
    if not candidates:
        raise SkillInstallError("No matching Skill was found in the GitHub repository.")
    if len(candidates) > 1 or (source.requested_skill is None and source.skill_path is None and len(candidates) != 1):
        names = ", ".join(sorted({PurePosixPath(path).name for path in candidates}))
        raise SkillInstallError(f"Repository contains multiple Skills; select one explicitly: {names}")
    path = candidates[0]
    fallback = source.requested_skill or (PurePosixPath(path).name if path else source.repo.casefold())
    return path, _validated_skill_name(fallback)


def _selected_files(
    skill_path: str,
    entries: tuple[_TreeEntry, ...],
) -> tuple[tuple[str, str, int], ...]:
    prefix = skill_path.rstrip("/") + "/" if skill_path else ""
    selected: list[tuple[str, str, int]] = []
    folded_paths: set[str] = set()
    for entry in entries:
        if prefix and not entry.path.startswith(prefix):
            continue
        relative = entry.path[len(prefix) :] if prefix else entry.path
        if entry.mode in _UNSAFE_MODES or entry.type == "commit":
            raise SkillInstallError(f"Skill package contains an unsafe link or submodule: {relative}")
        if entry.type == "tree" and entry.mode in _TREE_MODES:
            continue
        if entry.type != "blob" or entry.mode not in _REGULAR_MODES:
            raise SkillInstallError(f"Skill package contains an unsafe repository entry: {relative}")
        folded = PurePosixPath(relative).as_posix().casefold()
        if folded == _RESERVED_RECEIPT.casefold():
            raise SkillInstallError("Skill package occupies the reserved fakuiCode install receipt path.")
        if folded in folded_paths:
            raise SkillInstallError(f"Skill package contains a case-insensitive path collision: {relative}")
        folded_paths.add(folded)
        selected.append((entry.path, relative, entry.size))
    if not any(relative == "SKILL.md" for _, relative, _ in selected):
        raise SkillInstallError("Skill package does not contain SKILL.md.")
    if len(selected) + 1 > DEFAULT_INSTRUCTION_LIMITS.max_file_targets:
        raise SkillInstallError("Skill package contains too many files.")
    return tuple(sorted(selected, key=lambda item: item[1]))


def _downloaded_skill_name(content: bytes | None, fallback: str) -> str:
    if content is None:
        return fallback
    try:
        text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if not text.startswith("---\n"):
            return fallback
        end = text.find("\n---\n", 4)
        raw = yaml.safe_load(text[4:end]) if end >= 0 else None
    except (UnicodeDecodeError, yaml.YAMLError):
        return fallback
    if isinstance(raw, Mapping) and isinstance(raw.get("name"), str):
        return _validated_skill_name(raw["name"])
    return fallback


def _validated_skill_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or _NAME.fullmatch(value) is None
        or value.casefold() in _WINDOWS_RESERVED
    ):
        raise SkillInstallError("Skill name is invalid.")
    return value


def _validate_repository(owner: str, repo: str) -> None:
    if (
        _REPOSITORY_PART.fullmatch(owner) is None
        or _REPOSITORY_PART.fullmatch(repo) is None
        or "." in owner
        or owner in {".", ".."}
        or repo in {".", ".."}
    ):
        raise SkillInstallError("GitHub repository identity is invalid.")


def _safe_tree_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and all(_safe_local_part(part) for part in path.parts)
    )


def _safe_local_part(part: str) -> bool:
    return (
        part not in {"", ".", ".."}
        and _PATH_PART.fullmatch(part) is not None
        and not part.endswith(".")
        and part.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED
    )


def _default_preset(name: str) -> SkillInstallPreset:
    return SkillInstallPreset.CODING if name == "frontend-design" else SkillInstallPreset.INSTRUCTION


def _scope_source(scope: SkillInstallScope) -> SkillSource:
    return SkillSource.PROJECT if scope is SkillInstallScope.PROJECT else SkillSource.USER


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise SkillInstallError("Skill installation target has no accessible parent directory.")
        current = parent
    if not current.is_dir():
        raise SkillInstallError("Skill installation target parent is not a directory.")
    return current


def _assert_safe_install_root(path: Path) -> None:
    absolute = path.absolute()
    existing = _nearest_existing_directory(absolute)
    if _is_reparse(existing):
        raise SkillInstallError("Skill installation target traverses a reparse point.")
    try:
        if existing.resolve(strict=True) != existing.absolute():
            raise SkillInstallError("Skill installation target traverses a symbolic link.")
    except OSError as error:
        raise SkillInstallError("Skill installation target cannot be resolved safely.") from error
    current = existing
    relative_parts = absolute.relative_to(existing).parts
    for part in relative_parts:
        current = current / part
        if current.exists() and (not current.is_dir() or _is_reparse(current)):
            raise SkillInstallError("Skill installation target is not a safe directory.")


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _directory_identity(path: Path) -> tuple[int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise SkillInstallError("Existing Skill target cannot be inspected safely.") from error
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RequestCancelled()
