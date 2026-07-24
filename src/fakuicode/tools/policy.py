"""Mandatory workspace and command policy checks for untrusted model output."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

from fakuicode.errors import ToolPolicyError
from fakuicode.worktrees.models import PathMapping


_SENSITIVE_NAMES = {"fakuicode.yaml"}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".crt", ".cer", ".p12", ".pfx"}
_SENSITIVE_RELATIVE_PATHS = {
    (".fakuicode", "permissions.yaml"),
    (".fakuicode", "permissions.local.yaml"),
}
_CONTEXT_ARTIFACT_PREFIX = (".fakuicode", "context-artifacts")
_MANAGED_PREFIXES = {
    (".fakuicode", "worktrees"),
    (".fakuicode", "worktree-state"),
}


class WorkspacePolicy:
    """Constrain every tool operation to one explicitly selected workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        mappings: Sequence[PathMapping] = (),
    ) -> None:
        self.workspace = workspace.resolve()
        normalized: list[PathMapping] = []
        for mapping in mappings:
            alias = Path(os.path.abspath(mapping.alias))
            try:
                alias.relative_to(self.workspace)
            except ValueError as error:
                raise ValueError("Path mapping alias must be inside the workspace.") from error
            target = self._resolve_candidate(mapping.target)
            normalized.append(PathMapping(alias, target, mapping.access))
        aliases = sorted((item.alias for item in normalized), key=lambda item: len(item.parts))
        for index, alias in enumerate(aliases):
            if any(_is_relative_to(other, alias) for other in aliases[index + 1 :]):
                raise ValueError("Path mapping aliases cannot overlap.")
        self.mappings = tuple(normalized)

    def resolve_path(self, target: str, *, allow_context_artifact_read: bool = False) -> Path:
        raw_candidate = self.workspace / target if not Path(target).is_absolute() else Path(target)
        lexical_candidate = Path(os.path.abspath(raw_candidate))
        mapping, mapped_lexical = self._mapping_for(lexical_candidate)
        if mapping is not None:
            if mapping.access == "read_only" and not allow_context_artifact_read:
                raise ToolPolicyError("The mapped workspace path is read-only.")
            if self._is_sensitive(
                mapped_lexical,
                allow_context_artifact_read=allow_context_artifact_read,
            ):
                raise ToolPolicyError("sensitive workspace files are not available to tools.")
            relative = mapped_lexical.relative_to(mapping.alias)
            candidate = self._resolve_candidate(mapping.target / relative)
            try:
                candidate.relative_to(mapping.target)
            except ValueError as error:
                raise ToolPolicyError("Path is outside the workspace mapping.") from error
            return candidate
        if self._is_sensitive(
            lexical_candidate,
            allow_context_artifact_read=allow_context_artifact_read,
        ):
            raise ToolPolicyError("sensitive workspace files are not available to tools.")
        candidate = self._resolve_candidate(raw_candidate)
        try:
            candidate.relative_to(self.workspace)
        except ValueError as error:
            raise ToolPolicyError("Path is outside the workspace.") from error
        if self._is_sensitive(
            candidate,
            allow_context_artifact_read=allow_context_artifact_read,
        ):
            raise ToolPolicyError("sensitive workspace files are not available to tools.")
        return candidate

    @staticmethod
    def _resolve_candidate(candidate: Path) -> Path:
        missing: list[str] = []
        ancestor = candidate
        while not ancestor.exists() and ancestor != ancestor.parent:
            missing.append(ancestor.name)
            ancestor = ancestor.parent
        try:
            resolved = ancestor.resolve(strict=True)
        except OSError as error:
            raise ToolPolicyError("Unable to resolve the requested workspace path.") from error
        return resolved.joinpath(*reversed(missing)).resolve(strict=False)

    def relative_target(self, candidate: Path) -> str:
        absolute = self._resolve_candidate(candidate)
        for mapping in self.mappings:
            try:
                relative = absolute.relative_to(mapping.target)
            except ValueError:
                continue
            alias = mapping.alias.relative_to(self.workspace)
            combined = alias / relative
            return combined.as_posix() if combined.parts else "."
        relative = absolute.relative_to(self.workspace)
        return relative.as_posix() if relative.parts else "."

    def _mapping_for(self, lexical: Path) -> tuple[PathMapping | None, Path]:
        for mapping in self.mappings:
            if _is_relative_to(lexical, mapping.alias):
                return mapping, lexical
        for mapping in self.mappings:
            if _is_relative_to(lexical, mapping.target):
                relative = lexical.relative_to(mapping.target)
                return mapping, mapping.alias / relative
        return None, lexical

    def _is_sensitive(self, candidate: Path, *, allow_context_artifact_read: bool) -> bool:
        try:
            relative = candidate.relative_to(self.workspace)
        except ValueError:
            return True
        name = candidate.name.casefold()
        folded_parts = tuple(part.casefold() for part in relative.parts)
        is_context_artifact = folded_parts[:2] == _CONTEXT_ARTIFACT_PREFIX
        is_managed_root = any(
            folded_parts[: len(prefix)] == prefix for prefix in _MANAGED_PREFIXES
        )
        return (
            ".git" in set(folded_parts)
            or folded_parts in _SENSITIVE_RELATIVE_PATHS
            or is_managed_root
            or is_context_artifact and not allow_context_artifact_read
            or name in _SENSITIVE_NAMES
            or name == ".env"
            or name.startswith(".env.")
            or candidate.suffix.casefold() in _SENSITIVE_SUFFIXES
        )

    def validate_command(self, command: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(part.strip() for part in command)
        if not normalized or not normalized[0] or any(not part for part in normalized):
            raise ToolPolicyError("Command must contain non-empty arguments.")
        return normalized


def _is_relative_to(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
