"""Mandatory workspace and command policy checks for untrusted model output."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

from fakuicode.errors import ToolPolicyError


_SENSITIVE_NAMES = {"fakuicode.yaml"}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".crt", ".cer", ".p12", ".pfx"}
_SENSITIVE_RELATIVE_PATHS = {
    (".fakuicode", "permissions.yaml"),
    (".fakuicode", "permissions.local.yaml"),
}
_CONTEXT_ARTIFACT_PREFIX = (".fakuicode", "context-artifacts")


class WorkspacePolicy:
    """Constrain every tool operation to one explicitly selected workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def resolve_path(self, target: str, *, allow_context_artifact_read: bool = False) -> Path:
        raw_candidate = self.workspace / target if not Path(target).is_absolute() else Path(target)
        lexical_candidate = Path(os.path.abspath(raw_candidate))
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
        relative = candidate.relative_to(self.workspace)
        return relative.as_posix() if relative.parts else "."

    def _is_sensitive(self, candidate: Path, *, allow_context_artifact_read: bool) -> bool:
        try:
            relative = candidate.relative_to(self.workspace)
        except ValueError:
            return True
        name = candidate.name.casefold()
        folded_parts = tuple(part.casefold() for part in relative.parts)
        is_context_artifact = folded_parts[:2] == _CONTEXT_ARTIFACT_PREFIX
        return (
            ".git" in set(folded_parts)
            or folded_parts in _SENSITIVE_RELATIVE_PATHS
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
