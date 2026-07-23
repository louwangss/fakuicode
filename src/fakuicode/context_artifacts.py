"""Safe, durable storage for complete tool results removed from active context."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from fakuicode.context import approximate_token_count


_SAFE_CONVERSATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_TOMBSTONE_NAME = re.compile(r"\.deleting-([A-Za-z0-9][A-Za-z0-9_-]{0,127})-([0-9a-f]{32})\Z")


@dataclass(frozen=True)
class ContextArtifactRef:
    conversation_id: str
    source_sequence: int
    content_sha256: str
    byte_size: int
    estimated_tokens: int
    read_path: str
    success: bool
    newly_created: bool = field(compare=False)


class ContextArtifactStore:
    """Write artifacts beneath one validated workspace conversation directory."""

    def __init__(self, workspace: Path, conversation_id: str) -> None:
        if _SAFE_CONVERSATION_ID.fullmatch(conversation_id) is None:
            raise ValueError("conversation_id contains unsafe path characters.")
        self.workspace = workspace.resolve(strict=True)
        self.conversation_id = conversation_id
        self.root = self.workspace / ".fakuicode" / "context-artifacts"
        self.conversation_dir = self.root / conversation_id
        self._assert_within_workspace(self.root)
        self._assert_within_workspace(self.conversation_dir)

    def write_tool_result(
        self,
        *,
        source_sequence: int,
        output: str,
        success: bool,
        provider_call_id: str | None = None,
    ) -> ContextArtifactRef:
        """Atomically persist exact UTF-8 bytes; provider_call_id is never a path input."""

        del provider_call_id
        if source_sequence < 0:
            raise ValueError("source_sequence cannot be negative.")
        encoded = output.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        filename = f"{source_sequence:020d}-{digest[:20]}.txt"
        self.conversation_dir.mkdir(parents=True, exist_ok=True)
        self._assert_within_workspace(self.conversation_dir)
        target = self.conversation_dir / filename
        self._assert_within_workspace(target)

        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise OSError("Existing context artifact failed its integrity check.")
            return self._reference(
                source_sequence,
                digest,
                encoded,
                target,
                success,
                newly_created=False,
            )

        temporary = self.conversation_dir / f".{filename}.{uuid4().hex}.tmp"
        self._assert_within_workspace(temporary)
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._atomic_replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return self._reference(
            source_sequence,
            digest,
            encoded,
            target,
            success,
            newly_created=True,
        )

    def resolve_read_path(self, reference: ContextArtifactRef) -> Path:
        """Resolve one reference only after rechecking its tenant and integrity."""

        if reference.conversation_id != self.conversation_id:
            raise ValueError("Context artifact belongs to another conversation.")
        relative = Path(reference.read_path)
        if relative.is_absolute():
            raise ValueError("Context artifact reference must be workspace-relative.")
        candidate = self.workspace / relative
        self._assert_within_conversation(candidate)
        try:
            encoded = candidate.read_bytes()
        except OSError:
            raise
        if hashlib.sha256(encoded).hexdigest() != reference.content_sha256:
            raise OSError("Context artifact failed its integrity check.")
        return candidate

    def stage_conversation_deletion(self) -> Path | None:
        """Atomically hide this conversation's artifact directory before DB deletion."""

        if not self.conversation_dir.exists():
            return None
        self._assert_within_conversation(self.conversation_dir)
        tombstone = self.root / f".deleting-{self.conversation_id}-{uuid4().hex}"
        self._validate_tombstone(tombstone, require_exists=False)
        os.replace(self.conversation_dir, tombstone)
        return tombstone

    def restore_staged_deletion(self, tombstone: Path) -> None:
        """Roll a staged deletion back when the database transaction fails."""

        self._validate_tombstone(tombstone, require_exists=True)
        if self.conversation_dir.exists():
            raise OSError("Cannot restore artifacts over an existing conversation directory.")
        os.replace(tombstone, self.conversation_dir)

    def purge_staged_deletion(self, tombstone: Path) -> None:
        """Permanently remove one validated tombstone after database deletion."""

        self._validate_tombstone(tombstone, require_exists=True)
        shutil.rmtree(tombstone)

    def cleanup_stale_tombstones(
        self,
        *,
        retained_conversation_ids: set[str] | None = None,
    ) -> int:
        """Reconcile validated tombstones against conversations still in the database."""

        if not self.root.exists():
            return 0
        reconciled = 0
        retained = retained_conversation_ids or set()
        for candidate in self.root.iterdir():
            match = _TOMBSTONE_NAME.fullmatch(candidate.name)
            if match is None:
                continue
            self._validate_tombstone(
                candidate,
                require_exists=True,
                require_current_conversation=False,
            )
            conversation_id = match.group(1)
            if conversation_id in retained:
                destination = self.root / conversation_id
                self._assert_within_workspace(destination)
                if destination.exists():
                    continue
                os.replace(candidate, destination)
            else:
                shutil.rmtree(candidate)
            reconciled += 1
        return reconciled

    @staticmethod
    def _atomic_replace(source: Path, target: Path) -> None:
        os.replace(source, target)

    def _reference(
        self,
        source_sequence: int,
        digest: str,
        encoded: bytes,
        target: Path,
        success: bool,
        *,
        newly_created: bool,
    ) -> ContextArtifactRef:
        return ContextArtifactRef(
            conversation_id=self.conversation_id,
            source_sequence=source_sequence,
            content_sha256=digest,
            byte_size=len(encoded),
            estimated_tokens=approximate_token_count(encoded.decode("utf-8")),
            read_path=target.relative_to(self.workspace).as_posix(),
            success=success,
            newly_created=newly_created,
        )

    def _assert_within_workspace(self, candidate: Path) -> None:
        try:
            candidate.resolve(strict=False).relative_to(self.workspace)
        except (OSError, ValueError) as error:
            raise ValueError("Context artifact path escapes the workspace.") from error

    def _assert_within_conversation(self, candidate: Path) -> None:
        self._assert_within_workspace(candidate)
        try:
            candidate.resolve(strict=False).relative_to(self.conversation_dir.resolve(strict=False))
        except (OSError, ValueError) as error:
            raise ValueError("Context artifact path escapes its conversation directory.") from error

    def _validate_tombstone(
        self,
        candidate: Path,
        *,
        require_exists: bool,
        require_current_conversation: bool = True,
    ) -> None:
        if candidate.parent != self.root or _TOMBSTONE_NAME.fullmatch(candidate.name) is None:
            raise ValueError("Invalid context artifact tombstone.")
        match = _TOMBSTONE_NAME.fullmatch(candidate.name)
        if (
            match is None
            or require_current_conversation
            and match.group(1) != self.conversation_id
        ):
            raise ValueError("Context artifact tombstone belongs to another conversation.")
        self._assert_within_workspace(candidate)
        if require_exists and (not candidate.exists() or candidate.is_symlink() or not candidate.is_dir()):
            raise ValueError("Context artifact tombstone is missing or unsafe.")
