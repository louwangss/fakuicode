"""Immutable models for project-instruction snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import unicodedata


def sanitize_instruction_metadata(value: str) -> str:
    """Replace terminal-control and line-separator characters in safe metadata."""

    return "".join(
        "�" if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"} else character
        for character in value
    )


class InstructionScope(StrEnum):
    """A fixed project-instruction source, from low to high priority."""

    USER = "user"
    PROJECT = "project"
    PROJECT_LOCAL = "project_local"


class InstructionDiagnosticCode(StrEnum):
    """Stable, non-content diagnostic codes emitted by the loader."""

    INVALID_INCLUDE = "invalid_include"
    PATH_REJECTED = "path_rejected"
    NOT_MARKDOWN = "not_markdown"
    NOT_REGULAR_FILE = "not_regular_file"
    FILE_NOT_FOUND = "file_not_found"
    FILE_PERMISSION_DENIED = "file_permission_denied"
    FILE_READ_FAILED = "file_read_failed"
    FILE_TOO_LARGE = "file_too_large"
    INVALID_UTF8 = "invalid_utf8"
    INCLUDE_CYCLE = "include_cycle"
    DEPTH_LIMIT = "depth_limit"
    FILE_TARGET_LIMIT = "file_target_limit"
    INCLUDE_BUDGET = "include_budget"
    MAIN_TRUNCATED = "main_truncated"


class InstructionLoadFailure(StrEnum):
    """Stable top-level loader failures without path or exception details."""

    LOADER_FAILED = "loader_failed"


@dataclass(frozen=True)
class InstructionDiagnostic:
    """Safe metadata describing one rejected instruction source or include."""

    code: InstructionDiagnosticCode
    scope: InstructionScope
    source: str
    line: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", sanitize_instruction_metadata(self.source))


@dataclass(frozen=True)
class InstructionSnapshot:
    """One immutable, session-scoped instruction payload and its safe metadata."""

    text: str
    loaded_layers: tuple[InstructionScope, ...]
    processed_target_count: int
    diagnostics: tuple[InstructionDiagnostic, ...]
    global_failure: InstructionLoadFailure | None = None

    @property
    def byte_count(self) -> int:
        """Return the exact UTF-8 size of the model-visible instruction payload."""

        return len(self.text.encode("utf-8"))

    @property
    def warning_count(self) -> int:
        """Return the number of structured loader warnings."""

        return len(self.diagnostics) + int(self.global_failure is not None)

    @classmethod
    def empty(cls) -> "InstructionSnapshot":
        """Return the no-instruction snapshot used when loading is disabled."""

        return cls(text="", loaded_layers=(), processed_target_count=0, diagnostics=())

    @classmethod
    def failed(cls) -> "InstructionSnapshot":
        """Return a safe fallback for unexpected top-level loader failures."""

        return cls(
            text="",
            loaded_layers=(),
            processed_target_count=0,
            diagnostics=(),
            global_failure=InstructionLoadFailure.LOADER_FAILED,
        )


@dataclass(frozen=True)
class InstructionLimits:
    """Fixed resource limits for one instruction snapshot load."""

    max_include_depth: int = 5
    max_file_targets: int = 32
    max_payload_bytes: int = 32 * 1024

    def __post_init__(self) -> None:
        if any(
            limit <= 0
            for limit in (
                self.max_include_depth,
                self.max_file_targets,
                self.max_payload_bytes,
            )
        ):
            raise ValueError("Instruction limits must be positive.")

    @property
    def max_source_bytes(self) -> int:
        """Return the largest complete source file that can be accepted."""

        return 2 * self.max_payload_bytes + 3

    @property
    def max_read_bytes(self) -> int:
        """Return the bounded read size, including one byte for overflow detection."""

        return self.max_source_bytes + 1


DEFAULT_INSTRUCTION_LIMITS = InstructionLimits()
