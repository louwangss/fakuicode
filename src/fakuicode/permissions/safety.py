"""Minimal, non-configurable command tripwires for direct disasters."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import re


_SAFE_ARGUMENT = re.compile(r"[A-Za-z0-9_./:@%+=,-]+", re.ASCII)
_EXECUTABLE_PREFIX = r'(?:"[^\"]*[\\/]|\S*[\\/])?'
_SHELL_PATTERN = re.compile(
    rf"^{_EXECUTABLE_PREFIX}(?:powershell|pwsh|cmd|bash|sh|zsh|fish|wsl)(?:\.exe)?\"?(?:\s|$)",
    re.IGNORECASE,
)
_FILESYSTEM_PATTERN = re.compile(
    rf"^{_EXECUTABLE_PREFIX}(?:mkfs(?:\.[A-Za-z0-9_-]+)?|format(?:\.com)?|diskpart)(?:\.exe)?\"?(?:\s|$)",
    re.IGNORECASE,
)
_BLOCK_DEVICE_PATTERN = re.compile(r"^dd(?:\.exe)?(?:\s|$).*\bof=/dev/", re.IGNORECASE)
_SYSTEM_PATTERN = re.compile(
    rf"^{_EXECUTABLE_PREFIX}(?:shutdown|reboot|halt|poweroff)(?:\.exe)?\"?(?:\s|$)",
    re.IGNORECASE,
)


def serialize_command(command: Sequence[str]) -> str:
    """Return one deterministic, displayable representation of an argv sequence."""

    return " ".join(_serialize_argument(argument) for argument in command)


def _serialize_argument(argument: str) -> str:
    if argument and _SAFE_ARGUMENT.fullmatch(argument):
        return argument
    return json.dumps(argument, ensure_ascii=False, separators=(",", ":"))


class DangerousCommandGuard:
    """Reject a deliberately small set of directly recognizable disasters."""

    def __init__(self, workspace: Path, *, home: Path | None = None) -> None:
        protected_targets = {"/", "~", ".", "./", str(workspace.resolve()), str((home or Path.home()).resolve())}
        alternatives = "|".join(
            sorted((re.escape(serialize_command((target,))) for target in protected_targets), key=len, reverse=True)
        )
        removal = re.compile(
            rf"^rm(?:\.exe)?\s+"
            rf"(?=[^\n]*(?:-[A-Za-z]*r[A-Za-z]*|--recursive)(?:\s|$))"
            rf"(?=[^\n]*(?:-[A-Za-z]*f[A-Za-z]*|--force)(?:\s|$))"
            rf"[^\n]*\s(?:--\s+)?(?:{alternatives})(?:\s|$)",
            re.IGNORECASE,
        )
        self._patterns: tuple[tuple[re.Pattern[str], str], ...] = (
            (_SHELL_PATTERN, "Direct use of a general shell is blocked by the safety boundary."),
            (_FILESYSTEM_PATTERN, "Disk formatting or partition tools are blocked by the safety boundary."),
            (_BLOCK_DEVICE_PATTERN, "Direct writes to block devices are blocked by the safety boundary."),
            (_SYSTEM_PATTERN, "System shutdown or restart commands are blocked by the safety boundary."),
            (removal, "Recursive forced removal of a protected root is blocked by the safety boundary."),
        )

    def reason(self, target: str) -> str | None:
        for pattern, reason in self._patterns:
            if pattern.search(target):
                return reason
        return None
