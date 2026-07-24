"""Bounded, non-interactive Git subprocess execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Sequence


class GitCommandError(RuntimeError):
    def __init__(self, operation: str, *, timed_out: bool = False) -> None:
        super().__init__(operation)
        self.operation = operation
        self.timed_out = timed_out


@dataclass(frozen=True)
class GitResult:
    stdout: str
    returncode: int


class GitRunner:
    def run(
        self,
        cwd: Path,
        args: Sequence[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> GitResult:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GitCommandError(
                _operation(args),
                timed_out=isinstance(error, subprocess.TimeoutExpired),
            ) from error
        if check and completed.returncode != 0:
            raise GitCommandError(_operation(args))
        return GitResult(completed.stdout.strip(), completed.returncode)


def _operation(args: Sequence[str]) -> str:
    return " ".join(args[:2]) if args else "git"
