"""Compatibility exports for Team state locking."""

from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Callable

from fakuicode.locking import (
    FileLockPolicy,
    FileLockTimeoutError,
    KernelFileLock as _KernelFileLock,
)


class KernelFileLock(_KernelFileLock):
    """Preserve the Team-specific contention message over the shared lock."""

    def __init__(
        self,
        path: Path,
        *,
        policy: FileLockPolicy = FileLockPolicy(),
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        super().__init__(
            path,
            policy=policy,
            monotonic=monotonic,
            sleeper=sleeper,
            jitter=jitter,
            timeout_message="Team 状态正被其他进程使用。",
        )

__all__ = ["FileLockPolicy", "FileLockTimeoutError", "KernelFileLock"]
