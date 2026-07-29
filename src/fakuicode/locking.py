"""Reusable cross-process kernel file locks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import time
from typing import BinaryIO, Callable


class FileLockTimeoutError(TimeoutError):
    """Raised when a bounded file lock cannot be acquired."""


@dataclass(frozen=True)
class FileLockPolicy:
    timeout_seconds: float = 2.0
    initial_backoff_seconds: float = 0.01
    maximum_backoff_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0:
            raise ValueError("锁等待时限不能为负数。")
        if self.initial_backoff_seconds <= 0:
            raise ValueError("锁退避时间必须大于零。")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("锁最大退避时间不能小于初始值。")


class KernelFileLock:
    """A reusable lock file whose ownership is enforced by the OS kernel."""

    def __init__(
        self,
        path: Path,
        *,
        policy: FileLockPolicy = FileLockPolicy(),
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        timeout_message: str = "文件锁正被其他进程使用。",
    ) -> None:
        self.path = path
        self.policy = policy
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._jitter = jitter
        self._timeout_message = timeout_message
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("文件锁已被当前对象持有。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = self._monotonic()
        backoff = self.policy.initial_backoff_seconds
        while True:
            handle: BinaryIO | None = None
            try:
                handle = self.path.open("a+b")
                _ensure_lock_byte(handle)
                _lock_nonblocking(handle)
            except (OSError, ImportError) as error:
                if handle is not None:
                    handle.close()
                elapsed = self._monotonic() - started
                if elapsed >= self.policy.timeout_seconds:
                    raise FileLockTimeoutError(self._timeout_message) from error
                remaining = self.policy.timeout_seconds - elapsed
                delay = min(backoff, remaining)
                if delay > 0:
                    self._sleeper(self._jitter(delay * 0.5, delay))
                backoff = min(backoff * 2, self.policy.maximum_backoff_seconds)
                continue
            assert handle is not None
            self._handle = handle
            return

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> KernelFileLock:
        self.acquire()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.release()


def _ensure_lock_byte(handle: BinaryIO) -> None:
    handle.seek(0)
    if handle.read(1) != b"\0":
        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)


def _lock_nonblocking(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
