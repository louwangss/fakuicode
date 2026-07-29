"""Host-owned scheduling for provider-requested read-only tools."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from threading import Lock
from typing import ParamSpec, TypeVar

from fakuicode.lifecycle import DaemonFutureExecutor


_P = ParamSpec("_P")
_T = TypeVar("_T")


class ReadOnlyToolScheduler:
    """Share one bounded executor across all read-only work in a host."""

    def __init__(self, *, max_workers: int | None = None) -> None:
        self._executor = DaemonFutureExecutor(
            max_workers=max_workers,
            thread_name_prefix="fakuicode-read",
        )
        self._lock = Lock()
        self._closed = False

    def submit(
        self,
        function: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Read-only tool scheduler is closed.")
            return self._executor.submit(function, *args, **kwargs)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
