"""Shared lifecycle bounds and daemon-backed background execution."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from math import isfinite
import os
from queue import Queue
from threading import Lock, Thread
from time import monotonic
from typing import Any, ParamSpec, TypeVar

# Public reference CLIs do not currently document a shutdown grace period.
# Keep this internal heuristic explicit and measurable: one second is long
# enough for event-driven cancellation without making UI shutdown unbounded.
DEFAULT_COOPERATIVE_SHUTDOWN_GRACE_SECONDS = 1.0

_P = ParamSpec("_P")
_T = TypeVar("_T")
_WorkItem = tuple[Future[Any], Callable[..., Any], tuple[Any, ...], dict[str, Any]]


class DaemonFutureExecutor:
    """Fixed daemon worker pool with standard Future result semantics."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        thread_name_prefix: str = "fakuicode-worker",
    ) -> None:
        # Preserve Python 3.11 ThreadPoolExecutor's documented default while
        # changing only daemon/lifecycle behavior.
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._queue: Queue[_WorkItem | None] = Queue()
        self._lock = Lock()
        self._closed = False
        self._futures: set[Future[Any]] = set()
        self._threads: list[Thread] = []
        self._idle_workers = 0

    @property
    def worker_threads(self) -> tuple[Thread, ...]:
        with self._lock:
            return tuple(self._threads)

    def submit(
        self,
        function: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        future: Future[_T] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("executor is closed")
            self._futures.add(future)
            self._queue.put((future, function, args, kwargs))
            if self._idle_workers == 0 and len(self._threads) < self._max_workers:
                thread = Thread(
                    target=self._work,
                    name=f"{self._thread_name_prefix}-{len(self._threads)}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        future.add_done_callback(self._discard_future)
        return future

    def shutdown(
        self,
        *,
        wait: bool,
        cancel_futures: bool = False,
        timeout: float = DEFAULT_COOPERATIVE_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._lock:
            first_close = not self._closed
            self._closed = True
            futures = tuple(self._futures) if cancel_futures else ()
            threads = tuple(self._threads)
        for future in futures:
            future.cancel()
        if first_close:
            for _ in threads:
                self._queue.put(None)
        if not wait:
            return
        deadline = monotonic() + timeout
        for thread in threads:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return
            thread.join(remaining)

    def _discard_future(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def _work(self) -> None:
        while True:
            with self._lock:
                self._idle_workers += 1
            item = self._queue.get()
            with self._lock:
                self._idle_workers -= 1
            if item is None:
                return
            future, function, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)
