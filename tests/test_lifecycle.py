from __future__ import annotations

from threading import Event
from time import monotonic


def test_daemon_executor_bounds_wait_for_unresponsive_work() -> None:
    from fakuicode.lifecycle import DaemonFutureExecutor

    started = Event()
    release = Event()

    def block() -> str:
        started.set()
        release.wait()
        return "done"

    executor = DaemonFutureExecutor(max_workers=1, thread_name_prefix="test-daemon")
    future = executor.submit(block)
    assert started.wait(timeout=1)

    before = monotonic()
    try:
        executor.shutdown(wait=True, timeout=0.02)
        assert monotonic() - before < 0.5
        assert all(thread.daemon for thread in executor.worker_threads)
    finally:
        release.set()
    assert future.result(timeout=1) == "done"


def test_daemon_executor_cancels_queued_futures() -> None:
    from fakuicode.lifecycle import DaemonFutureExecutor

    started = Event()
    release = Event()

    def block() -> None:
        started.set()
        release.wait()

    executor = DaemonFutureExecutor(max_workers=1)
    running = executor.submit(block)
    assert started.wait(timeout=1)
    queued = executor.submit(lambda: "not run")

    try:
        executor.shutdown(wait=False, cancel_futures=True)
        assert queued.cancelled()
        assert not running.cancelled()
    finally:
        release.set()
    assert running.result(timeout=1) is None
