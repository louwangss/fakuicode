from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from pathlib import Path
import sys
from threading import Event, Thread

from fakuicode.mcp.models import (
    McpCallResult,
    McpInitializeInfo,
    McpRemoteTool,
    McpServerStatus,
    McpToolPage,
    ResolvedStdioServerConfig,
    SecretText,
)
from fakuicode.mcp.runtime import McpClientManager


def _config(name: str) -> ResolvedStdioServerConfig:
    return ResolvedStdioServerConfig(name, "python", (), {})


class _Session:
    def __init__(self, name: str, *, fail: bool = False, delay: float = 0) -> None:
        self.name = name
        self.fail = fail
        self.delay = delay
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def initialize(self) -> McpInitializeInfo:
        if self.fail:
            raise RuntimeError("sentinel-secret")
        return McpInitializeInfo("2025-06-18", True)

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        return McpToolPage((McpRemoteTool(f"{self.name}_tool", "tool", {"type": "object"}),))

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        return McpCallResult(public_summary="ok")


class _Factory:
    def __init__(self, *, failing: set[str] | None = None, delays: dict[str, float] | None = None) -> None:
        self.failing = failing or set()
        self.delays = delays or {}
        self.connects: list[str] = []
        self.closes: list[str] = []
        self.callbacks: dict[str, object] = {}
        self.sessions: dict[str, _Session] = {}

    @asynccontextmanager
    async def connect(self, config: ResolvedStdioServerConfig, *, on_list_changed: object):
        self.connects.append(config.name)
        self.callbacks[config.name] = on_list_changed
        session = _Session(config.name, fail=config.name in self.failing, delay=self.delays.get(config.name, 0))
        self.sessions[config.name] = session
        try:
            yield session
        finally:
            self.closes.append(config.name)


def test_parallel_start_isolates_failure_and_reuses_connection() -> None:
    factory = _Factory(failing={"bad"})
    manager = McpClientManager(factory, startup_timeout=0.5)
    snapshot = manager.start((_config("good"), _config("bad")))
    states = {state.name: state for state in snapshot.states}
    assert states["good"].status is McpServerStatus.CONNECTED
    assert states["bad"].status is McpServerStatus.FAILED
    assert "sentinel-secret" not in repr(snapshot)
    assert list(manager.discovered_tools()) == ["good"]
    manager.start((_config("good"),))
    assert factory.connects.count("good") == 1
    manager.close()
    assert sorted(factory.closes) == ["bad", "good"]


def test_call_timeout_is_local_and_close_is_idempotent() -> None:
    factory = _Factory(delays={"slow": 0.2})
    manager = McpClientManager(factory, call_timeout=0.02, close_timeout=0.5)
    manager.start((_config("slow"),))
    result = manager.call_tool("slow", "work", {"secret": "hidden"})
    assert result.is_error
    assert result.public_summary == "MCP 工具调用超时。"
    started = time.monotonic()
    manager.close()
    manager.close()
    assert time.monotonic() - started < 0.5


def test_list_changed_marks_restart_required_without_replacing_tools() -> None:
    factory = _Factory()
    manager = McpClientManager(factory)
    manager.start((_config("good"),))
    original = manager.discovered_tools()["good"]
    callback = factory.callbacks["good"]
    callback()  # type: ignore[operator]
    assert manager.snapshot().states[0].status is McpServerStatus.RESTART_REQUIRED
    assert manager.discovered_tools()["good"] is original
    manager.close()


def test_real_stdio_lifecycle_enters_and_exits_transport_in_owner_task() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "fake_mcp_stdio_server.py"
    config = ResolvedStdioServerConfig(
        "real",
        sys.executable,
        (str(fixture),),
        {"FAKUICODE_MCP_TEST": SecretText("runtime-ok")},
    )
    manager = McpClientManager(close_timeout=1)
    snapshot = manager.start((config,))
    assert snapshot.states[0].status is McpServerStatus.CONNECTED
    result = manager.call_tool("real", "read_env", {})
    assert result.content[0].text == "runtime-ok"
    manager.close()
    assert manager.snapshot().states[0].status is McpServerStatus.CLOSED


def test_cancel_event_cancels_in_flight_call() -> None:
    factory = _Factory(delays={"slow": 1})
    manager = McpClientManager(factory, call_timeout=2)
    manager.start((_config("slow"),))
    cancelled = Event()
    results: list[McpCallResult] = []
    worker = Thread(
        target=lambda: results.append(
            manager.call_tool("slow", "work", {}, cancel_event=cancelled)
        ),
        daemon=True,
    )
    worker.start()
    time.sleep(0.05)
    cancelled.set()
    worker.join(timeout=1)
    assert results[0].failure_code is not None
    assert results[0].failure_code.value == "call_cancelled"
    manager.close()
