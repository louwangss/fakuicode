"""Application-scoped MCP connections hosted on a private asyncio loop."""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
import threading
import time
from types import MappingProxyType
from typing import Any

from fakuicode.mcp.models import (
    McpCallResult,
    McpFailureCode,
    McpRemoteTool,
    McpServerState,
    McpServerStatus,
    McpStartupSnapshot,
    ResolvedServerConfig,
)
from fakuicode.mcp.sdk import McpSdkConnectionFactory, McpSdkSession


class McpClientManager:
    """Own cached connections and isolate every server/call failure."""

    def __init__(
        self,
        factory: McpSdkConnectionFactory | Any | None = None,
        *,
        startup_timeout: float = 10.0,
        call_timeout: float = 60.0,
        close_timeout: float = 5.0,
    ) -> None:
        self._factory = factory or McpSdkConnectionFactory()
        self._startup_timeout = startup_timeout
        self._call_timeout = call_timeout
        self._close_timeout = close_timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="fakuicode-mcp", daemon=True)
        self._thread.start()
        self._lock = threading.RLock()
        self._states: dict[str, McpServerState] = {}
        self._sessions: dict[str, McpSdkSession] = {}
        self._tools: dict[str, tuple[McpRemoteTool, ...]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._shutdown_event: asyncio.Event | None = None
        self._started = False
        self._closed = False

    def start(self, configs: tuple[ResolvedServerConfig, ...]) -> McpStartupSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP manager is closed")
            if self._started:
                return self.snapshot()
            self._started = True
            for config in configs:
                self._states[config.name] = McpServerState(
                    config.name, config.transport, McpServerStatus.CONNECTING
                )
        future = asyncio.run_coroutine_threadsafe(self._start_all(configs), self._loop)
        try:
            future.result(timeout=max(self._startup_timeout + 1.0, 2.0))
        except FutureTimeoutError:
            future.cancel()
            with self._lock:
                for config in configs:
                    if self._states[config.name].status is McpServerStatus.CONNECTING:
                        self._set_failed(
                            config, McpFailureCode.STARTUP_TIMEOUT, "启动发现超时。"
                        )
        return self.snapshot()

    def snapshot(self) -> McpStartupSnapshot:
        with self._lock:
            return McpStartupSnapshot(states=tuple(self._states[name] for name in sorted(self._states)))

    def discovered_tools(self) -> MappingProxyType[str, tuple[McpRemoteTool, ...]]:
        with self._lock:
            return MappingProxyType(dict(self._tools))

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
        *,
        cancel_event: threading.Event | None = None,
    ) -> McpCallResult:
        with self._lock:
            session = self._sessions.get(server_name)
            state = self._states.get(server_name)
        if session is None or state is None or state.status not in {
            McpServerStatus.CONNECTED,
            McpServerStatus.RESTART_REQUIRED,
        }:
            return _call_failure(McpFailureCode.CONNECTION_CLOSED, "MCP Server 当前不可用。")
        future = asyncio.run_coroutine_threadsafe(
            self._call_with_timeout(session, tool_name, dict(arguments)), self._loop
        )
        deadline = time.monotonic() + self._call_timeout + 1.0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                future.cancel()
                return _call_failure(McpFailureCode.CALL_CANCELLED, "MCP 工具调用已取消。")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                return _call_failure(McpFailureCode.CALL_TIMEOUT, "MCP 工具调用超时。")
            try:
                return future.result(timeout=min(0.05, remaining))
            except FutureTimeoutError:
                continue
            except BaseException:
                return _call_failure(McpFailureCode.PROTOCOL_ERROR, "MCP 工具调用失败。")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        future = asyncio.run_coroutine_threadsafe(self._close_all(), self._loop)
        try:
            future.result(timeout=self._close_timeout)
        except BaseException:
            future.cancel()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=self._close_timeout)

    async def _start_all(self, configs: tuple[ResolvedServerConfig, ...]) -> None:
        self._shutdown_event = asyncio.Event()
        readiness: list[asyncio.Event] = []
        for config in configs:
            ready = asyncio.Event()
            readiness.append(ready)
            self._tasks[config.name] = asyncio.create_task(
                self._server_lifecycle(config, ready), name=f"mcp-{config.name}"
            )
        await asyncio.gather(*(ready.wait() for ready in readiness))

    async def _server_lifecycle(
        self, config: ResolvedServerConfig, ready: asyncio.Event
    ) -> None:
        try:
            async with self._factory.connect(
                config, on_list_changed=lambda: self._mark_restart_required(config.name)
            ) as session:
                tools = await asyncio.wait_for(
                    self._discover(session), timeout=self._startup_timeout
                )
                with self._lock:
                    self._sessions[config.name] = session
                    self._tools[config.name] = tools
                    self._states[config.name] = McpServerState(
                        config.name,
                        config.transport,
                        McpServerStatus.CONNECTED,
                        tool_count=len(tools),
                    )
                ready.set()
                assert self._shutdown_event is not None
                await self._shutdown_event.wait()
        except TimeoutError:
            self._set_failed(config, McpFailureCode.STARTUP_TIMEOUT, "启动发现超时。")
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._set_failed(config, McpFailureCode.CONNECT_FAILED, "连接或工具发现失败。")
        finally:
            ready.set()
            with self._lock:
                self._sessions.pop(config.name, None)

    async def _discover(self, session: McpSdkSession) -> tuple[McpRemoteTool, ...]:
        initialized = await session.initialize()
        if not initialized.tools_supported:
            raise RuntimeError("server does not advertise tools")
        tools: list[McpRemoteTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            page = await session.list_tools(cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tuple(tools)
            if cursor in seen_cursors:
                raise RuntimeError("repeated tools cursor")
            seen_cursors.add(cursor)

    async def _call_with_timeout(
        self, session: McpSdkSession, tool_name: str, arguments: dict[str, object]
    ) -> McpCallResult:
        try:
            return await asyncio.wait_for(
                session.call_tool(tool_name, arguments), timeout=self._call_timeout
            )
        except TimeoutError:
            return _call_failure(McpFailureCode.CALL_TIMEOUT, "MCP 工具调用超时。")
        except asyncio.CancelledError:
            raise
        except BaseException:
            return _call_failure(McpFailureCode.PROTOCOL_ERROR, "MCP 工具调用失败。")

    def _mark_restart_required(self, name: str) -> None:
        with self._lock:
            state = self._states.get(name)
            if state is not None and state.status is McpServerStatus.CONNECTED:
                self._states[name] = replace(state, status=McpServerStatus.RESTART_REQUIRED)

    def _set_failed(
        self, config: ResolvedServerConfig, code: McpFailureCode, summary: str
    ) -> None:
        with self._lock:
            self._states[config.name] = McpServerState(
                config.name,
                config.transport,
                McpServerStatus.FAILED,
                failure_code=code,
                public_summary=summary,
            )

    async def _close_all(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with self._lock:
            self._tasks.clear()
            for name, state in tuple(self._states.items()):
                if state.status in {McpServerStatus.CONNECTED, McpServerStatus.RESTART_REQUIRED}:
                    self._states[name] = replace(state, status=McpServerStatus.CLOSED)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    def __enter__(self) -> McpClientManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _call_failure(code: McpFailureCode, summary: str) -> McpCallResult:
    return McpCallResult(failure_code=code, public_summary=summary, is_error=True)
