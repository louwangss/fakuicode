"""Thin public-API adapter around the official MCP Python SDK v1."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
import mcp.types as types
from pydantic import BaseModel, ConfigDict, Field

from fakuicode.mcp.models import (
    McpCallResult,
    McpContentBlock,
    McpInitializeInfo,
    McpRemoteTool,
    McpToolPage,
    ResolvedHttpServerConfig,
    ResolvedServerConfig,
    ResolvedStdioServerConfig,
)


ListChangedCallback = Callable[[], Awaitable[None] | None]


class _LooseTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Any = None
    description: Any = None
    input_schema: Any = Field(default=None, alias="inputSchema")


class _LooseListToolsResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    tools: list[_LooseTool] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class McpSdkSession:
    """One initialized SDK session; lifecycle is owned by its async context."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def initialize(self) -> McpInitializeInfo:
        result = await self._session.initialize()
        return McpInitializeInfo(
            protocol_version=str(result.protocolVersion),
            tools_supported=result.capabilities.tools is not None,
        )

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
        result = await self._session.send_request(
            types.ListToolsRequest(params=params),
            _LooseListToolsResult,
        )
        return McpToolPage(
            tools=tuple(
                McpRemoteTool(tool.name, tool.description, tool.input_schema)
                for tool in result.tools
            ),
            next_cursor=result.next_cursor,
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        result = await self._session.send_request(
            types.CallToolRequest(
                params=types.CallToolRequestParams(name=name, arguments=arguments)
            ),
            types.CallToolResult,
        )
        blocks: list[McpContentBlock] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                blocks.append(McpContentBlock("text", block.text))
            else:
                blocks.append(McpContentBlock(getattr(block, "type", "unknown")))
        return McpCallResult(
            content=tuple(blocks),
            structured_content=result.structuredContent,
            is_error=result.isError,
        )


class McpSdkConnectionFactory:
    """Create stdio or Streamable HTTP connections without private SDK APIs."""

    def __init__(self, http_client_factory: Callable[..., httpx.AsyncClient] | None = None) -> None:
        self._http_client_factory = http_client_factory or httpx.AsyncClient

    @asynccontextmanager
    async def connect(
        self,
        config: ResolvedServerConfig,
        *,
        on_list_changed: ListChangedCallback | None = None,
    ) -> AsyncIterator[McpSdkSession]:
        async with AsyncExitStack() as stack:
            if isinstance(config, ResolvedStdioServerConfig):
                parameters = StdioServerParameters(
                    command=config.command,
                    args=list(config.args),
                    env={key: secret.value for key, secret in config.environment.items()},
                    cwd=config.working_directory,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(parameters))
            else:
                assert isinstance(config, ResolvedHttpServerConfig)
                client = await stack.enter_async_context(
                    self._http_client_factory(
                        headers={key: secret.value for key, secret in config.headers.items()},
                        follow_redirects=False,
                    )
                )
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(config.url, http_client=client)
                )

            async def handle_message(message: object) -> None:
                if on_list_changed is None or not isinstance(message, types.ServerNotification):
                    return
                if isinstance(message.root, types.ToolListChangedNotification):
                    outcome = on_list_changed()
                    if outcome is not None:
                        await outcome

            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=None,
                    message_handler=handle_message,
                )
            )
            yield McpSdkSession(session)
