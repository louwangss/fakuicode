from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mcp.types as types
import httpx
from fakuicode.mcp.sdk import McpSdkSession
from fakuicode.mcp.models import ResolvedHttpServerConfig, ResolvedStdioServerConfig, SecretText
from fakuicode.mcp.sdk import McpSdkConnectionFactory


class _Session:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def initialize(self) -> object:
        return SimpleNamespace(
            protocolVersion="2025-06-18",
            capabilities=SimpleNamespace(tools=object()),
        )

    async def send_request(self, request: object, result_type: type[object]) -> object:
        self.requests.append(request)
        if isinstance(request, types.ListToolsRequest):
            return result_type.model_validate(
                {
                    "tools": [
                        {"name": "good", "description": "Works", "inputSchema": {"type": "object"}},
                        {"name": 7, "description": [], "inputSchema": "bad"},
                    ],
                    "nextCursor": "next",
                }
            )
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text="hello"),
                types.ImageContent(type="image", data="ignored", mimeType="image/png"),
            ],
            structuredContent={"ok": True},
            isError=False,
        )


def test_initialize_list_page_and_call_use_standard_requests() -> None:
    async def scenario() -> None:
        raw = _Session()
        session = McpSdkSession(raw)  # type: ignore[arg-type]
        initialized = await session.initialize()
        first = await session.list_tools()
        second = await session.list_tools("next")
        result = await session.call_tool("good", {"value": 1})
        assert initialized.tools_supported
        assert first.next_cursor == "next"
        assert first.tools[1].name == 7
        assert second.tools[0].name == "good"
        assert isinstance(raw.requests[0], types.ListToolsRequest)
        assert raw.requests[1].params.cursor == "next"  # type: ignore[union-attr]
        assert isinstance(raw.requests[2], types.CallToolRequest)
        assert raw.requests[2].params.name == "good"  # type: ignore[union-attr]
        assert result.content[0].text == "hello"
        assert result.content[1].kind == "image"
        assert result.structured_content == {"ok": True}

    asyncio.run(scenario())


def test_stdio_end_to_end_initializes_paginates_calls_and_passes_explicit_env(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "fake_mcp_stdio_server.py"
        config = ResolvedStdioServerConfig(
            "local",
            sys.executable,
            (str(fixture),),
            {"FAKUICODE_MCP_TEST": SecretText("stdio-ok")},
            working_directory=tmp_path,
        )
        factory = McpSdkConnectionFactory()
        async with factory.connect(config) as session:
            initialized = await session.initialize()
            first = await session.list_tools()
            second = await session.list_tools(first.next_cursor)
            result = await session.call_tool("read_env", {})
            cwd = await session.call_tool("read_cwd", {})
        assert initialized.tools_supported
        assert [tool.name for tool in first.tools + second.tools] == [
            "echo",
            "read_env",
            "read_cwd",
        ]
        assert result.content[0].text == "stdio-ok"
        assert Path(cwd.content[0].text or "").resolve() == tmp_path.resolve()

    asyncio.run(scenario())


def test_streamable_http_end_to_end_uses_headers_and_standard_json_rpc() -> None:
    seen_headers: list[str | None] = []
    seen_methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200)
        payload = json.loads(request.content)
        seen_headers.append(request.headers.get("authorization"))
        seen_methods.append(payload.get("method", ""))
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-http", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]}
        else:
            result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            headers={"content-type": "application/json", "mcp-session-id": "test-session"},
        )

    async def scenario() -> None:
        transport = httpx.MockTransport(handler)
        factory = McpSdkConnectionFactory(
            lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs)
        )
        config = ResolvedHttpServerConfig(
            "web", "https://example.test/mcp", {"Authorization": SecretText("Bearer test")}
        )
        async with factory.connect(config) as session:
            await session.initialize()
            page = await session.list_tools()
            result = await session.call_tool("ping", {})
        assert page.tools[0].name == "ping"
        assert result.content[0].text == "pong"

    asyncio.run(scenario())
    assert all(value == "Bearer test" for value in seen_headers)
    assert seen_methods[:4] == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
