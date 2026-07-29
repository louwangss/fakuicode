"""Small raw JSON-RPC MCP server used only by local subprocess tests."""

from __future__ import annotations

import json
import os
import sys


def respond(request_id: object, result: object) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message:
        continue
    if method == "initialize":
        respond(
            message["id"],
            {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-stdio", "version": "1"},
            },
        )
    elif method == "tools/list":
        cursor = (message.get("params") or {}).get("cursor")
        if cursor is None:
            respond(
                message["id"],
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text",
                            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                        }
                    ],
                    "nextCursor": "page-2",
                },
            )
        else:
            respond(
                message["id"],
                {
                    "tools": [
                        {
                            "name": "read_env",
                            "description": "Read explicit test env",
                            "inputSchema": {"type": "object"},
                        },
                        {
                            "name": "read_cwd",
                            "description": "Read process working directory",
                            "inputSchema": {"type": "object"},
                        },
                    ]
                },
            )
    elif method == "tools/call":
        name = message["params"]["name"]
        arguments = message["params"].get("arguments", {})
        if name == "read_env":
            value = os.environ.get("FAKUICODE_MCP_TEST", "")
        elif name == "read_cwd":
            value = os.getcwd()
        else:
            value = arguments.get("text", "")
        respond(message["id"], {"content": [{"type": "text", "text": value}], "isError": False})
