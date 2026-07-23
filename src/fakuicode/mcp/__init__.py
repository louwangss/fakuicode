"""MCP client support for external tools."""

from fakuicode.mcp.models import (
    McpConfigSnapshot,
    McpServerState,
    McpServerStatus,
    McpStartupSnapshot,
)
from fakuicode.mcp.adapter import McpToolAdapter, build_adapters
from fakuicode.mcp.config import McpConfigRepository, McpPaths
from fakuicode.mcp.runtime import McpClientManager
from fakuicode.mcp.trust import McpTrustRepository

__all__ = [
    "McpConfigSnapshot",
    "McpConfigRepository",
    "McpPaths",
    "McpClientManager",
    "McpToolAdapter",
    "McpTrustRepository",
    "McpServerState",
    "McpServerStatus",
    "McpStartupSnapshot",
    "build_adapters",
]
