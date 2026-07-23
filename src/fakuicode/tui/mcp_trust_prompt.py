"""Keyboard-first project MCP server trust prompt."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from fakuicode.mcp.models import McpTrustRequest, McpTransportType


class McpTrustPrompt(Vertical):
    class Resolved(Message):
        def __init__(self, prompt: McpTrustPrompt, *, approve: bool) -> None:
            super().__init__()
            self.prompt = prompt
            self.approve = approve

    BINDINGS = [
        ("escape", "reject", "拒绝"),
        ("1", "approve", "允许并记住"),
        ("2", "reject", "拒绝本次"),
    ]

    def __init__(self, request: McpTrustRequest) -> None:
        super().__init__(classes="inline-choice-prompt mcp-trust-prompt")
        self.request = request
        self._resolved = False

    def compose(self) -> ComposeResult:
        request = self.request
        yield Static(f"项目 MCP Server：{request.identity.server_name}", classes="inline-choice-title", markup=False)
        if request.transport is McpTransportType.STDIO:
            yield Static(
                f"stdio · command: {request.command} · args: {request.argument_count}",
                classes="inline-choice-target",
                markup=False,
            )
        else:
            yield Static(f"HTTP · {request.redacted_url}", classes="inline-choice-target", markup=False)
        names = [
            *(f"env:{name}" for name in request.environment_names),
            *(f"header:{name}" for name in request.header_names),
        ]
        if names:
            yield Static("可见名称：" + ", ".join(names), classes="inline-choice-help", markup=False)
        yield Static("此 Server 可向 Agent 注册并执行外部工具。是否信任？", classes="inline-choice-question", markup=False)
        yield OptionList(Option("1. 允许并记住"), Option("2. 拒绝本次"), id="mcp-trust-options", classes="inline-choice-options")
        yield Static("默认拒绝 · Enter 确认 · Esc 拒绝；密钥值不会显示", classes="inline-choice-help", markup=False)

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 1
        options.focus()

    @on(OptionList.OptionSelected, "#mcp-trust-options")
    def _selected(self, message: OptionList.OptionSelected) -> None:
        self._resolve(message.option_index == 0)

    def action_approve(self) -> None:
        self._resolve(True)

    def action_reject(self) -> None:
        self._resolve(False)

    def _resolve(self, approve: bool) -> None:
        if self._resolved:
            return
        self._resolved = True
        self.post_message(self.Resolved(self, approve=approve))
