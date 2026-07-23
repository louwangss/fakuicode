"""Keyboard-first trust prompt for executable project Hook configuration."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class HookTrustPrompt(Vertical):
    class Resolved(Message):
        def __init__(self, prompt: "HookTrustPrompt", *, approve: bool) -> None:
            super().__init__()
            self.prompt = prompt
            self.approve = approve

    BINDINGS = [
        ("escape", "reject", "拒绝"),
        ("1", "approve", "允许并记住"),
        ("2", "reject", "拒绝本次"),
    ]

    def __init__(self, path: str, fingerprint: str, rule_count: int) -> None:
        super().__init__(classes="inline-choice-prompt hook-trust-prompt")
        self.path = path
        self.fingerprint = fingerprint
        self.rule_count = rule_count
        self._resolved = False

    def compose(self) -> ComposeResult:
        yield Static("项目生命周期 Hook", classes="inline-choice-title", markup=False)
        yield Static(self.path, classes="inline-choice-target", markup=False)
        yield Static(
            f"规则数：{self.rule_count} · 内容指纹：{self.fingerprint[:12]}…",
            classes="inline-choice-help",
            markup=False,
        )
        yield Static(
            "Hook 可执行本地命令或发出 HTTP 请求。是否信任当前文件内容？",
            classes="inline-choice-question",
            markup=False,
        )
        yield OptionList(
            Option("1. 允许并记住当前内容指纹"),
            Option("2. 拒绝本次"),
            id="hook-trust-options",
            classes="inline-choice-options",
        )
        yield Static("默认拒绝 · Enter 确认 · Esc 拒绝", classes="inline-choice-help", markup=False)

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 1
        options.focus()

    @on(OptionList.OptionSelected, "#hook-trust-options")
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
