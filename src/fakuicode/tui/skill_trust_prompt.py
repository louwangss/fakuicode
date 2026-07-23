"""Keyboard-first trust prompt for project Skill packages containing Python code."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from fakuicode.skills.trust import SkillTrustRequest


class SkillTrustPrompt(Vertical):
    class Resolved(Message):
        def __init__(self, prompt: "SkillTrustPrompt", *, approve: bool) -> None:
            super().__init__()
            self.prompt = prompt
            self.approve = approve

    BINDINGS = [
        ("escape", "reject", "拒绝"),
        ("1", "approve", "允许并记住"),
        ("2", "reject", "拒绝本次"),
    ]

    def __init__(self, request: SkillTrustRequest) -> None:
        super().__init__(classes="inline-choice-prompt skill-trust-prompt")
        self.request = request
        self._resolved = False

    def compose(self) -> ComposeResult:
        yield Static(f"项目 Skill：{self.request.name}", classes="inline-choice-title", markup=False)
        yield Static(str(self.request.package_path), classes="inline-choice-target", markup=False)
        if self.request.capabilities:
            yield Static(
                "专属工具：" + "；".join(self.request.capabilities),
                classes="inline-choice-help",
                markup=False,
            )
        yield Static(
            "此能力包包含会以当前 Python 解释器运行的本地脚本，且不具备 OS 沙箱。是否信任当前内容指纹？",
            classes="inline-choice-question",
            markup=False,
        )
        yield OptionList(
            Option("1. 允许并记住当前指纹"),
            Option("2. 拒绝本次"),
            id="skill-trust-options",
            classes="inline-choice-options",
        )
        yield Static("默认拒绝 · Enter 确认 · Esc 拒绝", classes="inline-choice-help", markup=False)

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 1
        options.focus()

    @on(OptionList.OptionSelected, "#skill-trust-options")
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
