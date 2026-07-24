"""Keyboard-first permission confirmation screens."""

from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from fakuicode.permissions.config import PermissionConfigSnapshot
from fakuicode.permissions.models import ApprovalChoice, PermissionMode, PermissionRequest, PermissionScope


_APPROVAL_OPTIONS: tuple[tuple[ApprovalChoice, str], ...] = (
    (ApprovalChoice.ONCE, "1. 允许本次"),
    (ApprovalChoice.SESSION, "2. 允许本会话，不再询问此目标"),
    (ApprovalChoice.PERMANENT, "3. 永久允许此目标"),
    (ApprovalChoice.DENY, "4. 仅拒绝此次调用"),
)


class PermissionPrompt(Vertical):
    """Compact in-conversation approval selector for one exact tool target."""

    class Resolved(Message):
        def __init__(
            self,
            prompt: PermissionPrompt,
            choice: ApprovalChoice,
            *,
            cancel_turn: bool = False,
        ) -> None:
            super().__init__()
            self.prompt = prompt
            self.choice = choice
            self.cancel_turn = cancel_turn

    BINDINGS = [
        ("escape", "cancel_turn", "停止任务"),
        ("1", "once", "仅本次"),
        ("2", "session", "本会话"),
        ("3", "permanent", "永久"),
        ("4", "deny", "拒绝"),
    ]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__(classes="inline-choice-prompt")
        self.request = request

    def compose(self) -> ComposeResult:
        source = f" · 来自 SubAgent {self.request.source}" if self.request.source else ""
        yield Static(
            f"{self.request.tool_name} 需要权限{source}",
            classes="inline-choice-title",
            markup=False,
        )
        target = (
            "整个 MCP 工具（授权适用于所有参数值）"
            if self.request.scope is PermissionScope.TOOL
            else self.request.target
        )
        yield Static(target, classes="inline-choice-target", markup=False)
        yield Static("是否允许执行？", classes="inline-choice-question", markup=False)
        yield OptionList(id="permission-options", classes="inline-choice-options", markup=False)
        yield Static(
            f"↑/↓ 选择 · Enter 确认 · Esc 停止任务  ·  精确规则：{self.request.exact_rule}",
            classes="inline-choice-help",
            markup=False,
        )

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.set_options(Option(label) for _, label in _APPROVAL_OPTIONS)
        options.highlighted = 0
        options.focus()

    @on(OptionList.OptionSelected, "#permission-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        if 0 <= message.option_index < len(_APPROVAL_OPTIONS):
            self.post_message(self.Resolved(self, _APPROVAL_OPTIONS[message.option_index][0]))

    def action_deny(self) -> None:
        self.post_message(self.Resolved(self, ApprovalChoice.DENY))

    def action_cancel_turn(self) -> None:
        self.post_message(self.Resolved(self, ApprovalChoice.DENY, cancel_turn=True))

    def action_once(self) -> None:
        self.post_message(self.Resolved(self, ApprovalChoice.ONCE))

    def action_session(self) -> None:
        self.post_message(self.Resolved(self, ApprovalChoice.SESSION))

    def action_permanent(self) -> None:
        self.post_message(self.Resolved(self, ApprovalChoice.PERMANENT))


class PlanExecutionPrompt(Vertical):
    """Offer direct execution after a read-only plan has completed."""

    class Resolved(Message):
        def __init__(self, prompt: PlanExecutionPrompt, *, execute: bool) -> None:
            super().__init__()
            self.prompt = prompt
            self.execute = execute

    BINDINGS = [
        ("escape", "later", "暂不执行"),
        ("1", "execute", "执行计划"),
        ("2", "later", "暂不执行"),
    ]

    def __init__(self) -> None:
        super().__init__(classes="inline-choice-prompt plan-execution-prompt")

    def compose(self) -> ComposeResult:
        yield Static("计划已就绪", classes="inline-choice-title")
        yield Static("现在退出 Plan 模式并执行吗？", classes="inline-choice-question")
        yield OptionList(id="plan-execution-options", classes="inline-choice-options", markup=False)
        yield Static(
            "执行时会重新经过当前权限规则 · Esc 暂不执行",
            classes="inline-choice-help",
        )

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.set_options((Option("1. 执行计划"), Option("2. 暂不执行，保留计划")))
        options.highlighted = 0
        options.focus()

    @on(OptionList.OptionSelected, "#plan-execution-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        self.post_message(self.Resolved(self, execute=message.option_index == 0))

    def action_execute(self) -> None:
        self.post_message(self.Resolved(self, execute=True))

    def action_later(self) -> None:
        self.post_message(self.Resolved(self, execute=False))


@dataclass(frozen=True)
class PermissionSettingsAction:
    """One local settings change selected by the user."""

    mode: PermissionMode | None = None
    project_trusted: bool | None = None


class PermissionSettingsScreen(ModalScreen[PermissionSettingsAction | None]):
    """Manage the current session mode and explicit workspace trust."""

    BINDINGS = [("escape", "cancel", "取消")]

    def __init__(self, mode: PermissionMode, snapshot: PermissionConfigSnapshot) -> None:
        super().__init__()
        self.mode = mode
        self.snapshot = snapshot
        self._actions: tuple[PermissionSettingsAction | None, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-settings-dialog"):
            yield Static("权限设置 · ↑↓ 选择 · Enter 确认 · Esc 取消", id="permission-settings-title")
            yield Static(f"当前会话模式：{self.mode.value}", markup=False)
            trust = "已信任" if self.snapshot.project_trusted else "未信任"
            yield Static(f"项目共享规则：{trust}", markup=False)
            if self.snapshot.locked:
                yield Static("配置存在错误，权限已锁定为 strict。修复配置并重启后才能切换。", markup=False)
            for diagnostic in self.snapshot.diagnostics:
                yield Static(diagnostic, classes="permission-diagnostic", markup=False)
            for warning in self.snapshot.warnings:
                yield Static(warning, classes="permission-warning", markup=False)
            yield OptionList(id="permission-settings-options", markup=False)

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        if self.snapshot.locked:
            labels = ("关闭",)
            self._actions = (None,)
            highlighted = 0
        else:
            labels = (
                "strict · 未被规则放行的操作一律拒绝",
                "default · 安全读取自动放行，其余操作询问",
                "trusted · 常规项目操作自动放行，危险命令仍硬拦截",
                ("撤销项目共享规则信任" if self.snapshot.project_trusted else "信任项目共享规则"),
            )
            self._actions = (
                PermissionSettingsAction(mode=PermissionMode.STRICT),
                PermissionSettingsAction(mode=PermissionMode.DEFAULT),
                PermissionSettingsAction(mode=PermissionMode.TRUSTED),
                PermissionSettingsAction(project_trusted=not self.snapshot.project_trusted),
            )
            highlighted = (
                PermissionMode.STRICT,
                PermissionMode.DEFAULT,
                PermissionMode.TRUSTED,
            ).index(self.mode)
        options.set_options(Option(label) for label in labels)
        options.highlighted = highlighted
        options.focus()

    @on(OptionList.OptionSelected, "#permission-settings-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        if 0 <= message.option_index < len(self._actions):
            self.dismiss(self._actions[message.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)
