"""Keyboard-first preview for installing one public Skill package."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from fakuicode.skills.install import (
    SkillInstallDecision,
    SkillInstallPreset,
    SkillInstallPreview,
)


_PRESETS = (
    SkillInstallPreset.INSTRUCTION,
    SkillInstallPreset.READ_ONLY,
    SkillInstallPreset.CODING,
)


class SkillInstallScreen(ModalScreen[SkillInstallDecision]):
    """Show the immutable source facts before allowing files to be installed."""

    BINDINGS = [("escape", "cancel", "取消")]

    def __init__(self, preview: SkillInstallPreview) -> None:
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-install-dialog"):
            yield Static(f"安装 Skill：{self.preview.name}", id="skill-install-title", markup=False)
            with VerticalScroll(id="skill-install-details-scroll"):
                yield Static(self._details(), id="skill-install-details", markup=False)
            yield OptionList(
                Option("取消（默认）", id="cancel"),
                *(
                    Option(self._preset_label(preset), id=preset.value)
                    for preset in _PRESETS
                ),
                id="skill-install-options",
                markup=False,
            )
            yield Static("↑↓ 选择 · Enter 确认 · Esc 取消", id="skill-install-help", markup=False)

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 0
        options.focus()

    @on(OptionList.OptionSelected, "#skill-install-options")
    def _selected(self, message: OptionList.OptionSelected) -> None:
        if message.option_index == 0:
            self.action_cancel()
            return
        self.dismiss(SkillInstallDecision(True, _PRESETS[message.option_index - 1]))

    def action_cancel(self) -> None:
        self.dismiss(SkillInstallDecision(False, self.preview.preset))

    def _details(self) -> str:
        license_name = self.preview.license or "未声明"
        scripts = "有" if self.preview.contains_scripts else "无"
        dedicated = "、".join(self.preview.dedicated_tools) or "无"
        relation: list[str] = []
        if self.preview.replacing:
            relation.append("将替换同层现有目录")
        if self.preview.shadows:
            relation.append("将遮蔽 " + "、".join(self.preview.shadows) + " 层同名 Skill")
        relation_text = "；".join(relation) or "无覆盖或遮蔽"
        files = "\n".join(f"  - {path}" for path in self.preview.files)
        preset_tools = "、".join(self.preview.visible_tools) or "无"
        upstream_tools = self.preview.upstream_allowed_tools or "未声明"
        return (
            f"请求：{self.preview.requested_url}\n"
            f"来源：{self.preview.source_url}\n"
            f"固定 commit：{self.preview.revision}\n"
            f"Skill 子目录：{self.preview.skill_path}\n"
            f"目标：{self.preview.target_path}\n"
            f"许可证：{license_name}\n"
            f"文件：{self.preview.file_count} 个，共 {self.preview.total_bytes:,} bytes\n"
            f"脚本：{scripts}；专属工具：{dedicated}\n"
            f"上游 allowed-tools：{upstream_tools}（仅建议，不授予权限）\n"
            f"建议预设：{self.preview.preset.value}；可见工具：{preset_tools}\n"
            f"关系：{relation_text}\n"
            f"文件清单：\n{files}"
        )

    def _preset_label(self, preset: SkillInstallPreset) -> str:
        suffix = "（建议）" if preset is self.preview.preset else ""
        tools = {
            SkillInstallPreset.INSTRUCTION: "不暴露工具",
            SkillInstallPreset.READ_ONLY: "只读文件工具",
            SkillInstallPreset.CODING: "读写与命令工具",
        }[preset]
        return f"安装 · {preset.value}{suffix} · {tools}"
