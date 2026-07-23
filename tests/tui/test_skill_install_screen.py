from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App
from textual.widgets import OptionList, Static

from fakuicode.skills.install import (
    SkillInstallDecision,
    SkillInstallPreset,
    SkillInstallPreview,
    SkillInstallScope,
)
from fakuicode.tui.skill_install_screen import SkillInstallScreen


def _preview() -> SkillInstallPreview:
    return SkillInstallPreview(
        "frontend-design",
        "Build polished interfaces",
        "Apache-2.0",
        "https://www.skills.sh/anthropics/skills/frontend-design",
        "https://github.com/anthropics/skills",
        "a" * 40,
        "skills/frontend-design",
        Path("project/.fakuicode/skills/frontend-design"),
        SkillInstallScope.PROJECT,
        SkillInstallPreset.CODING,
        ("read_file", "write_file"),
        ("LICENSE.txt", "SKILL.md", "scripts/check.py"),
        512,
        True,
        ("audit",),
        True,
        ("user",),
        "Read Write Bash",
    )


class InstallScreenApp(App[None]):
    pass


def test_preview_defaults_to_cancel_and_returns_selected_preset() -> None:
    async def run() -> None:
        app = InstallScreenApp()
        result: list[SkillInstallDecision] = []
        async with app.run_test() as pilot:
            app.push_screen(SkillInstallScreen(_preview()), result.append)
            await pilot.pause()

            details = str(app.screen.query_one("#skill-install-details", Static).content)
            assert "github.com/anthropics/skills" in details
            assert "a" * 40 in details
            assert "Apache-2.0" in details
            assert "scripts/check.py" in details
            assert "专属工具：audit" in details
            assert "allowed-tools：Read Write Bash（仅建议，不授予权限）" in details
            assert app.screen.query_one(OptionList).highlighted == 0

            await pilot.press("down", "down", "enter")
            await pilot.pause()

        assert result == [SkillInstallDecision(True, SkillInstallPreset.READ_ONLY)]

    asyncio.run(run())


def test_escape_cancels_with_original_preset() -> None:
    async def run() -> None:
        app = InstallScreenApp()
        result: list[SkillInstallDecision] = []
        async with app.run_test() as pilot:
            app.push_screen(SkillInstallScreen(_preview()), result.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert result == [SkillInstallDecision(False, SkillInstallPreset.CODING)]

    asyncio.run(run())
