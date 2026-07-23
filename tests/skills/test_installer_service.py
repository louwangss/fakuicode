from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest
import yaml

from fakuicode.skills import SkillDiscovery
from fakuicode.skills.install import (
    RemoteSkillPackage,
    SkillInstallDecision,
    SkillInstallError,
    SkillInstallPreset,
    SkillInstallRequest,
    SkillInstallScope,
    SkillInstaller,
    parse_install_source,
)


def _remote_package(*, body: str = "Build it.\n") -> RemoteSkillPackage:
    source = parse_install_source(
        "https://www.skills.sh/anthropics/skills/frontend-design"
    )
    return RemoteSkillPackage(
        source=source,
        name="frontend-design",
        revision="a" * 40,
        skill_path="skills/frontend-design",
        files={
            "SKILL.md": (
                "---\n"
                "name: frontend-design\n"
                "description: Build distinctive frontend interfaces\n"
                "license: Complete terms in LICENSE.txt\n"
                "---\n"
                f"{body}"
            ).encode(),
            "LICENSE.txt": b"upstream license\n",
        },
    )


class FakeFetcher:
    def __init__(self, package: RemoteSkillPackage) -> None:
        self.package = package

    def fetch(self, source, *, cancel_event: Event | None = None) -> RemoteSkillPackage:
        return self.package


def _installer(
    tmp_path: Path,
    *,
    package: RemoteSkillPackage | None = None,
    refresh=None,
) -> tuple[SkillInstaller, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user_root = tmp_path / "home" / ".fakuicode" / "skills"
    installer = SkillInstaller(
        workspace,
        user_root,
        fetcher=FakeFetcher(package or _remote_package()),
        refresh=refresh,
    )
    return installer, workspace, user_root


def test_install_keeps_upstream_files_and_writes_versioned_effective_receipt(tmp_path: Path) -> None:
    previews = []
    installer, workspace, _ = _installer(tmp_path)

    result = installer.install(
        SkillInstallRequest(
            "https://www.skills.sh/anthropics/skills/frontend-design",
            preset=SkillInstallPreset.CODING,
        ),
        confirm=lambda preview: previews.append(preview) or SkillInstallDecision(True, preview.preset),
    )

    target = workspace / ".fakuicode" / "skills" / "frontend-design"
    assert result.success is True
    assert result.target_path == target
    assert (target / "SKILL.md").read_bytes() == _remote_package().files["SKILL.md"]
    assert (target / "LICENSE.txt").read_bytes() == b"upstream license\n"
    receipt = yaml.safe_load((target / ".fakuicode" / "install.yaml").read_text(encoding="utf-8"))
    assert receipt["schema-version"] == 1
    assert receipt["revision"] == "a" * 40
    assert receipt["source-url"] == "https://github.com/anthropics/skills"
    assert receipt["fakuicode"]["visible-tools"] == [
        "read_file",
        "find_files",
        "search_code",
        "write_file",
        "edit_file",
        "run_command",
    ]
    assert previews[0].license == "Complete terms in LICENSE.txt"
    assert previews[0].file_count == 2
    assert previews[0].contains_scripts is False
    assert previews[0].upstream_allowed_tools is None


def test_cancelled_preview_leaves_no_skill_or_temporary_directory(tmp_path: Path) -> None:
    installer, workspace, _ = _installer(tmp_path)

    result = installer.install(
        SkillInstallRequest("https://www.skills.sh/anthropics/skills/frontend-design"),
        confirm=lambda preview: SkillInstallDecision(False, preview.preset),
    )

    assert result.success is False
    assert not (workspace / ".fakuicode" / "skills" / "frontend-design").exists()
    assert not tuple(workspace.glob(".fakuicode-skill-*"))


def test_preview_honors_instruction_preset_and_reports_lower_layer_shadow(tmp_path: Path) -> None:
    installer, _, user_root = _installer(tmp_path)
    (user_root / "frontend-design").mkdir(parents=True)
    previews = []

    installer.install(
        SkillInstallRequest(
            "https://www.skills.sh/anthropics/skills/frontend-design",
            preset=SkillInstallPreset.INSTRUCTION,
        ),
        confirm=lambda preview: previews.append(preview)
        or SkillInstallDecision(False, preview.preset),
    )

    assert previews[0].preset is SkillInstallPreset.INSTRUCTION
    assert previews[0].visible_tools == ()
    assert previews[0].shadows == ("user",)


def test_same_layer_conflict_requires_explicit_replace(tmp_path: Path) -> None:
    installer, workspace, _ = _installer(tmp_path)
    existing = workspace / ".fakuicode" / "skills" / "frontend-design"
    existing.mkdir(parents=True)
    (existing / "user.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(SkillInstallError, match="--replace"):
        installer.install(
            SkillInstallRequest("https://www.skills.sh/anthropics/skills/frontend-design"),
            confirm=lambda preview: pytest.fail("conflicts must fail before confirmation"),
        )

    assert (existing / "user.txt").read_text(encoding="utf-8") == "keep"


def test_replace_rolls_back_old_directory_when_hot_refresh_rejects_new_skill(tmp_path: Path) -> None:
    refresh_calls = 0

    def refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            raise RuntimeError("refresh failed")

    installer, workspace, _ = _installer(tmp_path, refresh=refresh)
    existing = workspace / ".fakuicode" / "skills" / "frontend-design"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(SkillInstallError, match="rolled back"):
        installer.install(
            SkillInstallRequest(
                "https://www.skills.sh/anthropics/skills/frontend-design",
                replace=True,
            ),
            confirm=lambda preview: SkillInstallDecision(True, preview.preset),
        )

    assert (existing / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (existing / "SKILL.md").exists()
    assert refresh_calls == 2


def test_user_scope_installs_under_private_user_root(tmp_path: Path) -> None:
    installer, _, user_root = _installer(tmp_path)

    result = installer.install(
        SkillInstallRequest(
            "https://www.skills.sh/anthropics/skills/frontend-design",
            scope=SkillInstallScope.USER,
        ),
        confirm=lambda preview: SkillInstallDecision(True, SkillInstallPreset.READ_ONLY),
    )

    assert result.target_path == user_root / "frontend-design"
    snapshot = SkillDiscovery(tmp_path / "none", user_root, tmp_path / "builtin").refresh(
        {"read_file", "find_files", "search_code"}
    )
    assert snapshot.skills["frontend-design"].visible_tools == (
        "read_file",
        "find_files",
        "search_code",
    )


def test_invalid_upstream_skill_is_rejected_before_confirmation(tmp_path: Path) -> None:
    invalid = _remote_package(body="")
    invalid = RemoteSkillPackage(
        invalid.source,
        invalid.name,
        invalid.revision,
        invalid.skill_path,
        {"SKILL.md": b"not a skill\n", "LICENSE.txt": b"license\n"},
    )
    installer, _, _ = _installer(tmp_path, package=invalid)

    with pytest.raises(SkillInstallError, match="invalid"):
        installer.install(
            SkillInstallRequest("https://www.skills.sh/anthropics/skills/frontend-design"),
            confirm=lambda preview: pytest.fail("invalid package must not be shown"),
        )


def test_installer_revalidates_target_after_confirmation(tmp_path: Path) -> None:
    installer, workspace, _ = _installer(tmp_path)
    target = workspace / ".fakuicode" / "skills" / "frontend-design"

    def replace_target(preview):
        target.mkdir(parents=True)
        (target / "race.txt").write_text("race", encoding="utf-8")
        return SkillInstallDecision(True, preview.preset)

    with pytest.raises(SkillInstallError, match="changed"):
        installer.install(
            SkillInstallRequest("https://www.skills.sh/anthropics/skills/frontend-design"),
            confirm=replace_target,
        )

    assert (target / "race.txt").read_text(encoding="utf-8") == "race"
