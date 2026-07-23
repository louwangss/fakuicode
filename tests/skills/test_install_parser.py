from __future__ import annotations

from pathlib import Path

import yaml

from fakuicode.skills import SkillDiscovery
from fakuicode.skills.parser import fingerprint_upstream
from fakuicode.skills.parser import parse_skill_package, SkillParseError
from fakuicode.skills.models import SkillSource
import pytest


def _write_public_skill(root: Path) -> Path:
    package = root / "frontend-design"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\n"
        "name: frontend-design\n"
        "description: Build distinctive production-grade frontend interfaces\n"
        "license: Complete terms in LICENSE.txt\n"
        "compatibility: Requires a coding agent with project file tools\n"
        "metadata:\n"
        "  author: anthropics\n"
        "allowed-tools: Read Write Bash\n"
        "---\n"
        "Create a polished frontend for $ARGUMENTS.\n",
        encoding="utf-8",
    )
    (package / "LICENSE.txt").write_text("Sample license\n", encoding="utf-8")
    return package


def test_standard_agent_skill_fields_are_accepted_without_rewriting_upstream(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = _write_public_skill(project)
    original = (package / "SKILL.md").read_bytes()

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    skill = snapshot.skills["frontend-design"]
    assert skill.license == "Complete terms in LICENSE.txt"
    assert skill.compatibility == "Requires a coding agent with project file tools"
    assert skill.metadata == {"author": "anthropics"}
    assert skill.allowed_tools == "Read Write Bash"
    assert skill.visible_tools == ()
    assert (package / "SKILL.md").read_bytes() == original


def test_install_receipt_overrides_effective_fakuicode_settings_and_tracks_both_fingerprints(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    package = _write_public_skill(project)
    upstream_fingerprint = fingerprint_upstream(package)
    receipt_root = package / ".fakuicode"
    receipt_root.mkdir()
    receipt = {
        "schema-version": 1,
        "requested-url": "https://www.skills.sh/anthropics/skills/frontend-design",
        "source-url": "https://github.com/anthropics/skills",
        "revision": "a" * 40,
        "skill-path": "skills/frontend-design",
        "upstream-fingerprint": upstream_fingerprint,
        "fakuicode": {
            "invocation": "auto",
            "visible-tools": [
                "read_file",
                "find_files",
                "search_code",
                "write_file",
                "edit_file",
                "run_command",
            ],
            "execution": "shared",
            "history-turns": 0,
            "profile": "inherit",
        },
    }
    (receipt_root / "install.yaml").write_text(
        yaml.safe_dump(receipt, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(
        {"read_file", "find_files", "search_code", "write_file", "edit_file", "run_command"}
    )

    skill = snapshot.skills["frontend-design"]
    assert skill.install_receipt is not None
    assert skill.install_receipt.upstream_fingerprint == upstream_fingerprint
    assert skill.fingerprint != upstream_fingerprint
    assert skill.visible_tools == (
        "read_file",
        "find_files",
        "search_code",
        "write_file",
        "edit_file",
        "run_command",
    )


def test_installed_skill_is_disabled_when_upstream_files_no_longer_match_receipt(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = _write_public_skill(project)
    receipt_root = package / ".fakuicode"
    receipt_root.mkdir()
    (receipt_root / "install.yaml").write_text(
        yaml.safe_dump(
            {
                "schema-version": 1,
                "requested-url": "https://github.com/anthropics/skills",
                "source-url": "https://github.com/anthropics/skills",
                "revision": "b" * 40,
                "skill-path": "skills/frontend-design",
                "upstream-fingerprint": fingerprint_upstream(package),
                "fakuicode": {
                    "invocation": "auto",
                    "visible-tools": [],
                    "execution": "shared",
                    "history-turns": 0,
                    "profile": "inherit",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (package / "LICENSE.txt").write_text("tampered\n", encoding="utf-8")

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert "frontend-design" not in snapshot.skills
    assert any("fingerprint" in item.message for item in snapshot.diagnostics)


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested-url", "https://github.com:444/anthropics/skills"),
        ("requested-url", "https://github.com/anthropics/skills?ref=main"),
        ("requested-url", "https://github.com/anthropics/skills/issues/1"),
        ("source-url", "https://github.com/anthropics/skills#readme"),
    ],
)
def test_install_receipt_rejects_noncanonical_urls(tmp_path: Path, field: str, value: str) -> None:
    package = _write_public_skill(tmp_path)
    receipt_root = package / ".fakuicode"
    receipt_root.mkdir()
    receipt = {
        "schema-version": 1,
        "requested-url": "https://github.com/anthropics/skills",
        "source-url": "https://github.com/anthropics/skills",
        "revision": "a" * 40,
        "skill-path": "skills/frontend-design",
        "upstream-fingerprint": fingerprint_upstream(package),
        "fakuicode": {
            "invocation": "auto",
            "visible-tools": [],
            "execution": "shared",
            "history-turns": 0,
            "profile": "inherit",
        },
    }
    receipt[field] = value
    (receipt_root / "install.yaml").write_text(yaml.safe_dump(receipt), encoding="utf-8")

    with pytest.raises(SkillParseError, match="URL|url|invalid"):
        parse_skill_package(package, SkillSource.PROJECT)
