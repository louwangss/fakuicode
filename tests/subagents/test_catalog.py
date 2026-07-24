from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.subagents.catalog import AgentCatalog, CatalogLoadError
from fakuicode.subagents.models import AgentSource, PermissionBehavior


def _write_agent(root: Path, filename: str, frontmatter: str, body: str = "完成分配的任务。") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(
        f"---\n{frontmatter.strip()}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_catalog_prefers_project_then_user_then_builtin(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    common = "name: explore\ndescription: 探索代码"
    _write_agent(builtin, "explore.md", common, "builtin")
    _write_agent(user, "explore.md", common, "user")
    _write_agent(project, "explore.md", common, "project")

    catalog = AgentCatalog.load(project_root=project, user_root=user, builtin_root=builtin)

    definition = catalog.resolve("explore")
    assert definition.prompt == "project"
    assert definition.source is AgentSource.PROJECT


def test_catalog_parses_capability_and_runtime_fields(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_agent(
        project,
        "reviewer.md",
        """
name: code-reviewer
description: 审查代码
tools: [read_file, search_code]
disallowedTools: [run_command]
profile: inherit
maxTurns: 12
permissionMode: dontAsk
background: true
""",
    )

    definition = AgentCatalog.load(project_root=project).resolve("code-reviewer")

    assert definition.tools == ("read_file", "search_code")
    assert definition.disallowed_tools == ("run_command",)
    assert definition.max_turns == 12
    assert definition.permission_mode is PermissionBehavior.DONT_ASK
    assert definition.background is True


def test_invalid_project_override_shadows_lower_priority_definition(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    project = tmp_path / "project"
    _write_agent(builtin, "explore.md", "name: explore\ndescription: 内置探索", "builtin")
    _write_agent(
        project,
        "explore.md",
        "name: explore\ndescription: 项目探索\npermissionMode: autoApprove",
        "invalid",
    )

    catalog = AgentCatalog.load(project_root=project, builtin_root=builtin)

    with pytest.raises(KeyError, match="explore"):
        catalog.resolve("explore")
    assert any(item.name == "explore" and item.source is AgentSource.PROJECT for item in catalog.diagnostics)


@pytest.mark.parametrize(
    "frontmatter",
    (
        "name: Bad_Name\ndescription: invalid name",
        "name: valid\ndescription: x\nmaxTurns: 31",
        "name: valid\ndescription: x\nunknownField: true",
        "name: valid\ndescription: x\ntools: read_file",
    ),
)
def test_invalid_user_definition_is_reported_and_skipped(tmp_path: Path, frontmatter: str) -> None:
    user = tmp_path / "user"
    _write_agent(user, "invalid.md", frontmatter)

    catalog = AgentCatalog.load(user_root=user)

    assert catalog.definitions == {}
    assert len(catalog.diagnostics) == 1


def test_invalid_builtin_definition_fails_fast(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    _write_agent(builtin, "broken.md", "name: broken\ndescription: x\npermissionMode: allowEverything")

    with pytest.raises(CatalogLoadError, match="broken.md"):
        AgentCatalog.load(builtin_root=builtin)

