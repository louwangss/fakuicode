from __future__ import annotations

from pathlib import Path
import json

import pytest

from fakuicode.skills import (
    BuiltinSkillError,
    SkillDiscovery,
    SkillInvocation,
    SkillSource,
    render_skill_catalog,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Reusable workflow",
    extension: str = "",
    body: str = "Do $ARGUMENTS",
) -> Path:
    package = root / name
    package.mkdir(parents=True)
    extra = f"\n{extension.rstrip()}" if extension.strip() else ""
    (package / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "fakuicode:"
        f"{extra}\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return package


def test_project_skill_shadows_user_and_builtin(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user = tmp_path / "user"
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "deploy", description="builtin")
    _write_skill(user, "deploy", description="user")
    _write_skill(project, "deploy", description="project")

    snapshot = SkillDiscovery(project, user, builtin).refresh({"read_file"})

    assert snapshot.skills["deploy"].source is SkillSource.PROJECT
    assert snapshot.skills["deploy"].description == "project"


def test_invalid_higher_priority_candidate_still_shadows_lower_skill(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user = tmp_path / "user"
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "deploy", description="builtin")
    package = _write_skill(project, "deploy")
    (package / "SKILL.md").write_text("not frontmatter", encoding="utf-8")

    snapshot = SkillDiscovery(project, user, builtin).refresh(set())

    assert "deploy" not in snapshot.skills
    assert any(item.name == "deploy" and item.source is SkillSource.PROJECT for item in snapshot.diagnostics)


@pytest.mark.parametrize(
    "document",
    [
        "---\nname: demo\nname: other\ndescription: x\nfakuicode: {}\n---\nbody\n",
        "---\nname: Demo\ndescription: x\nfakuicode: {}\n---\nbody\n",
        "---\nname: demo\ndescription: x\nunknown: true\nfakuicode: {}\n---\nbody\n",
        "---\nname: demo\ndescription: x\nfakuicode:\n  execution: shared\n  history-turns: 1\n---\nbody\n",
        "---\nname: demo\ndescription: x\nfakuicode:\n  execution: shared\n  profile: other\n---\nbody\n",
    ],
)
def test_strict_frontmatter_rejects_invalid_documents(tmp_path: Path, document: str) -> None:
    project = tmp_path / "project"
    package = project / "demo"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(document, encoding="utf-8")

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert "demo" not in snapshot.skills
    assert snapshot.diagnostics


def test_manual_skill_is_hidden_from_model_catalog_and_kept_for_commands(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project, "commit", extension="  invocation: manual")
    _write_skill(project, "test", extension="  invocation: auto")
    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert snapshot.skills["commit"].invocation is SkillInvocation.MANUAL
    assert set(snapshot.command_names) == {"commit", "test"}
    assert "commit" not in render_skill_catalog(snapshot, context_window=8_000).text
    assert "test" in render_skill_catalog(snapshot, context_window=8_000).text


def test_unknown_visible_tool_disables_only_that_skill(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project, "bad", extension="  visible-tools: [missing_tool]")
    _write_skill(project, "good", extension="  visible-tools: [read_file]")

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh({"read_file"})

    assert set(snapshot.skills) == {"good"}
    assert any(item.name == "bad" and item.code == "unknown_tool" for item in snapshot.diagnostics)


def test_builtin_parse_failure_is_fatal(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    package = builtin / "broken"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("broken", encoding="utf-8")

    with pytest.raises(BuiltinSkillError):
        SkillDiscovery(tmp_path / "project", tmp_path / "user", builtin).refresh(set())


def test_argument_rendering_replaces_placeholder_or_appends_data_block(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project, "replace", body="Handle: $ARGUMENTS")
    _write_skill(project, "append", body="Handle it")
    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert snapshot.skills["replace"].render("a b") == "Handle: a b"
    assert snapshot.skills["append"].render("a b") == "Handle it\n\nARGUMENTS:\na b"


def test_catalog_budget_is_bounded_and_warns_when_entries_are_omitted(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for name in ("alpha", "beta", "gamma"):
        _write_skill(project, name, description="x" * 1024)
    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    catalog = render_skill_catalog(snapshot, context_window=200)

    assert catalog.estimated_tokens <= 4
    assert catalog.omitted_names


def test_catalog_keeps_names_intact_and_omits_a_priority_suffix(tmp_path: Path) -> None:
    project = tmp_path / "project"
    long_name = "b" * 40
    for name in ("a", long_name, "c"):
        _write_skill(project, name, description="description")
    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    catalog = render_skill_catalog(snapshot, context_window=500)

    assert catalog.text.splitlines()[0].startswith("- a")
    assert long_name not in catalog.text
    assert catalog.omitted_names == (long_name, "c")


def test_dedicated_tool_descriptor_is_strict_and_changes_package_fingerprint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = _write_skill(project, "format")
    (package / "tools").mkdir()
    (package / "scripts").mkdir()
    script = package / "scripts" / "format.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    descriptor = {
        "name": "format_text",
        "description": "Format text",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "entrypoint": "scripts/format.py",
    }
    (package / "tools" / "format_text.json").write_text(json.dumps(descriptor), encoding="utf-8")

    first = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())
    script.write_text("print('changed')\n", encoding="utf-8")
    second = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert first.skills["format"].runtime_tool_names == ("skill__format__format_text",)
    assert first.skills["format"].fingerprint != second.skills["format"].fingerprint


def test_dedicated_tool_rejects_entrypoint_outside_scripts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = _write_skill(project, "unsafe")
    (package / "tools").mkdir()
    (package / "outside.py").write_text("print('bad')", encoding="utf-8")
    (package / "tools" / "unsafe.json").write_text(
        json.dumps(
            {
                "name": "unsafe",
                "description": "Unsafe",
                "input_schema": {"type": "object"},
                "entrypoint": "outside.py",
            }
        ),
        encoding="utf-8",
    )

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert "unsafe" not in snapshot.skills


def test_dedicated_tool_descriptor_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = _write_skill(project, "duplicate")
    (package / "tools").mkdir()
    (package / "scripts").mkdir()
    (package / "scripts" / "echo.py").write_text("print('{}')", encoding="utf-8")
    (package / "tools" / "echo.json").write_text(
        '{"name":"echo","name":"other","description":"x",'
        '"input_schema":{"type":"object"},"entrypoint":"scripts/echo.py"}',
        encoding="utf-8",
    )

    snapshot = SkillDiscovery(project, tmp_path / "user", tmp_path / "builtin").refresh(set())

    assert "duplicate" not in snapshot.skills


def test_bundled_commit_review_and_test_samples_match_their_execution_contract(tmp_path: Path) -> None:
    import fakuicode.skills as package

    builtin = Path(package.__file__).parent / "builtin"
    snapshot = SkillDiscovery(tmp_path / "project", tmp_path / "user", builtin).refresh(
        {"read_file", "find_files", "search_code", "run_command"}
    )

    assert set(snapshot.skills) == {"commit", "review", "test"}
    assert snapshot.skills["commit"].invocation is SkillInvocation.MANUAL
    assert snapshot.skills["review"].invocation is SkillInvocation.MANUAL
    assert snapshot.skills["test"].execution.value == "isolated"
    assert snapshot.skills["test"].history_turns == 1
