"""Tests for the model-visible project-instruction rendering boundary."""

from __future__ import annotations

from pathlib import Path

from fakuicode.instructions import (
    InstructionDiagnosticCode,
    InstructionLimits,
    InstructionLoader,
    InstructionScope,
)


def test_render_orders_sources_from_low_to_high_priority_with_safe_boundaries(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(user_home / ".fakuicode" / "AGENTS.md", "user sentinel")
    _write(workspace / "AGENTS.md", "project sentinel")
    _write(workspace / ".fakuicode" / "AGENTS.md", "local sentinel")

    text = InstructionLoader(workspace, user_home=user_home).load().text

    assert text.index("user sentinel") < text.index("project sentinel") < text.index("local sentinel")
    assert '<instruction-source scope="user" path="AGENTS.md">' in text
    assert '<instruction-source scope="project" path="AGENTS.md">' in text
    assert '<instruction-source scope="project_local" path=".fakuicode/AGENTS.md">' in text
    assert str(workspace) not in text
    assert str(user_home) not in text


def test_render_expands_includes_at_their_original_position_and_keeps_duplicates(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(
        workspace / "AGENTS.md",
        "before\n@include rules.md\nafter\n@include rules.md",
    )
    _write(workspace / "rules.md", "included sentinel")

    text = InstructionLoader(workspace, user_home=user_home).load().text

    assert text.index("before") < text.index("included sentinel") < text.index("after")
    assert text.count("included sentinel") == 2
    assert text.count('<included-instructions path="rules.md">') == 2


def test_render_keeps_higher_priority_layers_inside_the_payload_budget(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(user_home / ".fakuicode" / "AGENTS.md", "low priority\n" * 30)
    _write(workspace / ".fakuicode" / "AGENTS.md", "high priority sentinel")
    limits = InstructionLimits(max_payload_bytes=200)

    snapshot = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    assert snapshot.byte_count <= limits.max_payload_bytes
    assert "high priority sentinel" in snapshot.text


def test_render_reports_a_lower_priority_main_layer_dropped_by_the_payload_budget(
    tmp_path: Path,
) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(user_home / ".fakuicode" / "AGENTS.md", "low priority sentinel")
    _write(workspace / ".fakuicode" / "AGENTS.md", "H" * 200)
    limits = InstructionLimits(max_payload_bytes=150)

    snapshot = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    assert "low priority sentinel" not in snapshot.text
    assert InstructionDiagnosticCode.MAIN_TRUNCATED in {
        diagnostic.code
        for diagnostic in snapshot.diagnostics
        if diagnostic.scope is InstructionScope.USER
    }


def test_render_truncates_main_files_only_at_complete_line_boundaries(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(workspace / "AGENTS.md", "first complete line\n" + "x" * 100)
    limits = InstructionLimits(max_payload_bytes=130)

    snapshot = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    assert snapshot.byte_count <= limits.max_payload_bytes
    assert "first complete line" in snapshot.text
    assert "x" * 100 not in snapshot.text
    assert "[instruction truncated]" in snapshot.text
    assert any(
        diagnostic.code is InstructionDiagnosticCode.MAIN_TRUNCATED
        and diagnostic.scope is InstructionScope.PROJECT
        for diagnostic in snapshot.diagnostics
    )


def test_render_drops_an_over_budget_include_as_an_atomic_block(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(workspace / "AGENTS.md", "before\n@include child.md\nafter")
    _write(workspace / "child.md", "child payload " + "x" * 100)
    limits = InstructionLimits(max_payload_bytes=145)

    snapshot = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    assert snapshot.byte_count <= limits.max_payload_bytes
    assert "before" in snapshot.text
    assert "after" in snapshot.text
    assert "child payload" not in snapshot.text
    assert any(
        diagnostic.code is InstructionDiagnosticCode.INCLUDE_BUDGET
        and diagnostic.scope is InstructionScope.PROJECT
        and diagnostic.source == "AGENTS.md"
        and diagnostic.line == 2
        for diagnostic in snapshot.diagnostics
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
