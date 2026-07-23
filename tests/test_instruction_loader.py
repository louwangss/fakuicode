"""Tests for the project-instruction snapshot data model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from fakuicode.instructions import (
    DEFAULT_INSTRUCTION_LIMITS,
    InstructionDiagnostic,
    InstructionDiagnosticCode,
    InstructionLimits,
    InstructionLoadFailure,
    InstructionLoader,
    InstructionScope,
    InstructionSnapshot,
)


def test_instruction_snapshot_derives_metadata_from_its_immutable_state() -> None:
    diagnostic = InstructionDiagnostic(
        InstructionDiagnosticCode.INVALID_INCLUDE,
        InstructionScope.PROJECT,
        "AGENTS.md",
        line=3,
    )
    snapshot = InstructionSnapshot(
        text="项目规则",
        loaded_layers=(InstructionScope.PROJECT,),
        processed_target_count=2,
        diagnostics=(diagnostic,),
    )

    assert snapshot.byte_count == len("项目规则".encode("utf-8"))
    assert snapshot.warning_count == 1
    with pytest.raises(FrozenInstanceError):
        snapshot.text = "changed"  # type: ignore[misc]


def test_instruction_snapshot_empty_and_failed_have_safe_derived_defaults() -> None:
    empty = InstructionSnapshot.empty()
    failed = InstructionSnapshot.failed()

    assert empty.text == ""
    assert empty.loaded_layers == ()
    assert empty.processed_target_count == 0
    assert empty.warning_count == 0
    assert failed.text == ""
    assert failed.diagnostics == ()
    assert failed.global_failure is InstructionLoadFailure.LOADER_FAILED
    assert failed.warning_count == 1


def test_instruction_limits_derive_bounded_read_sizes_and_reject_non_positive_values() -> None:
    assert DEFAULT_INSTRUCTION_LIMITS.max_include_depth == 5
    assert DEFAULT_INSTRUCTION_LIMITS.max_file_targets == 32
    assert DEFAULT_INSTRUCTION_LIMITS.max_payload_bytes == 32 * 1024
    assert DEFAULT_INSTRUCTION_LIMITS.max_source_bytes == 65_539
    assert DEFAULT_INSTRUCTION_LIMITS.max_read_bytes == 65_540

    for field_name in ("max_include_depth", "max_file_targets", "max_payload_bytes"):
        with pytest.raises(ValueError):
            InstructionLimits(**{field_name: 0})


def test_instruction_diagnostics_only_store_safe_structured_fields() -> None:
    assert tuple(field.name for field in fields(InstructionDiagnostic)) == (
        "code",
        "scope",
        "source",
        "line",
    )


def test_instruction_diagnostics_sanitize_terminal_control_characters() -> None:
    diagnostic = InstructionDiagnostic(
        InstructionDiagnosticCode.FILE_NOT_FOUND,
        InstructionScope.PROJECT,
        "nested/unsafe\x1b[31m\x9b\u202erules.md",
    )

    assert diagnostic.source == "nested/unsafe�[31m��rules.md"


def test_loader_reads_only_the_three_fixed_layers_in_priority_order(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(user_home / ".fakuicode" / "AGENTS.md", "user rules")
    _write(workspace / "AGENTS.md", "project rules")
    _write(workspace / ".fakuicode" / "AGENTS.md", "local rules")
    _write(tmp_path / "AGENTS.md", "parent rules")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.text.index("user rules") < snapshot.text.index("project rules")
    assert snapshot.text.index("project rules") < snapshot.text.index("local rules")
    assert "parent rules" not in snapshot.text
    assert snapshot.loaded_layers == (
        InstructionScope.USER,
        InstructionScope.PROJECT,
        InstructionScope.PROJECT_LOCAL,
    )
    assert snapshot.processed_target_count == 3


def test_loader_cache_keeps_scope_metadata_separate_for_resolved_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.tools.policy import WorkspacePolicy

    user_home = tmp_path / "user"
    workspace = (tmp_path / "workspace").resolve()
    shared = workspace / "shared.md"
    _write(shared, "shared rules")
    original_resolve_path = WorkspacePolicy.resolve_path

    def resolve_alias(
        policy: WorkspacePolicy,
        target: str,
        *,
        allow_context_artifact_read: bool = False,
    ) -> Path:
        requested = Path(target)
        if policy.workspace == workspace and requested.name == "AGENTS.md":
            return shared.resolve()
        return original_resolve_path(
            policy,
            target,
            allow_context_artifact_read=allow_context_artifact_read,
        )

    monkeypatch.setattr(WorkspacePolicy, "resolve_path", resolve_alias)

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.loaded_layers == (
        InstructionScope.PROJECT,
        InstructionScope.PROJECT_LOCAL,
    )
    assert snapshot.text.count('scope="project"') == 1
    assert snapshot.text.count('scope="project_local"') == 1


def test_loader_treats_missing_or_empty_main_files_as_silent_optional_sources(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(workspace / "AGENTS.md", "")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.text == ""
    assert snapshot.loaded_layers == ()
    assert snapshot.diagnostics == ()
    assert snapshot.processed_target_count == 3


def test_loader_requires_complete_utf8_before_injecting_any_main_file_prefix(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    target = workspace / "AGENTS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"safe prefix\xff")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.text == ""
    assert snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.INVALID_UTF8,
            InstructionScope.PROJECT,
            "AGENTS.md",
        ),
    )


def test_loader_accepts_bom_and_crlf_after_complete_bounded_read(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    target = workspace / "AGENTS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xef\xbb\xbffirst\r\nsecond")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert "first\nsecond" in snapshot.text
    assert snapshot.diagnostics == ()


def test_loader_rejects_files_larger_than_the_derived_source_limit_without_prefix_injection(
    tmp_path: Path,
) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    target = workspace / "AGENTS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    limits = InstructionLimits(max_payload_bytes=16)
    target.write_bytes(b"a" * limits.max_source_bytes)

    accepted = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    target.write_bytes(b"b" * limits.max_read_bytes)
    rejected = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    assert all(
        diagnostic.code is not InstructionDiagnosticCode.FILE_TOO_LARGE
        for diagnostic in accepted.diagnostics
    )
    assert rejected.text == ""
    assert rejected.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.FILE_TOO_LARGE,
            InstructionScope.PROJECT,
            "AGENTS.md",
        ),
    )


def test_loader_rejects_a_main_file_symlink_that_escapes_its_allowed_root(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.md"
    _write(outside, "outside rules")
    target = workspace / "AGENTS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    _symlink_or_skip(target, outside)

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.text == ""
    assert snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.PATH_REJECTED,
            InstructionScope.PROJECT,
            "AGENTS.md",
        ),
    )
    assert str(outside) not in repr(snapshot.diagnostics)


def test_loader_rejects_sensitive_and_non_regular_main_file_targets(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    sensitive = workspace / ".env"
    _write(sensitive, "must not load")
    target = workspace / "AGENTS.md"
    _symlink_or_skip(target, sensitive)

    sensitive_snapshot = InstructionLoader(workspace, user_home=user_home).load()

    target.unlink()
    target.mkdir()
    directory_snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert sensitive_snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.PATH_REJECTED,
            InstructionScope.PROJECT,
            "AGENTS.md",
        ),
    )
    assert directory_snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.NOT_REGULAR_FILE,
            InstructionScope.PROJECT,
            "AGENTS.md",
        ),
    )


def test_loader_rejects_a_directory_main_file_as_non_regular(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    (workspace / "AGENTS.md").mkdir(parents=True)

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.NOT_REGULAR_FILE,
            InstructionScope.PROJECT,
            "AGENTS.md",
        ),
    )


def test_loader_enforces_depth_limit_after_five_include_edges(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(workspace / "AGENTS.md", "root\n@include level1.md")
    for level in range(1, 7):
        next_line = "" if level == 6 else f"\n@include level{level + 1}.md"
        _write(workspace / f"level{level}.md", f"level {level}{next_line}")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert "level 5" in snapshot.text
    assert "level 6" not in snapshot.text
    assert InstructionDiagnostic(
        InstructionDiagnosticCode.DEPTH_LIMIT,
        InstructionScope.PROJECT,
        "level5.md",
        line=2,
    ) in snapshot.diagnostics


def test_loader_detects_current_include_cycle_without_suppressing_non_recursive_duplicate(
    tmp_path: Path,
) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(workspace / "AGENTS.md", "root\n@include shared.md\n@include shared.md")
    _write(workspace / "shared.md", "shared payload\n@include AGENTS.md")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.text.count("shared payload") == 2
    assert snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.INCLUDE_CYCLE,
            InstructionScope.PROJECT,
            "shared.md",
            line=2,
        ),
        InstructionDiagnostic(
            InstructionDiagnosticCode.INCLUDE_CYCLE,
            InstructionScope.PROJECT,
            "shared.md",
            line=2,
        ),
    )
    assert snapshot.processed_target_count == 4


def test_loader_enforces_target_limit_before_reading_new_lexical_targets(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(workspace / "AGENTS.md", "root\n@include first.md")
    _write(workspace / "first.md", "first\n@include never-read.md")
    _write(workspace / "never-read.md", "must stay unread")
    limits = InstructionLimits(max_file_targets=4)

    snapshot = InstructionLoader(workspace, user_home=user_home, limits=limits).load()

    assert "first" in snapshot.text
    assert "must stay unread" not in snapshot.text
    assert snapshot.processed_target_count == 4
    assert snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.FILE_TARGET_LIMIT,
            InstructionScope.PROJECT,
            "first.md",
            line=2,
        ),
    )


def test_loader_sorts_diagnostics_by_display_scope_source_line_and_code(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    workspace = tmp_path / "workspace"
    _write(user_home / ".fakuicode" / "AGENTS.md", "@include z.md\n@include a.txt")
    _write(workspace / ".fakuicode" / "AGENTS.md", "@include missing.md")

    snapshot = InstructionLoader(workspace, user_home=user_home).load()

    assert snapshot.diagnostics == (
        InstructionDiagnostic(
            InstructionDiagnosticCode.FILE_NOT_FOUND,
            InstructionScope.USER,
            "AGENTS.md",
            line=1,
        ),
        InstructionDiagnostic(
            InstructionDiagnosticCode.NOT_MARKDOWN,
            InstructionScope.USER,
            "AGENTS.md",
            line=2,
        ),
        InstructionDiagnostic(
            InstructionDiagnosticCode.FILE_NOT_FOUND,
            InstructionScope.PROJECT_LOCAL,
            ".fakuicode/AGENTS.md",
            line=1,
        ),
    )


def test_loader_returns_a_safe_failed_snapshot_for_an_unexpected_top_level_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_layer_specs(self: InstructionLoader) -> tuple[object, ...]:
        raise RuntimeError("must not escape")

    monkeypatch.setattr(InstructionLoader, "_layer_specs", fail_layer_specs)

    snapshot = InstructionLoader(tmp_path / "workspace", user_home=tmp_path / "user").load()

    assert snapshot == InstructionSnapshot.failed()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"当前平台无法创建符号链接：{error.__class__.__name__}")
