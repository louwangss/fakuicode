from __future__ import annotations

from pathlib import Path

import pytest


def test_file_tools_read_with_line_numbers_write_and_edit_once(tmp_path: Path) -> None:
    from fakuicode.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)
    written = WriteFileTool(policy).execute({"path": "nested/notes.txt", "content": "before\nsecond\n"})
    edited = EditFileTool(policy).execute({"path": "nested/notes.txt", "old_text": "before", "new_text": "after"})
    read = ReadFileTool(policy).execute({"path": "nested/notes.txt"})

    assert written.success is True
    assert "notes.txt" in written.summary
    assert edited.success is True
    assert "-before" in edited.summary and "+after" in edited.summary
    assert read.output == "1: after\n2: second\n"


def test_read_file_preserves_content_beyond_the_old_character_limit(tmp_path: Path) -> None:
    from fakuicode.tools.filesystem import ReadFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    content = "x" * 13_000 + "tail-marker\n"
    (tmp_path / "long.txt").write_text(content, encoding="utf-8")

    result = ReadFileTool(WorkspacePolicy(tmp_path)).execute({"path": "long.txt"})

    assert result.output.endswith("tail-marker\n")
    assert "output truncated" not in result.output


@pytest.mark.parametrize(("content", "expected_count"), [("other", 0), ("same same", 2)])
def test_edit_file_reports_exact_match_count_without_changing_file(
    tmp_path: Path, content: str, expected_count: int
) -> None:
    from fakuicode.errors import ToolExecutionError
    from fakuicode.tools.filesystem import EditFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    target = tmp_path / "notes.txt"
    target.write_text(content, encoding="utf-8")

    with pytest.raises(ToolExecutionError, match=fr"matched {expected_count} times"):
        EditFileTool(WorkspacePolicy(tmp_path)).execute({"path": "notes.txt", "old_text": "same", "new_text": "new"})

    assert target.read_text(encoding="utf-8") == content


def test_find_files_and_search_code_accept_a_relative_scope(tmp_path: Path) -> None:
    from fakuicode.tools.filesystem import FindFilesTool, SearchCodeTool
    from fakuicode.tools.policy import WorkspacePolicy

    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text("needle = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("needle outside scope\n", encoding="utf-8")
    policy = WorkspacePolicy(tmp_path)

    found = FindFilesTool(policy).execute({"pattern": "**/*.py", "path": "src"})
    matches = SearchCodeTool(policy).execute({"query": "needle", "path": "src"})

    assert found.output == "src/main.py"
    assert matches.output == "src/main.py:1: needle = 1"


def test_find_and_search_skip_generated_cache_directories(tmp_path: Path) -> None:
    from fakuicode.tools.filesystem import FindFilesTool, SearchCodeTool
    from fakuicode.tools.policy import WorkspacePolicy

    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text("needle\n", encoding="utf-8")
    python_cache = source / "__pycache__"
    python_cache.mkdir()
    (python_cache / "main.cpython-311.pyc").write_text("needle\n", encoding="utf-8")
    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "lastfailed").write_text("needle\n", encoding="utf-8")

    policy = WorkspacePolicy(tmp_path)
    found = FindFilesTool(policy).execute({"pattern": "**/*"})
    matches = SearchCodeTool(policy).execute({"query": "needle"})

    assert found.output == "src/main.py"
    assert matches.output == "src/main.py:1: needle"


def test_file_tools_reject_sensitive_configuration_targets(tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.filesystem import ReadFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    (tmp_path / "fakuicode.yaml").write_text("api_key: not-a-real-key", encoding="utf-8")

    with pytest.raises(ToolPolicyError, match="sensitive"):
        ReadFileTool(WorkspacePolicy(tmp_path)).execute({"path": "fakuicode.yaml"})


def test_only_read_file_can_access_context_artifacts(tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    relative = ".fakuicode/context-artifacts/conversation-1/result.txt"
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text("complete\n", encoding="utf-8")
    policy = WorkspacePolicy(tmp_path)

    assert ReadFileTool(policy).execute({"path": relative}).output == "1: complete\n"
    with pytest.raises(ToolPolicyError, match="sensitive"):
        WriteFileTool(policy).execute({"path": relative, "content": "changed"})
    with pytest.raises(ToolPolicyError, match="sensitive"):
        EditFileTool(policy).execute(
            {"path": relative, "old_text": "complete", "new_text": "changed"}
        )


def test_find_and_search_mark_results_when_the_match_limit_is_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fakuicode.tools.filesystem as filesystem

    (tmp_path / "one.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(filesystem, "_MAX_MATCHES", 1)
    policy = filesystem.WorkspacePolicy(tmp_path)

    found = filesystem.FindFilesTool(policy).execute({"pattern": "*.py"})
    matches = filesystem.SearchCodeTool(policy).execute({"query": "needle"})

    assert "truncated" in found.output
    assert "truncated" in matches.output


def test_file_tool_preparation_exposes_only_the_normalized_permission_target(tmp_path: Path) -> None:
    from fakuicode.tools.filesystem import EditFileTool, SearchCodeTool, WriteFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)
    write = WriteFileTool(policy).prepare({"path": "src/../notes.txt", "content": "private content"})
    edit = EditFileTool(policy).prepare(
        {"path": "notes.txt", "old_text": "private old", "new_text": "private new"}
    )
    search = SearchCodeTool(policy).prepare({"query": "private query", "path": "."})

    assert write.target == "notes.txt"
    assert edit.target == "notes.txt"
    assert search.target == "."
    assert "private" not in write.target + edit.target + search.target


def test_file_tool_rejects_unknown_arguments_during_preparation(tmp_path: Path) -> None:
    from fakuicode.errors import ToolExecutionError
    from fakuicode.tools.filesystem import ReadFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolExecutionError, match="unexpected"):
        ReadFileTool(WorkspacePolicy(tmp_path)).prepare({"path": "README.md", "content": "ignored"})


def test_prepared_file_arguments_are_immutable_and_execute_the_prepared_path(tmp_path: Path) -> None:
    from fakuicode.tools.filesystem import WriteFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    raw = {"path": "first.txt", "content": "first"}
    tool = WriteFileTool(WorkspacePolicy(tmp_path))
    prepared = tool.prepare(raw)
    raw["path"] = "second.txt"
    raw["content"] = "second"

    tool.execute_prepared(prepared.arguments)

    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "first"
    assert not (tmp_path / "second.txt").exists()
    with pytest.raises(TypeError):
        prepared.arguments["content"] = "changed"  # type: ignore[index]


def test_prepared_write_rechecks_a_symlink_replaced_after_authorization(tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.filesystem import WriteFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    tool = WriteFileTool(WorkspacePolicy(tmp_path))
    prepared = tool.prepare({"path": "inside/note.txt", "content": "blocked"})
    inside.rmdir()
    try:
        inside.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows environment cannot create directory symlinks.")

    with pytest.raises(ToolPolicyError, match="outside"):
        tool.execute_prepared(prepared.arguments)

    assert not (outside / "note.txt").exists()


def test_prepared_write_rejects_a_target_that_resolves_differently_at_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.filesystem import WriteFileTool
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)
    tool = WriteFileTool(policy)
    prepared = tool.prepare({"path": "note.txt", "content": "blocked"})
    monkeypatch.setattr(policy, "resolve_path", lambda target: tmp_path / "changed.txt")

    with pytest.raises(ToolPolicyError, match="changed"):
        tool.execute_prepared(prepared.arguments)

    assert not (tmp_path / "note.txt").exists()
    assert not (tmp_path / "changed.txt").exists()
