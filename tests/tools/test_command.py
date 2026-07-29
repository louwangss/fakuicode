from __future__ import annotations

import sys
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest


def test_command_tool_uses_the_unified_execute_contract(tmp_path: Path) -> None:
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    result = RunCommandTool(WorkspacePolicy(tmp_path)).execute({"command": [sys.executable, "-c", "print('ok')"]})

    assert result.success is True
    assert result.output == "stdout:\nok\n\nstderr:\n\nexit_code: 0"
    assert "exit 0" in result.summary


def test_command_tool_returns_a_bounded_failure_result(tmp_path: Path) -> None:
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    result = RunCommandTool(WorkspacePolicy(tmp_path)).execute(
        {"command": [sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"]}
    )

    assert result.success is False
    assert "stdout:\nbad\n" in result.output
    assert "exit_code: 3" in result.output
    assert "exit 3" in result.summary


def test_command_tool_preserves_output_beyond_the_old_character_limit(tmp_path: Path) -> None:
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    result = RunCommandTool(WorkspacePolicy(tmp_path)).execute(
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; print('x' * 13000 + 'stdout-tail'); "
                "print('y' * 13000 + 'stderr-tail', file=sys.stderr)",
            ]
        }
    )

    assert result.success is True
    assert "stdout-tail" in result.output
    assert "stderr-tail" in result.output
    assert "output truncated" not in result.output


def test_command_tool_stages_large_output_and_returns_a_bounded_recoverable_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fakuicode.context import ContextPolicy, approximate_token_count
    from fakuicode.context_artifacts import ContextArtifactStore
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    artifacts = ContextArtifactStore(tmp_path, "conversation-1")
    integrity_checks: list[Path] = []

    def track_integrity_check(path: Path) -> str:
        integrity_checks.append(path)
        return ContextArtifactStore._file_sha256(path)

    monkeypatch.setattr(artifacts, "_file_sha256", track_integrity_check)
    result = RunCommandTool(
        WorkspacePolicy(tmp_path),
        artifact_store=artifacts,
    ).execute(
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; print('HEAD-' + 'x' * 50000 + '-STDOUT-TAIL'); "
                "print('ERR-HEAD-' + 'y' * 50000 + '-STDERR-TAIL', file=sys.stderr)",
            ]
        }
    )

    artifact_metadata = result.metadata["context_artifact"]
    artifact = tmp_path / artifact_metadata["read_path"]
    complete = artifact.read_text(encoding="utf-8")
    assert result.success is True
    assert approximate_token_count(result.output) <= ContextPolicy().tool_preview_max_tokens
    assert "HEAD-" in result.output
    assert "-STDERR-TAIL" in result.output
    assert "-STDOUT-TAIL" in complete
    assert "-STDERR-TAIL" in complete
    assert artifact_metadata["byte_size"] == len(complete.encode("utf-8"))
    assert artifact_metadata["content_sha256"] in artifact.name
    assert integrity_checks == [artifact]
    assert list(artifacts.conversation_dir.glob(".staging-*.tmp")) == []


def test_command_tool_supports_portable_ls_without_listing_sensitive_files(tmp_path: Path) -> None:
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")
    (tmp_path / ".env").write_text("not-a-real-secret", encoding="utf-8")
    (tmp_path / "fakuicode.yaml").write_text("api_key: not-a-real-key", encoding="utf-8")

    result = RunCommandTool(WorkspacePolicy(tmp_path)).execute({"command": ["ls", "-la"]})

    assert result.success is True
    assert "visible.txt" in result.output
    assert ".env" not in result.output
    assert "fakuicode.yaml" not in result.output


def test_portable_ls_stages_large_output_instead_of_returning_it_inline(tmp_path: Path) -> None:
    from fakuicode.context import ContextPolicy, approximate_token_count
    from fakuicode.context_artifacts import ContextArtifactStore
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    for index in range(750):
        (tmp_path / f"entry-{index:04d}-{'x' * 30}.txt").touch()
    artifacts = ContextArtifactStore(tmp_path, "conversation-1")

    result = RunCommandTool(
        WorkspacePolicy(tmp_path),
        artifact_store=artifacts,
    ).execute({"command": ["ls"]})

    metadata = result.metadata["context_artifact"]
    complete = (tmp_path / metadata["read_path"]).read_text(encoding="utf-8")
    assert result.success is True
    assert approximate_token_count(result.output) <= ContextPolicy().tool_preview_max_tokens
    assert "entry-0000-" in complete
    assert "entry-0749-" in complete


def test_command_tool_stops_a_timed_out_process(tmp_path: Path) -> None:
    from fakuicode.errors import ToolExecutionError
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolExecutionError, match="timed out"):
        RunCommandTool(WorkspacePolicy(tmp_path), timeout_seconds=0.01).execute(
            {"command": [sys.executable, "-c", "import time; time.sleep(1)"]}
        )


def test_command_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    from fakuicode.errors import ToolExecutionError
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    marker = tmp_path / "descendant-survived.txt"
    child = (
        "import pathlib,time; time.sleep(0.4); "
        f"pathlib.Path({str(marker)!r}).write_text('survived', encoding='utf-8')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(5)"
    )

    with pytest.raises(ToolExecutionError, match="timed out"):
        RunCommandTool(WorkspacePolicy(tmp_path), timeout_seconds=0.1).execute(
            {"command": [sys.executable, "-c", parent]}
        )

    deadline = monotonic() + 1
    while monotonic() < deadline and not marker.exists():
        sleep(0.05)
    assert not marker.exists()


def test_command_tool_stops_when_the_active_turn_is_cancelled(tmp_path: Path) -> None:
    from fakuicode.errors import RequestCancelled
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    cancelled = Event()
    cancelled.set()

    with pytest.raises(RequestCancelled):
        RunCommandTool(WorkspacePolicy(tmp_path)).execute(
            {"command": [sys.executable, "-c", "import time; time.sleep(1)"]}, cancel_event=cancelled
        )


def test_command_preparation_freezes_argv_and_uses_stable_permission_target(tmp_path: Path) -> None:
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    argv = [sys.executable, "-c", "print('hello world')"]
    tool = RunCommandTool(WorkspacePolicy(tmp_path))
    prepared = tool.prepare({"command": argv})
    argv[-1] = "raise SystemExit(9)"

    result = tool.execute_prepared(prepared.arguments)

    assert prepared.target.endswith(' -c "print(\'hello world\')"')
    assert result.success is True
    assert "hello world" in result.output
    assert isinstance(prepared.arguments["command"], tuple)


def test_command_preparation_rejects_unknown_arguments(tmp_path: Path) -> None:
    from fakuicode.errors import ToolExecutionError
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolExecutionError, match="unexpected"):
        RunCommandTool(WorkspacePolicy(tmp_path)).prepare({"command": ["git", "status"], "shell": True})
