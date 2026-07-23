from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

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


def test_command_tool_stops_a_timed_out_process(tmp_path: Path) -> None:
    from fakuicode.errors import ToolExecutionError
    from fakuicode.tools.command import RunCommandTool
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolExecutionError, match="timed out"):
        RunCommandTool(WorkspacePolicy(tmp_path), timeout_seconds=0.01).execute(
            {"command": [sys.executable, "-c", "import time; time.sleep(1)"]}
        )


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
