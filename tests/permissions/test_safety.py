from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.permissions.safety import DangerousCommandGuard, serialize_command


@pytest.mark.parametrize(
    "command",
    [
        ["powershell", "-Command", "Get-ChildItem"],
        ["PWSH.EXE", "-NoProfile"],
        [r"C:\Windows\System32\cmd.exe", "/c", "dir"],
        ["bash", "-lc", "pwd"],
        ["sh", "script.sh"],
        ["zsh"],
        ["fish"],
        ["wsl.exe", "git", "status"],
        ["mkfs.ext4", "/dev/sda"],
        ["format.com", "D:"],
        ["diskpart"],
        ["dd", "if=image.iso", "of=/dev/sda"],
        ["shutdown", "-h", "now"],
        ["reboot"],
    ],
)
def test_known_catastrophic_or_shell_commands_are_forbidden(tmp_path: Path, command: list[str]) -> None:
    guard = DangerousCommandGuard(tmp_path)

    assert guard.reason(serialize_command(command)) is not None


@pytest.mark.parametrize("target", ["/", "~", ".", "./"])
def test_recursive_forced_removal_of_broad_roots_is_forbidden(tmp_path: Path, target: str) -> None:
    guard = DangerousCommandGuard(tmp_path)

    assert guard.reason(serialize_command(["rm", "-rf", target])) is not None


def test_recursive_removal_of_home_and_workspace_roots_is_forbidden(tmp_path: Path) -> None:
    home = tmp_path / "home with spaces"
    workspace = tmp_path / "workspace with spaces"
    guard = DangerousCommandGuard(workspace, home=home)

    assert guard.reason(serialize_command(["rm", "-rf", str(home)])) is not None
    assert guard.reason(serialize_command(["rm", "-fr", str(workspace)])) is not None
    assert guard.reason(serialize_command(["rm", "--recursive", "--force", str(workspace)])) is not None


@pytest.mark.parametrize(
    "command",
    [
        ["git", "status"],
        ["git", "reset", "--soft", "HEAD~1"],
        ["git", "push", "origin", "feature"],
        ["rm", "-rf", "build"],
        ["python", "-m", "pytest"],
        ["python", "-m", "pip", "install", "-e", "."],
        ["npm", "run", "build"],
        ["curl", "https://example.com"],
    ],
)
def test_legitimate_development_commands_are_not_hard_denied(tmp_path: Path, command: list[str]) -> None:
    guard = DangerousCommandGuard(tmp_path)

    assert guard.reason(serialize_command(command)) is None


def test_command_serialization_is_stable_and_unambiguous() -> None:
    command = ["python", "-c", "print('hello world')", "*", "back\\slash"]

    assert serialize_command(command) == 'python -c "print(\'hello world\')" "*" "back\\\\slash"'
