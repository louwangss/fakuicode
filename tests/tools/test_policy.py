from __future__ import annotations

from pathlib import Path

import pytest


def test_workspace_policy_allows_paths_inside_the_workspace(tmp_path: Path) -> None:
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)

    assert policy.resolve_path("src/app.py") == tmp_path / "src" / "app.py"


@pytest.mark.parametrize("target", ["../outside.txt", str(Path(tmp_path := Path.cwd()).parent / "outside.txt")])
def test_workspace_policy_rejects_paths_outside_the_workspace(target: str, tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolPolicyError, match="workspace"):
        WorkspacePolicy(tmp_path).resolve_path(target)


@pytest.mark.parametrize(
    "command",
    [
        ["git", "push"],
        ["git", "reset", "--hard"],
        ["curl", "https://example.test"],
        ["powershell", "Invoke-WebRequest", "https://example.test"],
    ],
)
def test_workspace_policy_only_normalizes_commands_and_leaves_authorization_to_permission_layers(
    command: list[str], tmp_path: Path
) -> None:
    from fakuicode.tools.policy import WorkspacePolicy

    assert WorkspacePolicy(tmp_path).validate_command(command) == tuple(command)


def test_workspace_policy_allows_local_test_and_safe_git_commands(tmp_path: Path) -> None:
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)

    assert policy.validate_command(["python", "-m", "pytest", "-q"]) == ("python", "-m", "pytest", "-q")
    assert policy.validate_command(["git", "status", "--short"]) == ("git", "status", "--short")


@pytest.mark.parametrize("target", ["fakuicode.yaml", ".env", ".env.local", "cert.pem"])
def test_workspace_policy_rejects_sensitive_file_targets(target: str, tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolPolicyError, match="sensitive"):
        WorkspacePolicy(tmp_path).resolve_path(target)


@pytest.mark.parametrize(
    "target",
    [".fakuicode/permissions.yaml", ".fakuicode/permissions.local.yaml"],
)
def test_workspace_policy_hides_project_permission_files(target: str, tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolPolicyError, match="sensitive"):
        WorkspacePolicy(tmp_path).resolve_path(target)


def test_workspace_policy_only_allows_context_artifacts_with_the_read_exception(tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)
    target = ".fakuicode/context-artifacts/conversation-1/result.txt"

    with pytest.raises(ToolPolicyError, match="sensitive"):
        policy.resolve_path(target)

    assert policy.resolve_path(target, allow_context_artifact_read=True) == tmp_path / target


@pytest.mark.parametrize(
    "target",
    [
        ".fakuicode/permissions.yaml",
        ".fakuicode/permissions.local.yaml",
        ".env",
        "private.key",
    ],
)
def test_context_artifact_read_exception_does_not_expose_other_sensitive_files(
    target: str, tmp_path: Path
) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    with pytest.raises(ToolPolicyError, match="sensitive"):
        WorkspacePolicy(tmp_path).resolve_path(target, allow_context_artifact_read=True)


def test_context_artifact_read_exception_does_not_match_neighboring_paths(tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)
    neighbor = ".fakuicode/context-artifacts-private/result.txt"

    assert policy.resolve_path(neighbor, allow_context_artifact_read=True) == tmp_path / neighbor
    with pytest.raises(ToolPolicyError, match="sensitive"):
        policy.resolve_path(".fakuicode/context-artifacts/result.txt")


def test_sensitive_lexical_path_cannot_be_hidden_by_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    policy = WorkspacePolicy(tmp_path)
    safe_target = tmp_path / "ordinary.yaml"
    monkeypatch.setattr(policy, "_resolve_candidate", lambda candidate: safe_target)

    with pytest.raises(ToolPolicyError, match="sensitive"):
        policy.resolve_path(".fakuicode/permissions.yaml")


def test_nonexistent_target_resolves_its_nearest_existing_symlink_ancestor(tmp_path: Path) -> None:
    from fakuicode.errors import ToolPolicyError
    from fakuicode.tools.policy import WorkspacePolicy

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(ToolPolicyError, match="workspace"):
        WorkspacePolicy(workspace).resolve_path("linked/new/deep/file.txt")

    assert not (outside / "new").exists()
