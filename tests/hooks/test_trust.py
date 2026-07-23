from __future__ import annotations

from pathlib import Path

from fakuicode.hooks.trust import HookTrustIdentity, HookTrustRepository
from fakuicode.hooks.config import HookConfigRepository, HookPaths
from fakuicode.mcp.trust import workspace_id


def test_hook_trust_is_scoped_to_workspace_and_exact_content_fingerprint(tmp_path: Path) -> None:
    repository = HookTrustRepository(tmp_path / "trusted-hooks.yaml")
    identity = HookTrustIdentity("a" * 64, "b" * 64)

    assert repository.is_trusted(identity) is False

    repository.approve(identity)

    assert repository.is_trusted(identity) is True
    assert repository.is_trusted(HookTrustIdentity("a" * 64, "c" * 64)) is False
    assert repository.is_trusted(HookTrustIdentity("d" * 64, "b" * 64)) is False


def test_invalid_trust_storage_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "trusted-hooks.yaml"
    path.write_text("version: [\n", encoding="utf-8")
    repository = HookTrustRepository(path)

    assert repository.is_trusted(HookTrustIdentity("a" * 64, "b" * 64)) is False
    assert repository.diagnostic is not None


def test_project_rules_activate_only_for_the_approved_file_fingerprint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = HookPaths.for_workspace(workspace, home=tmp_path / "home")
    paths.project.parent.mkdir()
    paths.project.write_text(
        "version: 1\nhooks:\n  - event: turn_start\n    action: {type: prompt, content: project}\n",
        encoding="utf-8",
    )
    trust = HookTrustRepository(paths.trust)
    repository = HookConfigRepository(paths, workspace, trust_repository=trust)

    untrusted = repository.load()
    trust.approve(HookTrustIdentity(workspace_id(workspace), untrusted.project_fingerprint or ""))
    trusted = repository.load()
    paths.project.write_text(paths.project.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = repository.load()

    assert untrusted.rules == ()
    assert len(untrusted.project_rules) == 1
    assert len(trusted.rules) == 1
    assert changed.rules == ()
