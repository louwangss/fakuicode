from __future__ import annotations

import os
from pathlib import Path

import pytest

from fakuicode.errors import PermissionPersistenceError
from fakuicode.permissions.config import PermissionConfigRepository, PermissionPaths
from fakuicode.permissions.models import PermissionMode, RuleEffect, RuleSource


def _paths(tmp_path: Path) -> tuple[Path, PermissionPaths]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    return workspace, PermissionPaths.for_workspace(workspace, home=home)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_missing_permission_files_load_an_empty_default_snapshot(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)

    snapshot = PermissionConfigRepository(paths, workspace).load()

    assert snapshot.mode is PermissionMode.DEFAULT
    assert snapshot.locked is False
    assert snapshot.project_trusted is False
    assert snapshot.user_rules == ()
    assert snapshot.project_shared_rules == ()
    assert snapshot.project_local_rules == ()
    assert snapshot.diagnostics == ()


def test_user_config_loads_mode_and_rules_from_strict_schema(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        """version: 1
mode: strict
rules:
  allow:
    - run_command(git status)
  deny:
    - read_file(secrets/**)
""",
    )

    snapshot = PermissionConfigRepository(paths, workspace).load()

    assert snapshot.mode is PermissionMode.STRICT
    assert [(rule.effect, rule.source) for rule in snapshot.user_rules] == [
        (RuleEffect.ALLOW, RuleSource.USER),
        (RuleEffect.DENY, RuleSource.USER),
    ]


@pytest.mark.parametrize("path_name", ["project_shared", "project_local"])
def test_project_config_cannot_set_permission_mode(tmp_path: Path, path_name: str) -> None:
    workspace, paths = _paths(tmp_path)
    _write(getattr(paths, path_name), "version: 1\nmode: trusted\nrules: {}\n")

    snapshot = PermissionConfigRepository(paths, workspace).load()

    assert snapshot.locked is True
    assert snapshot.mode is PermissionMode.STRICT
    assert any(path_name.replace("_", " ") in diagnostic for diagnostic in snapshot.diagnostics)


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\nversion: 1\nrules: {}\n",
        "version: 2\nrules: {}\n",
        "version: 1\nunknown: true\nrules: {}\n",
        "version: 1\nrules:\n  maybe:\n    - read_file(*)\n",
        "version: 1\nrules:\n  allow:\n    - Bash(git *)\n",
        "? [unhashable, key]\n: value\nversion: 1\nrules: {}\n",
        "version: [\n",
    ],
)
def test_invalid_rule_file_enters_locked_strict_state(tmp_path: Path, content: str) -> None:
    workspace, paths = _paths(tmp_path)
    _write(paths.project_shared, content)

    snapshot = PermissionConfigRepository(paths, workspace).load()

    assert snapshot.locked is True
    assert snapshot.mode is PermissionMode.STRICT
    assert snapshot.diagnostics
    assert content not in "\n".join(snapshot.diagnostics)


def test_invalid_trust_file_fails_closed_without_locking_safe_reads(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(paths.trust, "version: [\n")

    snapshot = PermissionConfigRepository(paths, workspace).load()

    assert snapshot.locked is False
    assert snapshot.project_trusted is False
    assert snapshot.warnings


def test_trust_repository_normalizes_identity_and_avoids_duplicates(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    repository = PermissionConfigRepository(paths, workspace)

    trusted = repository.set_project_trusted(True)
    trusted_again = repository.set_project_trusted(True)
    loaded = repository.load()

    assert trusted.project_trusted is True
    assert trusted_again.project_trusted is True
    assert loaded.project_trusted is True
    data = paths.trust.read_text(encoding="utf-8")
    assert data.count(repository.workspace_identity) == 1


def test_permanent_rule_save_is_idempotent_and_reloadable(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    repository = PermissionConfigRepository(paths, workspace)
    snapshot = repository.load()

    first = repository.save_project_local_allow(snapshot, r"run_command(python -c \*\?\[x\])")
    second = repository.save_project_local_allow(first, r"run_command(python -c \*\?\[x\])")
    reloaded = repository.load()

    assert len(second.project_local_rules) == 1
    assert len(reloaded.project_local_rules) == 1
    assert reloaded.project_local_rules[0].matches("python -c *?[x]")
    assert paths.project_local.read_text(encoding="utf-8").count("run_command") == 1


def test_project_config_path_outside_workspace_locks_load_and_rejects_save(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    outside_local = tmp_path / "outside" / "permissions.local.yaml"
    redirected = PermissionPaths(
        paths.user,
        paths.project_shared,
        outside_local,
        paths.trust,
    )
    repository = PermissionConfigRepository(redirected, workspace)

    snapshot = repository.load()

    assert snapshot.locked is True
    assert any("project local" in diagnostic for diagnostic in snapshot.diagnostics)
    with pytest.raises(PermissionPersistenceError, match="not safe"):
        repository.save_project_local_allow(snapshot, "write_file(src/main.py)")
    assert not outside_local.exists()


def test_project_local_symlink_cannot_supply_rules_or_redirect_permanent_save(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    external = tmp_path / "external.yaml"
    _write(external, "version: 1\nrules:\n  allow:\n    - write_file(*)\n")
    paths.project_local.parent.mkdir(parents=True)
    try:
        paths.project_local.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")
    repository = PermissionConfigRepository(paths, workspace)

    snapshot = repository.load()

    assert snapshot.locked is True
    assert snapshot.project_local_rules == ()
    with pytest.raises(PermissionPersistenceError, match="not safe"):
        repository.save_project_local_allow(snapshot, "write_file(src/main.py)")
    assert "write_file(*)" in external.read_text(encoding="utf-8")


def test_failed_atomic_replace_preserves_original_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, paths = _paths(tmp_path)
    _write(paths.project_local, "version: 1\nrules:\n  deny:\n    - write_file(dist/**)\n")
    repository = PermissionConfigRepository(paths, workspace)
    snapshot = repository.load()
    before = paths.project_local.read_text(encoding="utf-8")

    def fail_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr("fakuicode.permissions.config.os.replace", fail_replace)

    with pytest.raises(PermissionPersistenceError):
        repository.save_project_local_allow(snapshot, "write_file(src/new.py)")

    assert paths.project_local.read_text(encoding="utf-8") == before
    assert list(paths.project_local.parent.glob("*.tmp")) == []
