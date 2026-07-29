from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fakuicode.mcp.models import HttpServerConfig, McpConfigSource, StdioServerConfig
from fakuicode.mcp.trust import (
    McpTrustRepository,
    McpTrustStorageError,
    build_trust_request,
    server_identity,
    workspace_id,
)


def _stdio(**changes: object) -> StdioServerConfig:
    values = {
        "name": "local",
        "source": McpConfigSource.PROJECT,
        "command": "python",
        "args": ("server.py",),
        "env_templates": {"TOKEN": "${TOKEN}"},
        "enabled_tools": None,
        "disabled_tools": frozenset(),
    }
    values.update(changes)
    return StdioServerConfig(**values)


def test_workspace_identity_is_stable(tmp_path: Path) -> None:
    assert workspace_id(tmp_path) == workspace_id(tmp_path / ".")
    assert len(workspace_id(tmp_path)) == 64


def test_fingerprint_is_order_independent_and_change_sensitive(tmp_path: Path) -> None:
    first = _stdio(env_templates={"B": "2", "A": "1"})
    reordered = _stdio(env_templates={"A": "1", "B": "2"})
    changed = _stdio(args=("different.py",))
    assert server_identity(tmp_path, first).fingerprint == server_identity(tmp_path, reordered).fingerprint
    assert server_identity(tmp_path, first).fingerprint != server_identity(tmp_path, changed).fingerprint


def test_repository_approve_reuse_invalidate_and_preserve_other_workspace(tmp_path: Path) -> None:
    path = tmp_path / "trust.yaml"
    repository = McpTrustRepository(path)
    first = server_identity(tmp_path / "one", _stdio())
    second = server_identity(tmp_path / "two", _stdio())
    repository.approve(first)
    repository.approve(second)
    assert repository.is_trusted(first)
    assert repository.is_trusted(second)
    assert not repository.is_trusted(server_identity(tmp_path / "one", _stdio(command="node")))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(data["trusted_servers"]) == {first.workspace_id, second.workspace_id}


def test_corrupt_repository_fails_closed_and_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "trust.yaml"
    path.write_text("version: [", encoding="utf-8")
    repository = McpTrustRepository(path)
    identity = server_identity(tmp_path, _stdio())
    assert not repository.is_trusted(identity)
    assert repository.diagnostic is not None
    with pytest.raises(McpTrustStorageError):
        repository.approve(identity)
    assert path.read_text(encoding="utf-8") == "version: ["


def test_duplicate_trust_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "trust.yaml"
    path.write_text("version: 1\nversion: 1\ntrusted_servers: {}\n", encoding="utf-8")
    repository = McpTrustRepository(path)
    assert not repository.is_trusted(server_identity(tmp_path, _stdio()))
    assert repository.diagnostic is not None


def test_trust_prompt_model_is_redacted(tmp_path: Path) -> None:
    config = _stdio(command="sentinel-command", env_templates={"TOKEN": "sentinel-secret-${TOKEN}"})
    request = build_trust_request(tmp_path, config)
    assert request is not None
    rendered = repr(request)
    assert "sentinel-secret" not in rendered
    assert request.environment_names == ("TOKEN",)
    assert request.referenced_variable_names == ("TOKEN",)


def test_stdio_trust_request_preserves_full_argv_and_working_directory(
    tmp_path: Path,
) -> None:
    long_argument = "value with spaces " + "x" * 200
    request = build_trust_request(
        tmp_path,
        _stdio(
            command="python executable",
            args=("server.py", "--label", long_argument),
        ),
    )

    assert request is not None
    assert request.command == "python executable"
    assert request.arguments == ("server.py", "--label", long_argument)
    assert request.working_directory == tmp_path.resolve()
    assert request.argument_count == 3


def test_http_prompt_exposes_names_not_header_values(tmp_path: Path) -> None:
    config = HttpServerConfig(
        "web",
        McpConfigSource.PROJECT,
        "https://example.test/mcp?token=sentinel-query-secret",
        {"Authorization": "Bearer sentinel-secret"},
    )
    request = build_trust_request(tmp_path, config)
    assert request is not None
    assert request.header_names == ("Authorization",)
    assert "sentinel-secret" not in repr(request)
    assert "sentinel-query-secret" not in repr(request)
    assert request.redacted_url == "https://example.test/mcp?<redacted>"
