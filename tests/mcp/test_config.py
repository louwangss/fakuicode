from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.mcp.config import McpConfigRepository, McpPaths, resolve_server
from fakuicode.mcp.models import (
    DisabledServerConfig,
    HttpServerConfig,
    McpConfigSource,
    ResolvedHttpServerConfig,
    ResolvedStdioServerConfig,
    StdioServerConfig,
)


def _repository(tmp_path: Path) -> tuple[McpConfigRepository, McpPaths]:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    paths = McpPaths.for_workspace(workspace, home=home)
    return McpConfigRepository(paths, workspace), paths


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("body", ["{}", "mcp_servers: {}", "mcp_servers:\n"])
def test_missing_or_empty_configuration_is_quiet(tmp_path: Path, body: str) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, body)
    snapshot = repository.load()
    assert snapshot.servers == ()
    assert snapshot.diagnostics == ()
    assert snapshot.has_configuration is False


def test_yaml_server_error_is_isolated(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(
        paths.user,
        """mcp_servers:
  broken:
    type: stdio
    command: one
    command: two
  valid:
    type: stdio
    command: python
""",
    )
    snapshot = repository.load()
    assert [server.name for server in snapshot.servers] == ["valid"]
    assert snapshot.diagnostics[0].server_name == "broken"


@pytest.mark.parametrize(
    "body",
    [
        "mcp_servers: {}\nmcp_servers: {}",
        "unknown: true",
        "mcp_servers:\n  one: {type: stdio, command: x}\n  one: {type: stdio, command: y}",
    ],
)
def test_root_duplicate_or_unknown_invalidates_layer(tmp_path: Path, body: str) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, body)
    snapshot = repository.load()
    assert snapshot.servers == ()
    assert len(snapshot.diagnostics) == 1


def test_stdio_http_disabled_and_field_types(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(
        paths.user,
        """mcp_servers:
  local:
    type: stdio
    command: python
    args: [server.py]
    env: {TOKEN: "${TOKEN}"}
  remote:
    type: http
    url: https://example.test/mcp
    headers: {Authorization: "Bearer ${TOKEN}"}
  disabled:
    enabled: false
  wrong:
    type: stdio
    command: 5
""",
    )
    snapshot = repository.load()
    assert isinstance(snapshot.servers[0], DisabledServerConfig)
    assert isinstance(snapshot.servers[1], StdioServerConfig)
    assert isinstance(snapshot.servers[2], HttpServerConfig)
    assert [item.server_name for item in snapshot.diagnostics] == ["wrong"]


def test_project_fully_overrides_and_shadows_in_stable_order(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, "mcp_servers:\n  zed: {type: stdio, command: user}\n  alpha: {type: stdio, command: a}")
    _write(paths.project, "mcp_servers:\n  zed: {enabled: false}\n  beta: {type: stdio, command: b}")
    snapshot = repository.load()
    assert [item.name for item in snapshot.servers] == ["alpha", "beta", "zed"]
    assert isinstance(snapshot.servers[-1], DisabledServerConfig)
    assert snapshot.servers[-1].source is McpConfigSource.PROJECT


def test_invalid_project_layer_preserves_user_layer(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, "mcp_servers:\n  good: {type: stdio, command: python}")
    _write(paths.project, "mcp_servers: [")
    snapshot = repository.load()
    assert [item.name for item in snapshot.servers] == ["good"]
    assert snapshot.diagnostics[0].source is McpConfigSource.PROJECT


def test_expand_secrets_and_leave_non_secret_fields_literal(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(
        paths.user,
        """mcp_servers:
  local:
    type: stdio
    command: "${COMMAND}"
    args: ["${ARG}"]
    enabled_tools: ["${TOOL}"]
    env: {TOKEN: "pre-${TOKEN}-${SUFFIX}"}
""",
    )
    raw = repository.load().servers[0]
    resolved, diagnostic = resolve_server(raw, {"TOKEN": "sentinel-secret", "SUFFIX": "x"})
    assert diagnostic is None
    assert isinstance(resolved, ResolvedStdioServerConfig)
    assert resolved.command == "${COMMAND}"
    assert resolved.args == ("${ARG}",)
    assert resolved.environment["TOKEN"].value == "pre-sentinel-secret-x"
    assert "sentinel-secret" not in repr(resolved)


def test_missing_or_malformed_environment_is_server_local(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, 'mcp_servers:\n  local:\n    type: stdio\n    command: x\n    env: {A: "${MISSING}", B: "${BAD-NAME}"}')
    resolved, diagnostic = resolve_server(repository.load().servers[0], {})
    assert resolved is None
    assert diagnostic is not None
    assert "sentinel" not in diagnostic.message


@pytest.mark.parametrize(
    "url,valid",
    [
        ("https://example.test/mcp", True),
        ("http://localhost:8000/mcp", True),
        ("http://127.0.0.8/mcp", True),
        ("http://[::1]/mcp", True),
        ("http://example.test/mcp", False),
        ("https://user:pass@example.test/mcp", False),
        ("https://example.test/mcp#secret", False),
    ],
)
def test_http_url_security(tmp_path: Path, url: str, valid: bool) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, f'mcp_servers:\n  web:\n    type: http\n    url: "{url}"')
    snapshot = repository.load()
    assert bool(snapshot.servers) is valid


def test_http_header_expansion(tmp_path: Path) -> None:
    repository, paths = _repository(tmp_path)
    _write(paths.user, 'mcp_servers:\n  web:\n    type: http\n    url: https://example.test/mcp\n    headers: {Authorization: "Bearer ${TOKEN}"}')
    resolved, diagnostic = resolve_server(repository.load().servers[0], {"TOKEN": "abc"})
    assert isinstance(resolved, ResolvedHttpServerConfig)
    assert diagnostic is None
    assert resolved.headers["Authorization"].value == "Bearer abc"
