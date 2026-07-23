from __future__ import annotations

from pathlib import Path

import pytest

from fakuicode.hooks.config import HookConfigRepository, HookPaths
from fakuicode.hooks.models import (
    CommandAction,
    HookEvent,
    HookSource,
    HttpAction,
    PromptAction,
)


def _paths(tmp_path: Path) -> tuple[Path, HookPaths]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace, HookPaths.for_workspace(workspace, home=tmp_path / "home")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_missing_files_load_an_empty_snapshot(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)

    snapshot = HookConfigRepository(paths, workspace).load()

    assert snapshot.rules == ()
    assert snapshot.diagnostics == ()
    assert snapshot.project_fingerprint is None


def test_loads_strict_rules_and_matches_flat_conditions(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        """version: 1
hooks:
  - name: format-python
    event: post_tool_use
    if:
      all:
        - field: /tool/name
          exact: write_file
        - field: /tool/arguments/path
          glob: "**/*.py"
        - field: /tool/outcome
          exact: failed
          not: true
    action:
      type: command
      command: python -m ruff format
      command_windows: py -m ruff format
      timeout_seconds: 20
      once: true
""",
    )

    snapshot = HookConfigRepository(paths, workspace).load()

    assert snapshot.diagnostics == ()
    rule = snapshot.rules[0]
    assert rule.name == "format-python"
    assert rule.event is HookEvent.POST_TOOL_USE
    assert rule.source is HookSource.USER
    assert isinstance(rule.action, CommandAction)
    assert rule.action.timeout_seconds == 20
    assert rule.action.once is True
    assert rule.condition is not None
    assert rule.condition.matches(
        {"tool": {"name": "write_file", "arguments": {"path": "src/main.py"}, "outcome": "ok"}}
    )
    assert not rule.condition.matches(
        {"tool": {"name": "write_file", "arguments": {"path": "README.md"}, "outcome": "ok"}}
    )


def test_any_condition_and_regex_use_full_match(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        """version: 1
hooks:
  - event: pre_tool_use
    if:
      any:
        - field: /tool/arguments/command
          regex: "(?:rm|del) .+"
        - field: /tool/name
          exact: dangerous_tool
    action:
      type: prompt
      content: Reconsider this operation.
""",
    )

    condition = HookConfigRepository(paths, workspace).load().rules[0].condition

    assert condition is not None
    assert condition.matches({"tool": {"name": "run_command", "arguments": {"command": "rm old.txt"}}})
    assert not condition.matches({"tool": {"name": "run_command", "arguments": {"command": "echo rm old.txt"}}})


def test_missing_field_never_matches_even_when_negated(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        """version: 1
hooks:
  - event: turn_start
    if:
      all:
        - field: /metadata/branch
          exact: main
          not: true
    action:
      type: prompt
      content: Branch policy.
""",
    )

    condition = HookConfigRepository(paths, workspace).load().rules[0].condition

    assert condition is not None
    assert not condition.matches({})


def test_http_action_requires_safe_url_and_explicit_env_allowlist(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        """version: 1
hooks:
  - event: turn_end
    action:
      type: http
      url: https://hooks.example.test/events
      headers:
        Authorization: "Bearer ${HOOK_TOKEN}"
      allowed_env_vars: [HOOK_TOKEN]
      include: [/event, /outcome]
      async: true
""",
    )

    action = HookConfigRepository(paths, workspace).load().rules[0].action

    assert isinstance(action, HttpAction)
    assert action.include == ("/event", "/outcome")
    assert action.async_ is True


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\nversion: 1\nhooks: []\n",
        "version: 2\nhooks: []\n",
        "version: 1\nunknown: true\nhooks: []\n",
        "version: 1\nhooks:\n  - event: nope\n    action: {type: prompt, content: hi}\n",
        "version: 1\nhooks:\n  - event: turn_start\n    if: {all: [], any: []}\n    action: {type: prompt, content: hi}\n",
        "version: 1\nhooks:\n  - event: turn_start\n    if:\n      all:\n        - field: /x\n          exact: a\n          glob: a*\n    action: {type: prompt, content: hi}\n",
        "version: 1\nhooks:\n  - event: pre_tool_use\n    action: {type: command, command: echo hi, async: true}\n",
        "version: 1\nhooks:\n  - event: turn_start\n    action: {type: http, url: http://example.com}\n",
        "version: 1\nhooks:\n  - {name: duplicate, event: turn_start, action: {type: prompt, content: one}}\n  - {name: duplicate, event: turn_end, action: {type: prompt, content: two}}\n",
        "version: 1\nhooks:\n  - event: turn_start\n    action:\n      type: http\n      url: https://example.com\n      headers: {Authorization: '${SECRET}'}\n",
        "version: 1\nhooks:\n  - event: turn_start\n    action: {type: http, url: https://example.com, headers: {Bad Header: value}}\n",
        "version: 1\nhooks:\n  - event: turn_start\n    action: {type: http, url: https://example.com, headers: {X-Test: '${lowercase}'}}\n",
        "version: [\n",
    ],
)
def test_invalid_user_source_is_disabled_with_sanitized_diagnostic(
    tmp_path: Path, content: str
) -> None:
    workspace, paths = _paths(tmp_path)
    _write(paths.user, content)

    snapshot = HookConfigRepository(paths, workspace).load()

    assert snapshot.rules == ()
    assert len(snapshot.diagnostics) == 1
    assert content not in snapshot.diagnostics[0]


def test_invalid_project_source_does_not_disable_valid_user_hooks(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        "version: 1\nhooks:\n  - event: turn_start\n    action: {type: prompt, content: user}\n",
    )
    _write(paths.project, "version: 1\nhooks: nope\n")

    snapshot = HookConfigRepository(paths, workspace).load()

    assert len(snapshot.rules) == 1
    assert isinstance(snapshot.rules[0].action, PromptAction)
    assert snapshot.rules[0].source is HookSource.USER
    assert snapshot.project_fingerprint is not None
    assert len(snapshot.diagnostics) == 1


def test_hook_glob_uses_permission_style_escaping(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    _write(
        paths.user,
        """version: 1
hooks:
  - event: turn_start
    if:
      all:
        - {field: /value, glob: 'literal\\*'}
    action: {type: prompt, content: matched}
""",
    )

    condition = HookConfigRepository(paths, workspace).load().rules[0].condition

    assert condition is not None
    assert condition.matches({"value": "literal*"})
    assert not condition.matches({"value": "literal-value"})


def test_project_hook_symlink_cannot_supply_rules_or_a_trust_fingerprint(tmp_path: Path) -> None:
    workspace, paths = _paths(tmp_path)
    external = tmp_path / "outside-hooks.yaml"
    _write(
        external,
        "version: 1\nhooks:\n  - {event: turn_start, action: {type: command, command: outside}}\n",
    )
    paths.project.parent.mkdir(parents=True)
    try:
        paths.project.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    snapshot = HookConfigRepository(paths, workspace, project_trusted=True).load()

    assert snapshot.rules == ()
    assert snapshot.project_rules == ()
    assert snapshot.project_fingerprint is None
    assert snapshot.diagnostics
