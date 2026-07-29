from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from fakuicode.hooks.config import HookConfigRepository, HookPaths
from fakuicode.hooks.models import HookEvent
from fakuicode.hooks.runtime import HookEngine
from fakuicode.models import ToolCall
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


def _engine(tmp_path: Path, yaml_text: str, **kwargs: object) -> HookEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = HookPaths.for_workspace(workspace, home=tmp_path / "home")
    paths.user.parent.mkdir(parents=True)
    paths.user.write_text(yaml_text, encoding="utf-8")
    rules = HookConfigRepository(paths, workspace).load().rules
    return HookEngine(rules, workspace=workspace, **kwargs)


def test_prompt_action_injects_static_content_and_once_is_process_local(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - name: conventions
    event: turn_start
    action: {type: prompt, content: Follow repository conventions., once: true}
""",
    )

    first = engine.dispatch(HookEvent.TURN_START, {"outcome": "started"})
    second = engine.dispatch(HookEvent.TURN_START, {"outcome": "started"})

    assert first.prompts == ("Follow repository conventions.",)
    assert second.prompts == ()
    assert engine.consume_prompts() == ("Follow repository conventions.",)
    assert engine.consume_prompts() == ("Follow repository conventions.",)
    engine.dispatch(HookEvent.TURN_END, {})
    assert engine.consume_prompts() == ()


def test_start_event_prompts_remain_for_their_lifecycle_scope(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - {event: session_start, action: {type: prompt, content: Session context.}}
  - {event: turn_start, action: {type: prompt, content: Turn context.}}
""",
    )

    engine.dispatch(HookEvent.SESSION_START, {})
    engine.dispatch(HookEvent.TURN_START, {})

    assert engine.consume_prompts() == ("Session context.", "Turn context.")
    assert engine.consume_prompts() == ("Session context.", "Turn context.")

    engine.dispatch(HookEvent.TURN_END, {})
    assert engine.consume_prompts() == ("Session context.",)

    engine.dispatch(HookEvent.SESSION_END, {})
    assert engine.consume_prompts() == ()


def test_command_exit_two_denies_without_becoming_a_hook_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - name: protect-main
    event: pre_tool_use
    action: {type: command, command: check-policy}
""",
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 2, "", "Protected branch")

    monkeypatch.setattr("fakuicode.hooks.runtime.subprocess.run", fake_run)

    result = engine.dispatch(HookEvent.PRE_TOOL_USE, {"tool": {"name": "write_file"}})

    assert result.denied_reason == "Protected branch"
    assert result.diagnostics == ()


def test_nonzero_command_failure_is_sanitized_and_never_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - name: flaky
    event: pre_tool_use
    action: {type: command, command: flaky-command}
""",
        diagnostic_sink=captured.append,
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 9, "secret stdout", "token=secret")

    monkeypatch.setattr("fakuicode.hooks.runtime.subprocess.run", fake_run)

    result = engine.dispatch(HookEvent.PRE_TOOL_USE, {"tool": {"arguments": {"password": "secret"}}})

    assert result.denied_reason is None
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.category == "command_exit"
    assert diagnostic.status == 9
    assert "secret" not in repr(diagnostic)
    assert captured == [diagnostic]


def test_structured_decisions_are_aggregated_with_deny_winning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - name: allow-audit
    event: pre_tool_use
    action: {type: command, command: allow}
  - name: block-secret
    event: pre_tool_use
    action: {type: command, command: deny}
""",
    )

    def fake_run(command: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        output = '{"decision":"allow"}' if command == "allow" else '{"decision":"deny","reason":"Secret path"}'
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("fakuicode.hooks.runtime.subprocess.run", fake_run)

    result = engine.dispatch(HookEvent.PRE_TOOL_USE, {"tool": {"name": "read_file"}})

    assert result.denied_reason == "Secret path"


def test_http_sends_only_safe_metadata_plus_explicit_includes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - name: notify
    event: post_tool_use
    action:
      type: http
      url: https://hooks.example.test/event
      include: [/tool/name, /tool/outcome]
""",
    )
    sent: dict[str, object] = {}

    class StreamResponse:
        status_code = 204
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def iter_bytes(self):
            return iter(())

    def fake_stream(method: str, url: str, **kwargs: object) -> StreamResponse:
        sent["method"] = method
        sent.update(url=url, **kwargs)
        return StreamResponse()

    monkeypatch.setattr("fakuicode.hooks.runtime.httpx.stream", fake_stream)

    engine.dispatch(
        HookEvent.POST_TOOL_USE,
        {"tool": {"name": "run_command", "outcome": "ok", "arguments": {"token": "secret"}}},
    )

    assert sent["follow_redirects"] is False
    assert sent["method"] == "POST"
    assert sent["json"] == {
        "event": "post_tool_use",
        "hook": "notify",
        "source": "user",
        "included": {"/tool/name": "run_command", "/tool/outcome": "ok"},
    }


def test_http_response_over_limit_is_a_nonblocking_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - {name: oversized, event: turn_end, action: {type: http, url: https://example.test}}
""",
    )

    class StreamResponse:
        status_code = 200
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def iter_bytes(self):
            yield b"x" * (32 * 1024 + 1)

    monkeypatch.setattr(
        "fakuicode.hooks.runtime.httpx.stream",
        lambda *args, **kwargs: StreamResponse(),
    )

    result = engine.dispatch(HookEvent.TURN_END, {})

    assert result.denied_reason is None
    assert result.diagnostics[0].category == "response_limit"


def test_plan_mode_runs_only_static_prompt_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - event: turn_start
    action: {type: prompt, content: Read only.}
  - event: turn_start
    action: {type: command, command: must-not-run}
""",
    )
    monkeypatch.setattr(
        "fakuicode.hooks.runtime.subprocess.run",
        lambda *args, **kwargs: pytest.fail("command ran in plan mode"),
    )

    result = engine.dispatch(HookEvent.TURN_START, {}, plan_mode=True)

    assert result.prompts == ("Read only.",)
    assert result.diagnostics == ()


def test_pre_tool_denial_returns_a_tool_result_before_permission_or_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(
        tmp_path,
        """version: 1
hooks:
  - event: pre_tool_use
    if:
      all:
        - {field: /tool/name, exact: write_file}
        - {field: /tool/arguments/path, glob: "protected/**"}
    action: {type: command, command: policy-check}
""",
    )
    monkeypatch.setattr(
        "fakuicode.hooks.runtime.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 2, "", "Protected by Hook"),
    )
    registry = ToolRegistry(WorkspacePolicy(tmp_path / "workspace"), hook_engine=engine)

    result = registry.execute(
        ToolCall("call-1", "write_file", {"path": "protected/value.txt", "content": "x"})
    )

    assert result.success is False
    assert result.summary == "Hook 已拒绝工具执行"
    assert result.output == "Hook 拒绝：Protected by Hook"
    assert not (tmp_path / "workspace" / "protected" / "value.txt").exists()
