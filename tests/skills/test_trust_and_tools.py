from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import yaml

from fakuicode.models import ToolCall
from fakuicode.permissions.config import PermissionConfigSnapshot
from fakuicode.permissions.manager import PermissionManager
from fakuicode.permissions.models import ApprovalChoice
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.skills import SkillDiscovery, SkillManager
from fakuicode.skills.parser import fingerprint_upstream
from fakuicode.skills.trust import SkillTrustRepository, skill_identity
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


class _AllowOnce:
    def request(self, request, *, cancel_event=None):
        del request, cancel_event
        return ApprovalChoice.ONCE


def _script_skill(root: Path, *, script: str) -> Path:
    package = root / "scripted"
    (package / "tools").mkdir(parents=True)
    (package / "scripts").mkdir()
    (package / "SKILL.md").write_text(
        "---\n"
        "name: scripted\n"
        "description: scripted workflow\n"
        "fakuicode:\n"
        "  invocation: auto\n"
        "  execution: shared\n"
        "---\n"
        "Use the dedicated tool.\n",
        encoding="utf-8",
    )
    (package / "tools" / "echo.json").write_text(
        json.dumps(
            {
                "name": "echo",
                "description": "Echo structured input",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                "entrypoint": "scripts/echo.py",
            }
        ),
        encoding="utf-8",
    )
    (package / "scripts" / "echo.py").write_text(script, encoding="utf-8")
    return package


def _runtime(tmp_path: Path, *, script: str, trust_handler=None):
    skill_root = tmp_path / ".fakuicode" / "skills"
    _script_skill(skill_root, script=script)
    permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
        approval_handler=_AllowOnce(),
    )
    registry = ToolRegistry(WorkspacePolicy(tmp_path), permission_manager=permissions)
    trust = SkillTrustRepository(tmp_path / "skill-trust.yaml")
    manager = SkillManager(
        SkillDiscovery(skill_root, tmp_path / "user", tmp_path / "builtin"),
        registry,
        context_window=128_000,
        trust_repository=trust,
        trust_handler=trust_handler,
    )
    manager.refresh()
    return manager, registry, trust


def test_project_script_requires_fingerprint_trust_before_registration(tmp_path: Path) -> None:
    requests = []
    manager, registry, trust = _runtime(
        tmp_path,
        script="import json,sys\nargs=json.load(sys.stdin)\nprint(json.dumps({'output': args['value'], 'summary': 'ok'}))\n",
        trust_handler=lambda request: requests.append(request) or True,
    )

    result = manager.invoke("scripted", "")
    definition_names = {item.name for item in registry.definitions()}
    identity = skill_identity(tmp_path, manager.snapshot.skills["scripted"])

    assert result.success is True
    assert requests and requests[0].fingerprint == identity.fingerprint
    assert trust.is_trusted(identity)
    assert "skill__scripted__echo" in definition_names


def test_project_script_rejected_trust_is_not_registered(tmp_path: Path) -> None:
    manager, registry, _ = _runtime(tmp_path, script="print('x')", trust_handler=lambda request: False)

    result = manager.invoke("scripted", "")

    assert result.success is False
    assert "skill__scripted__echo" not in registry.all_names()


def test_remote_installed_user_script_is_not_pretrusted_by_its_target_scope(tmp_path: Path) -> None:
    user_root = tmp_path / "user"
    package = _script_skill(user_root, script="print('{}')\n")
    receipt_root = package / ".fakuicode"
    receipt_root.mkdir()
    receipt = {
        "schema-version": 1,
        "requested-url": "https://github.com/acme/skills",
        "source-url": "https://github.com/acme/skills",
        "revision": "a" * 40,
        "skill-path": "skills/scripted",
        "upstream-fingerprint": fingerprint_upstream(package),
        "fakuicode": {
            "invocation": "auto",
            "visible-tools": [],
            "execution": "shared",
            "history-turns": 0,
            "profile": "inherit",
        },
    }
    (receipt_root / "install.yaml").write_text(
        yaml.safe_dump(receipt, sort_keys=False),
        encoding="utf-8",
    )
    requests = []
    registry = ToolRegistry(WorkspacePolicy(tmp_path))
    manager = SkillManager(
        SkillDiscovery(tmp_path / "project", user_root, tmp_path / "builtin"),
        registry,
        context_window=128_000,
        trust_repository=SkillTrustRepository(tmp_path / "trust.yaml"),
        trust_handler=lambda request: requests.append(request) or False,
    )
    manager.refresh()

    result = manager.invoke("scripted", "")

    assert result.success is False
    assert requests
    assert "skill__scripted__echo" not in registry.all_names()


def test_project_script_registration_rolls_back_when_activation_event_fails(tmp_path: Path) -> None:
    manager, registry, _ = _runtime(
        tmp_path,
        script="print('{\"output\":\"ok\",\"summary\":\"ok\"}')\n",
        trust_handler=lambda request: True,
    )
    manager.on_activation = lambda active: (_ for _ in ()).throw(OSError("database unavailable"))

    result = manager.invoke("scripted", "")

    assert result.success is False
    assert manager.active == ()
    assert "skill__scripted__echo" not in registry.all_names()


def test_script_tool_uses_json_protocol_and_fingerprint_permission_target(tmp_path: Path) -> None:
    manager, registry, _ = _runtime(
        tmp_path,
        script="import json,sys\nargs=json.load(sys.stdin)\nprint(json.dumps({'output': args['value'], 'summary': 'echoed'}))\n",
        trust_handler=lambda request: True,
    )
    assert manager.invoke("scripted", "").success

    result = registry.execute(ToolCall("1", "skill__scripted__echo", {"value": "hello"}))

    assert result.success is True
    assert result.output == "hello"
    assert result.summary == "echoed"


def test_activated_script_refuses_package_changes_before_the_next_refresh(tmp_path: Path) -> None:
    manager, registry, _ = _runtime(
        tmp_path,
        script="import json\nprint(json.dumps({'output': 'old', 'summary': 'old'}))\n",
        trust_handler=lambda request: True,
    )
    assert manager.invoke("scripted", "").success
    script = manager.snapshot.skills["scripted"].tools[0].entrypoint
    script.write_text(
        "from pathlib import Path\nPath('unexpected.txt').write_text('ran')\n"
        "print('{\"output\":\"new\",\"summary\":\"new\"}')\n",
        encoding="utf-8",
    )

    result = registry.execute(ToolCall("changed", "skill__scripted__echo", {"value": "x"}))

    assert result.success is False
    assert "changed after activation" in result.output
    assert not (tmp_path / "unexpected.txt").exists()


def test_script_tool_invalid_json_is_normal_failure_and_plan_never_runs_it(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    manager, registry, _ = _runtime(
        tmp_path,
        script=f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\nprint('not-json')\n",
        trust_handler=lambda request: True,
    )
    assert manager.invoke("scripted", "").success

    blocked = registry.execute(
        ToolCall("plan", "skill__scripted__echo", {"value": "x"}),
        read_only_only=True,
    )
    assert blocked.success is False
    assert not marker.exists()

    failed = registry.execute(ToolCall("run", "skill__scripted__echo", {"value": "x"}))
    assert failed.success is False
    assert marker.exists()


def test_corrupt_skill_trust_store_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "trust.yaml"
    path.write_text("version: [", encoding="utf-8")
    manager, _, trust = _runtime(
        tmp_path,
        script="print('x')",
        trust_handler=lambda request: True,
    )
    # Point the runtime at the corrupt repository after construction.
    manager.trust_repository = SkillTrustRepository(path)

    result = manager.invoke("scripted", "")

    assert result.success is False
    assert manager.trust_repository.diagnostic is not None


def test_script_tool_cancellation_and_timeout_are_bounded_failures(tmp_path: Path) -> None:
    from fakuicode.skills.tool import SkillScriptTool

    manager, _, _ = _runtime(
        tmp_path,
        script="import time\ntime.sleep(5)\nprint('{}')\n",
        trust_handler=lambda request: True,
    )
    skill = manager.snapshot.skills["scripted"]
    cancelled_tool = SkillScriptTool(tmp_path, skill, skill.tools[0], timeout_seconds=1)
    cancelled = Event()
    cancelled.set()

    cancelled_result = cancelled_tool.execute({"value": "x"}, cancel_event=cancelled)

    assert cancelled_result.success is False
    assert "cancelled" in cancelled_result.summary

    timeout_permissions = PermissionManager(
        PermissionConfigSnapshot(),
        DangerousCommandGuard(tmp_path),
        approval_handler=_AllowOnce(),
    )
    timeout_registry = ToolRegistry(
        WorkspacePolicy(tmp_path),
        permission_manager=timeout_permissions,
    )
    timeout_registry.register(SkillScriptTool(tmp_path, skill, skill.tools[0], timeout_seconds=0.01))
    timed_out = timeout_registry.execute(
        ToolCall("timeout", "skill__scripted__echo", {"value": "x"})
    )
    assert timed_out.success is False
