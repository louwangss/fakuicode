from __future__ import annotations

from pathlib import Path


def test_registry_exposes_six_stable_schemas_and_dispatches_calls(tmp_path: Path) -> None:
    from fakuicode.models import ToolCall
    from fakuicode.permissions.config import PermissionConfigSnapshot
    from fakuicode.permissions.manager import PermissionManager
    from fakuicode.permissions.models import ApprovalChoice
    from fakuicode.permissions.safety import DangerousCommandGuard
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    class AllowOnce:
        def request(self, request, *, cancel_event=None):
            del request, cancel_event
            return ApprovalChoice.ONCE

    permissions = PermissionManager(
        PermissionConfigSnapshot(), DangerousCommandGuard(tmp_path), approval_handler=AllowOnce()
    )
    registry = ToolRegistry(WorkspacePolicy(tmp_path), permission_manager=permissions)
    definitions = registry.definitions()
    result = registry.execute(ToolCall("call-1", "write_file", {"path": "answer.txt", "content": "done"}))

    assert [definition.name for definition in definitions] == [
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "find_files",
        "search_code",
    ]
    assert all(definition.input_schema["type"] == "object" for definition in definitions)
    assert result.call_id == "call-1"
    assert result.success is True
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "done"


def test_registry_returns_safe_failure_for_unknown_invalid_and_crashing_calls(tmp_path: Path) -> None:
    from fakuicode.models import ToolCall
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    registry = ToolRegistry(WorkspacePolicy(tmp_path))
    unknown = registry.execute(ToolCall("call-2", "delete_everything", {}))
    invalid = registry.execute(ToolCall("call-3", "read_file", {}))

    assert unknown.success is False
    assert invalid.success is False
    assert "Unknown" in unknown.output
    assert "requires" in invalid.output


def test_registry_never_executes_a_side_effect_without_permission(tmp_path: Path) -> None:
    from fakuicode.models import ToolCall
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    registry = ToolRegistry(WorkspacePolicy(tmp_path))

    result = registry.execute(ToolCall("call-denied", "write_file", {"path": "denied.txt", "content": "no"}))

    assert result.success is False
    assert result.summary == "permission denied"
    assert not (tmp_path / "denied.txt").exists()


def test_registry_explains_hard_shell_wrapper_denials(tmp_path: Path) -> None:
    from fakuicode.models import ToolCall
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    registry = ToolRegistry(WorkspacePolicy(tmp_path))

    result = registry.execute(
        ToolCall("call-shell", "run_command", {"command": ["cmd", "/c", "mkdir", "nested"]})
    )

    assert result.success is False
    assert "general shell" in result.output
    assert "general shell" in result.summary


def test_registry_enforces_plan_boundary_even_with_an_explicit_allow(tmp_path: Path) -> None:
    from fakuicode.models import ToolCall
    from fakuicode.permissions.config import PermissionConfigSnapshot
    from fakuicode.permissions.manager import PermissionManager
    from fakuicode.permissions.models import RuleEffect, RuleSource
    from fakuicode.permissions.rules import parse_rule
    from fakuicode.permissions.safety import DangerousCommandGuard
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    allow = parse_rule("write_file(*)", RuleEffect.ALLOW, RuleSource.USER)
    permissions = PermissionManager(
        PermissionConfigSnapshot(user_rules=(allow,)), DangerousCommandGuard(tmp_path)
    )
    registry = ToolRegistry(WorkspacePolicy(tmp_path), permission_manager=permissions)

    result = registry.execute(
        ToolCall("call-plan", "write_file", {"path": "plan.txt", "content": "no"}),
        read_only_only=True,
    )

    assert result.success is False
    assert not (tmp_path / "plan.txt").exists()


def test_registry_classifies_read_only_tools_without_exposing_writes(tmp_path: Path) -> None:
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    registry = ToolRegistry(WorkspacePolicy(tmp_path))

    assert [definition.name for definition in registry.definitions(read_only_only=True)] == [
        "read_file",
        "find_files",
        "search_code",
    ]
    assert registry.is_known("read_file") is True
    assert registry.is_known("not_a_tool") is False
    assert registry.is_read_only("read_file") is True
    assert registry.is_read_only("write_file") is False


def test_registry_accepts_finish_turn_directive_only_from_system_tools(
    tmp_path: Path,
) -> None:
    from fakuicode.models import ToolCall, ToolDefinition
    from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    class DeferredTool:
        def __init__(self, name: str) -> None:
            self._name = name

        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                self._name,
                "deferred work",
                {"type": "object", "properties": {}},
            )

        @property
        def read_only(self) -> bool:
            return True

        def prepare(self, arguments):
            return ToolPreparation(freeze_arguments(arguments), self._name)

        def execute_prepared(self, arguments, *, cancel_event=None):
            del arguments, cancel_event
            return ToolExecution(
                True,
                "queued",
                "queued",
                metadata={
                    "finish_agent_turn": True,
                    "finish_agent_turn_message": "后台任务已启动。",
                },
            )

    registry = ToolRegistry(WorkspacePolicy(tmp_path), tools=())
    registry.register_system(DeferredTool("system_deferred"))
    registry.register(DeferredTool("ordinary_deferred"))

    system_result = registry.execute(ToolCall("call-system", "system_deferred", {}))
    ordinary_result = registry.execute(ToolCall("call-ordinary", "ordinary_deferred", {}))

    assert registry.finish_turn_message((system_result,)) == "后台任务已启动。"
    assert registry.finish_turn_message((ordinary_result,)) is None
