from pathlib import Path

from fakuicode.teams.coordinator import apply_coordinator_scope
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


def test_coordinator_hides_writers_shell_and_unrelated_system_tools(
    tmp_path: Path,
) -> None:
    from fakuicode.models import ToolDefinition
    from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments

    class ControlTool:
        read_only = False

        def __init__(self, name: str) -> None:
            self.definition = ToolDefinition(
                name,
                "control",
                {"type": "object", "properties": {}, "additionalProperties": False},
            )

        def prepare(self, arguments):
            return ToolPreparation(freeze_arguments(arguments), "team:test")

        def execute_prepared(self, arguments, *, cancel_event=None):
            del arguments, cancel_event
            return ToolExecution(True, "ok", "ok")

        def execute(self, arguments, *, cancel_event=None):
            return self.execute_prepared(arguments, cancel_event=cancel_event)

    registry = ToolRegistry(WorkspacePolicy(tmp_path))
    registry.register_system(ControlTool("agent"))
    registry.register(ControlTool("team_member_start"))
    registry.register(ControlTool("team_integrate_task"))

    apply_coordinator_scope(
        registry,
        {"team_member_start", "team_integrate_task"},
    )

    names = {definition.name for definition in registry.definitions()}
    assert names == {
        "read_file",
        "find_files",
        "search_code",
        "team_member_start",
        "team_integrate_task",
    }
    assert "run_command" not in names
    assert "write_file" not in names
    assert "edit_file" not in names
    assert "agent" not in names

    registry.set_visible_tools(registry.all_names())
    names_after_rescope = {definition.name for definition in registry.definitions()}
    assert names_after_rescope == names
