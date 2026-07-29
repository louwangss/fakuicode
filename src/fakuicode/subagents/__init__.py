"""Isolated child-agent definitions and runtime support."""

from fakuicode.subagents.catalog import AgentCatalog, CatalogLoadError
from fakuicode.subagents.models import (
    AgentDefinition,
    AgentSource,
    CatalogDiagnostic,
    PermissionBehavior,
)
from fakuicode.subagents.runtime import (
    ChildAgentSession,
    ChildRunResult,
    ChildRuntimeError,
    ChildRuntimeFactory,
    run_controller_to_completion,
)
from fakuicode.subagents.tasks import (
    TaskManager,
    TaskManagerCloseReport,
    TaskManagerError,
    TaskSnapshot,
)

__all__ = [
    "AgentCatalog",
    "AgentDefinition",
    "AgentSource",
    "CatalogDiagnostic",
    "CatalogLoadError",
    "ChildAgentSession",
    "ChildRunResult",
    "ChildRuntimeError",
    "ChildRuntimeFactory",
    "run_controller_to_completion",
    "PermissionBehavior",
    "TaskManager",
    "TaskManagerCloseReport",
    "TaskManagerError",
    "TaskSnapshot",
]
