"""Tool registration, definition export, and safe execution dispatch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Event, RLock

from fakuicode.errors import (
    RequestCancelled,
    ToolExecutionError,
    ToolOutputStorageError,
    ToolPolicyError,
)
from fakuicode.context_artifacts import ContextArtifactStore
from fakuicode.hooks.models import HookEvent
from fakuicode.hooks.runtime import HookEngine
from fakuicode.models import ToolCall, ToolDefinition, ToolResult
from fakuicode.permissions.config import PermissionConfigSnapshot
from fakuicode.permissions.manager import PermissionManager
from fakuicode.permissions.models import DecisionKind, PermissionMode
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.tools.base import (
    FINISH_AGENT_TURN,
    FINISH_AGENT_TURN_MESSAGE,
    PreparedToolCall,
    Tool,
)
from fakuicode.tools.command import RunCommandTool
from fakuicode.tools.filesystem import EditFileTool, FindFilesTool, ReadFileTool, SearchCodeTool, WriteFileTool
from fakuicode.tools.policy import WorkspacePolicy


class ToolRegistry:
    def __init__(
        self,
        policy: WorkspacePolicy,
        tools: Iterable[Tool] | None = None,
        *,
        permission_manager: PermissionManager | None = None,
        owns_permission_manager: bool = True,
        hook_engine: HookEngine | None = None,
    ) -> None:
        self.policy = policy
        self.permission_manager = permission_manager or PermissionManager(
            PermissionConfigSnapshot(mode=PermissionMode.STRICT),
            DangerousCommandGuard(policy.workspace),
        )
        self._owns_permission_manager = owns_permission_manager
        self.hook_engine = hook_engine
        self._tools: dict[str, Tool] = {}
        self._optional_names: set[str] = set()
        self._system_names: set[str] = set()
        self._visible_names: set[str] | None = None
        self._include_system_tools = True
        self._tool_ceiling: set[str] | None = None
        self._ceiling_includes_system_tools = True
        self._context_artifact_store: ContextArtifactStore | None = None
        self._lock = RLock()
        for tool in tools or _default_tools(policy):
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        with self._lock:
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered.")
            self._tools[name] = tool
            self._bind_artifact_store(tool)

    def set_context_artifact_store(
        self,
        artifact_store: ContextArtifactStore | None,
    ) -> None:
        """Bind one session-owned artifact store to tools that support staged output."""

        with self._lock:
            self._context_artifact_store = artifact_store
            for tool in self._tools.values():
                self._bind_artifact_store(tool)

    def _bind_artifact_store(self, tool: Tool) -> None:
        bind = getattr(tool, "set_artifact_store", None)
        if callable(bind):
            bind(self._context_artifact_store)

    def register_system(self, tool: Tool) -> None:
        """Register a host control tool that is always model-visible and bypasses ordinary approval."""
        self.register(tool)
        with self._lock:
            self._system_names.add(tool.definition.name)

    def unregister(self, name: str) -> None:
        with self._lock:
            if name in self._system_names:
                raise ValueError(f"System tool '{name}' cannot be unregistered.")
            self._tools.pop(name, None)
            self._optional_names.discard(name)

    def all_names(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._tools)

    def set_visible_tools(
        self,
        names: Iterable[str] | None,
        *,
        include_system_tools: bool = True,
    ) -> None:
        """Restrict provider-visible and executable tools.

        System tools remain visible by default. Restricted host modes may opt out,
        then explicitly allow only the control tools required by that mode.
        """
        with self._lock:
            self._visible_names = None if names is None else set(names)
            self._include_system_tools = include_system_tools

    def set_tool_ceiling(
        self,
        names: Iterable[str] | None,
        *,
        include_system_tools: bool = True,
    ) -> None:
        """Set a host-owned ceiling that later scopes cannot broaden."""

        with self._lock:
            self._tool_ceiling = None if names is None else set(names)
            self._ceiling_includes_system_tools = include_system_tools

    def replace_optional(self, name: str, tool: Tool | None) -> None:
        """Atomically replace one host-owned optional tool without touching defaults or MCP tools."""
        with self._lock:
            if name in self._tools and name not in self._optional_names:
                raise ValueError(f"Tool '{name}' is not host-optional.")
            if tool is None:
                self._tools.pop(name, None)
                self._optional_names.discard(name)
                return
            if tool.definition.name != name:
                raise ValueError("Optional tool name does not match its definition.")
            self._tools[name] = tool
            self._optional_names.add(name)

    def definitions(self, *, read_only_only: bool = False) -> list[ToolDefinition]:
        with self._lock:
            tools = tuple(
                tool
                for name, tool in self._tools.items()
                if self._is_visible_locked(name)
            )
        return [_reinforced_definition(tool.definition) for tool in tools if not read_only_only or tool.read_only]

    def is_known(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def is_read_only(self, name: str) -> bool:
        with self._lock:
            tool = self._tools.get(name)
        return tool.read_only if tool is not None else False

    def execute(
        self,
        call: ToolCall,
        *,
        cancel_event: Event | None = None,
        read_only_only: bool = False,
    ) -> ToolResult:
        prepared: PreparedToolCall | None = None
        try:
            with self._lock:
                tool = self._tools.get(call.name)
                visible = self._is_visible_locked(call.name)
                system_tool = call.name in self._system_names
            if tool is None:
                raise ToolExecutionError(f"Unknown tool '{call.name}'.")
            if not visible:
                raise ToolExecutionError(f"Tool '{call.name}' is not visible in the active Skill scope.")
            preparation = tool.prepare(call.arguments)
            prepared = PreparedToolCall(
                call.id,
                call.name,
                preparation.arguments,
                preparation.target,
                tool.read_only,
                preparation.permission_scope,
                preparation.permission_capability,
            )
            if self.hook_engine is not None:
                hook_result = self.hook_engine.dispatch(
                    HookEvent.PRE_TOOL_USE,
                    {"tool": _hook_tool_payload(prepared, self.policy)},
                    plan_mode=read_only_only,
                )
                if hook_result.denied_reason is not None:
                    result = ToolResult(
                        call.id,
                        call.name,
                        False,
                        f"Hook 拒绝：{hook_result.denied_reason}",
                        "Hook 已拒绝工具执行",
                    )
                    self._dispatch_post_hook(prepared, result, read_only_only, outcome="denied")
                    return result
            if not system_tool:
                decision = self.permission_manager.authorize(
                    prepared,
                    read_only_task=read_only_only,
                    cancel_event=cancel_event,
                )
                if decision.kind is not DecisionKind.ALLOW:
                    summary = "permission denied"
                    if decision.layer == "dangerous_command":
                        summary = f"permission denied: {decision.reason}"
                    result = ToolResult(
                        call.id,
                        call.name,
                        False,
                        f"Permission denied: {decision.reason}",
                        summary,
                    )
                    self._dispatch_post_hook(prepared, result, read_only_only, outcome="denied")
                    return result
            execution = tool.execute_prepared(prepared.arguments, cancel_event=cancel_event)
        except RequestCancelled:
            if prepared is not None:
                result = ToolResult(call.id, call.name, False, "Tool execution was cancelled.", "tool action cancelled")
                self._dispatch_post_hook(prepared, result, read_only_only, outcome="cancelled")
            raise
        except ToolOutputStorageError:
            raise
        except (ToolExecutionError, ToolPolicyError) as error:
            result = ToolResult(call.id, call.name, False, str(error), "tool action was rejected or failed")
            if prepared is not None:
                self._dispatch_post_hook(prepared, result, read_only_only, outcome="failed")
            return result
        except Exception:
            result = ToolResult(call.id, call.name, False, "Tool execution failed.", "tool action failed")
            if prepared is not None:
                self._dispatch_post_hook(prepared, result, read_only_only, outcome="failed")
            return result
        result = ToolResult(
            call.id,
            call.name,
            execution.success,
            execution.output,
            execution.summary,
            execution.duration_seconds,
            execution.metadata,
        )
        self._dispatch_post_hook(
            prepared,
            result,
            read_only_only,
            outcome="ok" if result.success else "failed",
        )
        return result

    def _dispatch_post_hook(
        self,
        prepared: PreparedToolCall,
        result: ToolResult,
        read_only_only: bool,
        *,
        outcome: str,
    ) -> None:
        if self.hook_engine is None:
            return
        self.hook_engine.dispatch(
            HookEvent.POST_TOOL_USE,
            {
                "tool": {
                    **_hook_tool_payload(prepared, self.policy),
                    "outcome": outcome,
                    "summary": result.summary,
                    "duration_seconds": result.duration_seconds,
                }
            },
            plan_mode=read_only_only,
        )

    def begin_request(self) -> None:
        self.permission_manager.begin_request()

    def finish_turn_message(self, results: tuple[ToolResult, ...]) -> str | None:
        """Honor deterministic completion only from successful host control tools."""

        if not results:
            return None
        with self._lock:
            system_names = frozenset(self._system_names)
        messages: list[str] = []
        for result in results:
            metadata = result.metadata
            if (
                not result.success
                or result.tool_name not in system_names
                or metadata is None
                or metadata.get(FINISH_AGENT_TURN) is not True
            ):
                return None
            message = metadata.get(FINISH_AGENT_TURN_MESSAGE)
            if not isinstance(message, str) or not message.strip():
                return None
            messages.append(message.strip())
        return "\n".join(messages)

    def _is_visible_locked(self, name: str) -> bool:
        in_system = name in self._system_names
        in_scope = (
            self._visible_names is None
            or name in self._visible_names
            or (self._include_system_tools and in_system)
        )
        under_ceiling = (
            self._tool_ceiling is None
            or name in self._tool_ceiling
            or (self._ceiling_includes_system_tools and in_system)
        )
        return in_scope and under_ceiling

    def close(self) -> None:
        if self._owns_permission_manager:
            self.permission_manager.close()


def _default_tools(policy: WorkspacePolicy) -> tuple[Tool, ...]:
    return (
        ReadFileTool(policy),
        WriteFileTool(policy),
        EditFileTool(policy),
        RunCommandTool(policy),
        FindFilesTool(policy),
        SearchCodeTool(policy),
    )


def _hook_tool_payload(prepared: PreparedToolCall, policy: WorkspacePolicy) -> dict[str, object]:
    return {
        "id": prepared.id,
        "name": prepared.name,
        "arguments": {
            key: _hook_value(value, policy) for key, value in prepared.arguments.items()
        },
        "target": prepared.target,
        "read_only": prepared.read_only,
    }


def _hook_value(value: object, policy: WorkspacePolicy) -> object:
    if isinstance(value, Path):
        try:
            return policy.relative_target(value)
        except ValueError:
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _hook_value(item, policy) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hook_value(item, policy) for item in value]
    return value


def _reinforced_definition(definition: ToolDefinition) -> ToolDefinition:
    """Repeat the two high-value tool rules where models consume schemas."""

    rule = " 适用时优先使用该专用工具；只有没有专用工具可完成任务时才使用 run_command。"
    if definition.name == "run_command":
        rule = " 仅在没有专用工具能够完成任务时使用本工具。"
    if definition.name in {"write_file", "edit_file"}:
        rule += " 修改已有文件前，必须先用 read_file 读取目标文件或足够的相关上下文。"
    return ToolDefinition(definition.name, definition.description + rule, definition.input_schema)
