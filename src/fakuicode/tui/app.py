"""The responsive Textual application for Fakuicode conversations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import inspect
import os
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread
from time import time_ns

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Collapsible, OptionList, Static

from fakuicode.errors import PermissionPersistenceError, ProviderError, RequestCancelled
from fakuicode.hooks.config import HookConfigRepository
from fakuicode.hooks.models import HookConfigSnapshot, HookEvent
from fakuicode.hooks.runtime import HookDiagnostic, HookEngine
from fakuicode.hooks.trust import HookTrustIdentity, HookTrustRepository, HookTrustStorageError
from fakuicode.commands import (
    DEFAULT_COMMAND_REGISTRY,
    RESERVED_COMMAND_NAMES,
    CommandRegistry,
    compose_command_registry,
)
from fakuicode.mcp.adapter import McpToolAdapter, build_adapters
from fakuicode.mcp.config import resolve_server
from fakuicode.mcp.models import (
    DisabledServerConfig,
    McpConfigSnapshot,
    McpConfigSource,
    McpFailureCode,
    McpServerConfig,
    McpServerState,
    McpServerStatus,
    ResolvedServerConfig,
)
from fakuicode.mcp.runtime import McpClientManager
from fakuicode.mcp.trust import McpTrustRepository, McpTrustStorageError, build_trust_request, server_identity, workspace_id
from fakuicode.memory.service import MemoryService, MemoryStatus
from fakuicode.models import (
    AgentStreamEvent,
    AgentMode,
    ContextStatus,
    ProfileSet,
    ProviderConfig,
    StreamEvent,
    ToolCall,
    ToolResult,
    TokenUsage,
)
from fakuicode.permissions.config import PermissionConfigRepository, PermissionConfigSnapshot
from fakuicode.permissions.manager import ApprovalBroker, PermissionManager
from fakuicode.permissions.models import ApprovalChoice
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.providers.base import ChatProvider
from fakuicode.providers.factory import create_provider
from fakuicode.instructions import (
    InstructionSnapshot,
    InstructionSnapshotLoader,
    sanitize_instruction_metadata,
)
from fakuicode.session import AgentSessionController, SessionController, delete_conversation_with_artifacts
from fakuicode.skills import IsolatedSkillExecutor, SkillDiscovery, SkillManager
from fakuicode.skills.broker import SkillTrustBroker
from fakuicode.skills.install import (
    SkillInstallDecision,
    SkillInstaller,
    SkillInstallRequest,
    SkillPackageFetcher,
)
from fakuicode.skills.install_broker import SkillInstallBroker
from fakuicode.skills.trust import SkillTrustRepository
import fakuicode.skills as skill_package
from fakuicode.storage import ConversationRecord, ConversationStore
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.base import ToolExecution
from fakuicode.tools.registry import ToolRegistry
from fakuicode.tui.model_picker import (
    ConfirmationScreen,
    MemoryChoice,
    MemoryPicker,
    ModelPicker,
    ProfileChoice,
    SessionChoice,
    SessionPicker,
)
from fakuicode.tui.mcp_trust_prompt import McpTrustPrompt
from fakuicode.tui.hook_trust_prompt import HookTrustPrompt
from fakuicode.tui.skill_trust_prompt import SkillTrustPrompt
from fakuicode.tui.skill_install_screen import SkillInstallScreen
from fakuicode.tui.permission_prompt import (
    PermissionPrompt,
    PermissionSettingsAction,
    PermissionSettingsScreen,
    PlanExecutionPrompt,
)
from fakuicode.tui.widgets import AssistantTurn, BrandPanel, ConversationView, PromptEditor, PromptPanel, SystemNotice, UserMessage


_RESUME_GAP_NS = 24 * 60 * 60 * 1_000_000_000
_MAX_TIMESTAMP_NS = (1 << 63) - 1


def _format_context_status(status: ContextStatus) -> str:
    """Render context lifecycle state without exposing summary or artifact content."""

    trigger = {
        "automatic": "Automatic context compaction",
        "manual": "Manual context compaction",
        "emergency": "Emergency context recovery",
    }[status.trigger]
    if status.result == "compacted":
        estimates = ""
        if status.estimated_before is not None and status.estimated_after is not None:
            estimates = f" · ~{status.estimated_before:,} → ~{status.estimated_after:,} tokens"
        return f"{trigger} complete{estimates}"
    if status.result == "noop":
        return f"{trigger} · nothing to compact"
    if status.result == "failed":
        attempts = f" ({status.consecutive_failures}/3)" if status.consecutive_failures else ""
        hint = f" · {status.recovery_hint}" if status.recovery_hint else ""
        return f"{trigger} failed{attempts}{hint}"
    if status.result == "breaker":
        hint = f" · {status.recovery_hint}" if status.recovery_hint else ""
        return f"Context compaction paused after repeated failures{hint}"
    hint = f" · {status.recovery_hint}" if status.recovery_hint else ""
    return f"Context request blocked at the hard limit{hint}"


def _format_memory_status(status: MemoryStatus) -> str:
    state = "on" if status.enabled else "off"
    lines = [
        f"Memory: {state} · user {status.user_count} · project {status.project_count} · "
        f"other projects {status.other_project_count}",
        f"Last update: {status.last_update_code}"
        + (f" · {status.last_update_at}" if status.last_update_at else ""),
    ]
    if status.summaries:
        lines.extend(status.summaries[:10])
    if status.diagnostic_codes:
        lines.append("Warnings: " + ", ".join(status.diagnostic_codes))
    return "\n".join(lines)


def _build_resume_gap_reminder(updated_at: object, now: object) -> tuple[str, str] | None:
    """Build safe user/model reminders for a valid gap of at least one day."""

    if (
        isinstance(updated_at, bool)
        or not isinstance(updated_at, int)
        or isinstance(now, bool)
        or not isinstance(now, int)
        or updated_at < 0
        or now < updated_at
        or now > _MAX_TIMESTAMP_NS
    ):
        return None
    gap = now - updated_at
    if gap < _RESUME_GAP_NS:
        return None
    hours = max(24, gap // (60 * 60 * 1_000_000_000))
    span = f"{hours // 24} days" if hours >= 48 else f"{hours} hours"
    user_notice = (
        f"This conversation was inactive for about {span}. "
        "Files, branches, dependencies, and external state may have changed."
    )
    model_reminder = (
        f"该会话已中断约 {span}。文件、分支、依赖、外部状态和既有结论可能已经变化；"
        "继续前必须重新验证关键事实，不得把旧会话状态直接当作当前事实。"
    )
    return user_notice, model_reminder


def _provider_supports_skill_context(provider: object) -> bool:
    try:
        parameters = inspect.signature(provider.stream_agent).parameters  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return False
    return "request" in parameters or "system_instruction" in parameters


class FakuicodeApp(App[None]):
    """A single-session TUI backed by the existing synchronous chat layer."""

    TITLE = "Fakuicode"
    CSS_PATH = "fakuicode.tcss"
    BINDINGS = [("escape", "cancel", "Cancel"), ("ctrl+c", "quit", "退出"), ("ctrl+q", "quit", "退出")]

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider: ChatProvider | None = None,
        provider_factory: Callable[[ProviderConfig], ChatProvider] = create_provider,
        store: ConversationStore | None = None,
        profile_name: str = "default",
        profiles: ProfileSet | None = None,
        workspace: Path | None = None,
        permission_snapshot: PermissionConfigSnapshot | None = None,
        permission_repository: PermissionConfigRepository | None = None,
        mcp_snapshot: McpConfigSnapshot | None = None,
        mcp_trust_repository: McpTrustRepository | None = None,
        mcp_environment: Mapping[str, str] | None = None,
        mcp_manager_factory: Callable[[], McpClientManager] = McpClientManager,
        hook_snapshot: HookConfigSnapshot | None = None,
        hook_repository: HookConfigRepository | None = None,
        hook_trust_repository: HookTrustRepository | None = None,
        instruction_loader: InstructionSnapshotLoader | None = None,
        memory_service: MemoryService | None = None,
        skill_user_root: Path | None = None,
        skill_trust_repository: SkillTrustRepository | None = None,
        skill_fetcher: SkillPackageFetcher | None = None,
        clock_ns: Callable[[], int] = time_ns,
    ) -> None:
        super().__init__()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.config = config
        self.profiles = profiles or ProfileSet({profile_name: config}, profile_name)
        self._provider_factory = provider_factory
        self._permission_snapshot = permission_snapshot or PermissionConfigSnapshot()
        self._permission_repository = permission_repository
        self._mcp_snapshot = mcp_snapshot or McpConfigSnapshot()
        self._mcp_trust_repository = mcp_trust_repository
        self._mcp_environment = os.environ if mcp_environment is None else mcp_environment
        self._mcp_manager_factory = mcp_manager_factory
        self._hook_snapshot = hook_snapshot or HookConfigSnapshot()
        self._hook_repository = hook_repository
        self._hook_trust_repository = hook_trust_repository
        self._hook_trust_prompt: HookTrustPrompt | None = None
        self._instruction_loader = instruction_loader
        self.memory_service = memory_service
        self._skill_user_root = skill_user_root
        self._skill_trust_repository = skill_trust_repository
        self._skill_fetcher = skill_fetcher
        self._skill_trust_broker = SkillTrustBroker()
        self._skill_trust_prompt: SkillTrustPrompt | None = None
        self._skill_install_broker = SkillInstallBroker()
        self._skill_install_screen: SkillInstallScreen | None = None
        self._skill_install_cancel_event: Event | None = None
        self._skill_install_active = False
        self.skill_manager: SkillManager | None = None
        self._command_registry: CommandRegistry = DEFAULT_COMMAND_REGISTRY
        self._clock_ns = clock_ns
        self.instruction_snapshot = self._load_instruction_snapshot()
        self._mcp_manager: McpClientManager | None = None
        self._mcp_adapters: tuple[McpToolAdapter, ...] = ()
        self._mcp_states: dict[str, McpServerState] = {}
        self._mcp_ready_configs: list[McpServerConfig] = []
        self._mcp_trust_queue: list[McpServerConfig] = []
        self._mcp_trust_prompt: McpTrustPrompt | None = None
        self._approval_broker: ApprovalBroker | None = None
        self._active_permission_request_id: str | None = None
        self._inline_prompt: PermissionPrompt | PlanExecutionPrompt | None = None
        self.store = store
        self.profile_name = profile_name
        self.conversation: ConversationRecord | None = self._select_conversation()
        self._shown_hook_diagnostics: set[tuple[str, str, str, str]] = set()
        self._hook_engine = HookEngine(
            self._hook_snapshot.rules,
            diagnostic_sink=self._record_hook_diagnostic,
            workspace=self.workspace,
        )
        self._hook_engine.dispatch(
            HookEvent.APP_START,
            {"app": {"workspace": str(self.workspace), "outcome": "started"}},
        )
        self._provider = provider or provider_factory(config)
        self.session = self._make_session(self._provider)
        self._last_prompt = next((message.content for message in reversed(self.session.history) if message.role == "user"), None)
        self._active_turn: AssistantTurn | None = None
        self._tool_calls: dict[str, ToolCall] = {}
        self._follow_stream = True
        self._pending_stream_events: SimpleQueue[StreamEvent | AgentStreamEvent] = SimpleQueue()
        self._stream_finished = False
        self._stream_error: str | None = None
        self._stream_follow_scheduled = False
        self._stream_drain_timer = None
        self._is_closing = False
        self._cancel_event: Event | None = None
        self._compact_active = False
        self._memory_diagnostic_error_reported = False
        self._shown_skill_diagnostics: set[tuple[str, str, str]] = set()

    def _select_conversation(self) -> ConversationRecord | None:
        if self.store is None:
            return None
        return self.store.create_conversation("New conversation", self.workspace, self.profile_name)

    def _make_session(self, provider: ChatProvider) -> SessionController | AgentSessionController:
        if hasattr(provider, "stream_agent"):
            broker = ApprovalBroker()
            self._approval_broker = broker
            policy = WorkspacePolicy(self.workspace)
            permissions = PermissionManager(
                self._permission_snapshot,
                DangerousCommandGuard(self.workspace),
                approval_handler=broker,
                repository=self._permission_repository,
            )
            registry = ToolRegistry(
                policy,
                permission_manager=permissions,
                hook_engine=self._hook_engine,
            )
            for adapter in self._mcp_adapters:
                registry.register(adapter)
            if not _provider_supports_skill_context(provider):
                self.skill_manager = None
                self._command_registry = DEFAULT_COMMAND_REGISTRY
                return AgentSessionController(
                    provider,
                    registry,
                    store=self.store,
                    conversation_id=self.conversation.id if self.conversation is not None else None,
                    custom_instructions=self.instruction_snapshot.text,
                    memory_service=self.memory_service,
                )
            user_root = self._skill_user_root or self.workspace / ".fakuicode" / "__user_skills_disabled__"
            builtin_root = Path(skill_package.__file__).parent / "builtin"
            discovery = SkillDiscovery(
                self.workspace / ".fakuicode" / "skills",
                user_root,
                builtin_root,
                reserved_commands=RESERVED_COMMAND_NAMES,
            )
            manager = SkillManager(
                discovery,
                registry,
                context_window=self.config.context_window,
                trust_repository=self._skill_trust_repository,
                trust_handler=lambda request: self._skill_trust_broker.request(
                    request,
                    cancel_event=self._cancel_event,
                ),
            )
            manager.refresh()

            def refresh_after_install() -> object:
                snapshot = manager.refresh()
                if self.is_running:
                    try:
                        self.call_from_thread(self._rebuild_command_registry)
                    except RuntimeError:
                        pass
                return snapshot

            manager.installer = SkillInstaller(
                self.workspace,
                user_root,
                fetcher=self._skill_fetcher,
                refresh=refresh_after_install,
                builtin_root=builtin_root,
            )
            manager.install_confirmation = lambda preview, cancel: self._skill_install_broker.request(
                preview,
                cancel_event=cancel,
            )

            def child_registry() -> ToolRegistry:
                child = ToolRegistry(
                    policy,
                    permission_manager=permissions,
                    owns_permission_manager=False,
                    hook_engine=self._hook_engine,
                )
                for adapter in self._mcp_adapters:
                    child.register(adapter)
                return child

            executor = IsolatedSkillExecutor(
                store=self.store,
                parent_conversation_id=self.conversation.id,
                workspace=self.workspace,
                profiles=self.profiles,
                active_profile_name=self.profile_name,
                parent_messages=lambda: manager.parent_messages,
                provider_factory=self._provider_factory,
                tool_registry_factory=child_registry,
                custom_instructions=self.instruction_snapshot.text,
                readonly_memory_snapshot=self._capture_readonly_memory_snapshot,
            ) if self.store is not None and self.conversation is not None else None
            if executor is not None:
                manager.isolated_runner = lambda name, arguments, cancel: executor.run(
                    manager.snapshot.skills[name], arguments, cancel
                )
            self.skill_manager = manager
            self._rebuild_command_registry()
            return AgentSessionController(
                provider,
                registry,
                store=self.store,
                conversation_id=self.conversation.id if self.conversation is not None else None,
                custom_instructions=self.instruction_snapshot.text,
                memory_service=self.memory_service,
                skill_manager=manager,
            )
        self.skill_manager = None
        self._command_registry = DEFAULT_COMMAND_REGISTRY
        return SessionController(provider)

    def compose(self) -> ComposeResult:
        yield ConversationView(id="conversation")
        yield PromptPanel(self.config.model, command_registry=self._command_registry, id="prompt-panel")

    def on_mount(self) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(BrandPanel(self.config, str(self.workspace)))
        for diagnostic in self._permission_snapshot.diagnostics:
            conversation.mount(SystemNotice(f"Permission configuration locked: {diagnostic}"))
        for warning in self._permission_snapshot.warnings:
            conversation.mount(SystemNotice(f"Permission configuration warning: {warning}"))
        for diagnostic in self._hook_snapshot.diagnostics:
            conversation.mount(SystemNotice(f"Hook 配置警告：{diagnostic}"))
        self._mount_instruction_warning(conversation)
        self._mount_first_memory_notice(conversation)
        self._restore_visible_history(conversation)
        self._show_hook_trust_prompt(conversation)
        self._stream_drain_timer = self.set_interval(1 / 30, self._drain_stream_events)
        if self._hook_trust_prompt is not None:
            self.query_one(PromptEditor).disabled = True
        elif self._mcp_snapshot.has_configuration:
            self.query_one(PromptEditor).disabled = True
            self._begin_mcp_startup()
        else:
            self.query_one(PromptEditor).focus()

    def _record_hook_diagnostic(self, diagnostic: HookDiagnostic) -> None:
        metadata: dict[str, object] = {
            "hook": diagnostic.hook,
            "source": diagnostic.source,
            "event": diagnostic.event,
            "action": diagnostic.action,
            "category": diagnostic.category,
            "background": diagnostic.background,
        }
        if diagnostic.duration_seconds is not None:
            metadata["duration_seconds"] = diagnostic.duration_seconds
        if diagnostic.status is not None:
            metadata["status"] = diagnostic.status
        if self.store is not None and self.conversation is not None:
            try:
                self.store.append_event(
                    self.conversation.id,
                    "hook_diagnostic",
                    "",
                    metadata=metadata,
                )
            except Exception:
                pass
        if not self.is_running:
            return
        try:
            self.call_from_thread(self._show_hook_diagnostic_once, diagnostic)
        except RuntimeError:
            self._show_hook_diagnostic_once(diagnostic)

    def _show_hook_diagnostic_once(self, diagnostic: HookDiagnostic) -> None:
        key = (diagnostic.hook, diagnostic.event, diagnostic.action, diagnostic.category)
        if key in self._shown_hook_diagnostics or not self.is_mounted:
            return
        self._shown_hook_diagnostics.add(key)
        self._notice(
            f"Hook {diagnostic.hook} 执行失败（{diagnostic.category}），Agent 已继续。"
        )

    def _show_hook_trust_prompt(self, conversation: VerticalScroll) -> None:
        snapshot = self._hook_snapshot
        if (
            snapshot.project_trusted
            or snapshot.project_fingerprint is None
            or not snapshot.project_rules
            or self._hook_trust_repository is None
            or self._hook_repository is None
        ):
            return
        prompt = HookTrustPrompt(
            str(self._hook_repository.paths.project),
            snapshot.project_fingerprint,
            len(snapshot.project_rules),
        )
        self._hook_trust_prompt = prompt
        conversation.mount(prompt)

    @on(HookTrustPrompt.Resolved)
    async def _resolve_hook_trust(self, message: HookTrustPrompt.Resolved) -> None:
        if self._hook_trust_prompt is not message.prompt:
            return
        self._hook_trust_prompt = None
        await message.prompt.remove()
        if not message.approve:
            self._notice("项目 Hook 本次保持禁用。")
            self._continue_startup_after_hook_trust()
            return
        fingerprint = self._hook_snapshot.project_fingerprint
        if fingerprint is None or self._hook_trust_repository is None or self._hook_repository is None:
            self._continue_startup_after_hook_trust()
            return
        try:
            self._hook_trust_repository.approve(
                HookTrustIdentity(workspace_id(self.workspace), fingerprint)
            )
            refreshed = self._hook_repository.load()
        except HookTrustStorageError:
            self._notice("Hook 信任无法保存，项目 Hook 保持禁用。")
            self._continue_startup_after_hook_trust()
            return
        self._hook_snapshot = refreshed
        self._hook_engine.replace_rules(refreshed.rules)
        if refreshed.project_trusted:
            self._notice("项目 Hook 已启用，将从后续生命周期事件开始生效。")
        else:
            self._notice("项目 Hook 内容已变化，仍保持禁用。")
        self._continue_startup_after_hook_trust()

    def _continue_startup_after_hook_trust(self) -> None:
        editor = self.query_one(PromptEditor)
        if self._mcp_snapshot.has_configuration:
            editor.disabled = True
            self._begin_mcp_startup()
        else:
            editor.disabled = False
            editor.focus()

    def _restore_visible_history(self, conversation: VerticalScroll) -> None:
        if self.store is not None and self.conversation is not None:
            self._restore_timeline_events(conversation)
            return
        for message in self.session.history:
            if message.role == "user":
                conversation.mount(UserMessage(message.content))
                continue
            turn = AssistantTurn()
            turn.append_text(message.content)
            conversation.mount(turn)
            self.call_later(self._start_restored_finalization, turn)

    def _restore_timeline_events(self, conversation: VerticalScroll) -> None:
        turn: AssistantTurn | None = None
        calls: dict[str, ToolCall] = {}
        latest_summary = self.store.load_latest_context_summary(self.conversation.id)
        for event in self.store.load_events(self.conversation.id):
            if event.kind == "user":
                conversation.mount(UserMessage(event.content))
                turn = None
            elif event.kind == "tool_call":
                if turn is None:
                    turn = AssistantTurn()
                    conversation.mount(turn)
                arguments = event.metadata.get("arguments", {}) if event.metadata is not None else {}
                call = ToolCall(event.call_id or "restored", event.content, arguments if isinstance(arguments, dict) else {})
                calls[call.id] = call
                turn.show_tool_call(call)
            elif event.kind == "tool_result":
                call = calls.get(event.call_id or "")
                if call is not None:
                    metadata = event.metadata or {}
                    tool_name = metadata.get("tool_name") if isinstance(metadata.get("tool_name"), str) else call.name
                    success = metadata.get("success") is True
                    summary = metadata.get("summary") if isinstance(metadata.get("summary"), str) else "restored tool result"
                    duration = metadata.get("duration_seconds")
                    turn.show_tool_result(
                        call,
                        ToolResult(
                            event.call_id or "",
                            tool_name,
                            success,
                            event.content,
                            summary,
                            float(duration) if isinstance(duration, (int, float)) else None,
                        ),
                    )
            elif event.kind == "assistant":
                if turn is None:
                    turn = AssistantTurn()
                    conversation.mount(turn)
                turn.append_text(event.content)
                has_tool_calls = bool(event.metadata and event.metadata.get("tool_calls"))
                if not has_tool_calls:
                    self.call_later(self._start_restored_finalization, turn)
                    turn = None
            elif event.kind == "summary":
                if latest_summary is None or event.sequence != latest_summary.sequence:
                    continue
                metadata = event.metadata or {}
                trigger = metadata.get("trigger")
                before = metadata.get("estimated_before")
                after = metadata.get("estimated_after")
                if trigger not in {"automatic", "manual", "emergency"}:
                    trigger = "automatic"
                conversation.mount(
                    SystemNotice(
                        _format_context_status(
                            ContextStatus(
                                trigger=trigger,
                                result="compacted",
                                estimated_before=before if isinstance(before, int) else None,
                                estimated_after=after if isinstance(after, int) else None,
                            )
                        )
                    )
                )
            elif event.kind in {"context_diagnostic", "hook_diagnostic"}:
                continue
            elif event.kind == "system":
                metadata = event.metadata or {}
                if metadata.get("context_boundary") == "clear" or not event.content:
                    continue
                conversation.mount(SystemNotice(event.content))

    def _start_restored_finalization(self, turn: AssistantTurn) -> None:
        self.run_worker(self._finalize_turn(turn), group="restored-markdown", exit_on_error=False)

    def on_unmount(self) -> None:
        self._is_closing = True
        if isinstance(self.session, AgentSessionController):
            self.session.close()
        self._hook_engine.dispatch(
            HookEvent.APP_STOP,
            {"app": {"workspace": str(self.workspace), "outcome": "completed"}},
        )
        if self.memory_service is not None:
            self.memory_service.close(wait=False)
        if self._mcp_manager is not None:
            self._mcp_manager.close()
        if self._stream_drain_timer is not None:
            self._stream_drain_timer.stop()
        self._skill_trust_broker.close()
        if self._skill_install_cancel_event is not None:
            self._skill_install_cancel_event.set()
        self._skill_install_broker.close()

    def on_prompt_editor_submitted(self, message: PromptEditor.Submitted) -> None:
        if self._active_turn is not None:
            return
        self._refresh_skills()
        if self._command_registry.dispatch(message.text, self):
            return
        self._begin_turn(message.text, message.editor)

    def _begin_turn(
        self,
        text: str,
        editor: PromptEditor,
        *,
        skill_invocation: tuple[str, str | None] | None = None,
    ) -> None:
        if self.conversation is not None and skill_invocation is None:
            self.conversation = self._ensure_conversation_title(self.conversation, text)
        self._dismiss_inline_prompt()
        self._follow_stream = True
        self._pending_stream_events = SimpleQueue()
        self._stream_finished = False
        self._stream_error = None
        self._tool_calls = {}
        self._cancel_event = Event()
        editor.disabled = True
        self._set_status("生成中…")
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(UserMessage(text))
        self._active_turn = AssistantTurn()
        conversation.mount(self._active_turn)
        self._schedule_stream_follow()
        self._last_prompt = text
        self._start_stream_turn(text, self._cancel_event, skill_invocation=skill_invocation)

    def _handle_command(self, text: str) -> None:
        self._command_registry.dispatch(text, self)

    def show_message(self, content: str) -> None:
        self._notice(content)

    def send_user_message(self, content: str) -> None:
        self._begin_turn(content, self.query_one(PromptEditor))

    def invoke_skill(self, name: str, arguments: str | None, original: str) -> None:
        if not isinstance(self.session, AgentSessionController):
            self._notice("Skills require an agent-capable provider.")
            return
        self._begin_turn(
            original,
            self.query_one(PromptEditor),
            skill_invocation=(name, arguments),
        )

    def handle_skills(self, request: SkillInstallRequest | None) -> None:
        if self.skill_manager is None:
            self._notice("当前 Provider 不支持 Skill。")
            return
        if request is None:
            lines = ["有效 Skills："]
            lines.extend(
                f"- {skill.name} · {skill.source.value} · {skill.description}"
                for skill in self.skill_manager.snapshot.skills.values()
            )
            if len(lines) == 1:
                lines.append("- 无")
            if self.skill_manager.snapshot.diagnostics:
                lines.append("禁用诊断：")
                lines.extend(
                    f"- {item.source.value}/{item.name} · {item.code}"
                    for item in self.skill_manager.snapshot.diagnostics
                )
            self._notice("\n".join(lines))
            return
        if self._skill_install_active:
            self._notice("已有 Skill 安装正在进行。")
            return
        self._skill_install_active = True
        cancel_event = Event()
        self._skill_install_cancel_event = cancel_event
        editor = self.query_one(PromptEditor)
        editor.disabled = True
        self._set_status("正在获取 Skill…")

        def install() -> None:
            assert self.skill_manager is not None
            try:
                outcome = self.skill_manager.install(request, cancel_event=cancel_event)
                error: str | None = None
            except RequestCancelled:
                outcome = None
                error = "Skill 安装已取消。"
            except Exception:
                outcome = None
                error = "Skill 安装意外失败。"
            try:
                self.call_from_thread(self._finish_skill_install, outcome, error)
            except RuntimeError:
                return

        Thread(target=install, name="fakuicode-skill-install", daemon=True).start()

    def _finish_skill_install(self, outcome: ToolExecution | None, error: str | None) -> None:
        self._skill_install_active = False
        self._skill_install_cancel_event = None
        if error is not None:
            self._notice(error)
        elif outcome is not None:
            self._notice(outcome.output)
            if outcome.success:
                self._refresh_skills()
        self._restore_input("Ready")

    def set_agent_mode(self, mode: AgentMode) -> bool:
        if not isinstance(self.session, AgentSessionController):
            self._notice("Plan mode requires an agent-capable provider.")
            return False
        if mode == "plan":
            self.session.enable_plan_mode()
        else:
            self.session.disable_plan_mode()
        return True

    def get_agent_mode(self) -> AgentMode:
        return self.session.mode if isinstance(self.session, AgentSessionController) else "execute"

    def has_saved_plan(self) -> bool:
        return isinstance(self.session, AgentSessionController) and bool(self.session.saved_plan)

    def execute_saved_plan(self) -> None:
        self._execute_saved_plan()

    def get_token_usage(self) -> TokenUsage | None:
        return self.session.token_usage if isinstance(self.session, AgentSessionController) else None

    def refresh_status(self) -> None:
        self._restore_input("Ready")

    def start_new_conversation(self) -> None:
        self._new_conversation()
        self._notice(f"Started {self.conversation.id[:8] if self.conversation else 'a new'} conversation.")

    def clear_context(self) -> None:
        if isinstance(self.session, AgentSessionController):
            self.session.clear_context()
        else:
            self.session.history.clear()
        self._notice("Cleared the in-memory model context for this conversation.")

    def compact_context(self) -> None:
        self._begin_compact()

    def show_sessions(self) -> None:
        if self.store is None:
            self._notice("Local session storage is unavailable.")
            return
        records = self._list_titled_conversations()
        self._notice(
            "\n".join(f"{item.id[:8]}  {item.title}  [{item.profile_name}]" for item in records)
            or "No sessions."
        )

    def open_resume_picker(self) -> None:
        self._open_session_picker()

    def delete_conversation(self, argument: str | None) -> None:
        if argument is None:
            self._open_session_delete_picker()
        else:
            self._delete_conversation(argument)

    def retry_last_prompt(self) -> None:
        if self._last_prompt is None:
            self._notice("There is no previous prompt to retry.")
        else:
            self._begin_turn(self._last_prompt, self.query_one(PromptEditor))

    def show_runtime_status(self) -> None:
        conversation_id = self.conversation.id[:8] if self.conversation else "memory"
        self._notice(
            f"Profile: {self.profile_name} · Model: {self.config.model} · "
            f"Session: {conversation_id} · {self._permission_status()}\n"
            f"Instructions: {len(self.instruction_snapshot.loaded_layers)} layers · "
            f"{self.instruction_snapshot.processed_target_count} targets · "
            f"{self.instruction_snapshot.byte_count} bytes · "
            f"{self.instruction_snapshot.warning_count} warning(s)"
        )

    def show_mcp_status(self) -> None:
        self._notice(self._format_mcp_status())

    def open_model_picker(self, argument: str | None) -> None:
        if argument is None:
            self._open_model_picker()
        else:
            self._notice("Please use /model to open the picker.")

    def handle_memory(self, argument: str | None) -> None:
        self._handle_memory_command(argument)

    def open_permissions(self) -> None:
        self._open_permission_settings()

    def _begin_mcp_startup(self) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        for diagnostic in self._mcp_snapshot.diagnostics:
            conversation.mount(SystemNotice(f"MCP configuration: {diagnostic.message}"))
            if diagnostic.server_name is not None:
                self._mcp_states[diagnostic.server_name] = McpServerState(
                    diagnostic.server_name,
                    None,
                    McpServerStatus.CONFIG_ERROR,
                    failure_code=diagnostic.failure_code,
                    public_summary=diagnostic.message,
                )
        for config in self._mcp_snapshot.servers:
            if isinstance(config, DisabledServerConfig):
                self._mcp_states[config.name] = McpServerState(
                    config.name, None, McpServerStatus.DISABLED
                )
                continue
            if config.source is McpConfigSource.USER:
                self._mcp_ready_configs.append(config)
                continue
            identity = server_identity(self.workspace, config)
            if (
                self._mcp_trust_repository is not None
                and self._mcp_trust_repository.is_trusted(identity)
            ):
                self._mcp_ready_configs.append(config)
            else:
                self._mcp_states[config.name] = McpServerState(
                    config.name, config.transport, McpServerStatus.PENDING_TRUST
                )
                self._mcp_trust_queue.append(config)
        self._show_next_mcp_trust()

    def _show_next_mcp_trust(self) -> None:
        if self._mcp_trust_queue:
            config = self._mcp_trust_queue[0]
            request = build_trust_request(self.workspace, config)
            if request is None:
                self._mcp_trust_queue.pop(0)
                self._show_next_mcp_trust()
                return
            prompt = McpTrustPrompt(request)
            self._mcp_trust_prompt = prompt
            self.query_one("#conversation", VerticalScroll).mount(prompt)
            return
        self._launch_mcp_discovery()

    def _show_next_skill_trust(self) -> None:
        if self._skill_trust_prompt is not None or not self.is_mounted:
            return
        request = self._skill_trust_broker.next_request()
        if request is None:
            return
        prompt = SkillTrustPrompt(request)
        self._skill_trust_prompt = prompt
        self.query_one("#conversation", VerticalScroll).mount(prompt)

    @on(SkillTrustPrompt.Resolved)
    async def _resolve_skill_trust(self, message: SkillTrustPrompt.Resolved) -> None:
        if self._skill_trust_prompt is not message.prompt:
            return
        self._skill_trust_prompt = None
        self._skill_trust_broker.resolve(message.prompt.request.fingerprint, message.approve)
        await message.prompt.remove()

    def _show_next_skill_install(self) -> None:
        if self._skill_install_screen is not None or not self.is_mounted:
            return
        preview = self._skill_install_broker.next_request()
        if preview is None:
            return
        screen = SkillInstallScreen(preview)
        self._skill_install_screen = screen
        self.push_screen(screen, self._resolve_skill_install)

    def _resolve_skill_install(self, decision: SkillInstallDecision) -> None:
        screen = self._skill_install_screen
        if screen is None:
            return
        self._skill_install_screen = None
        self._skill_install_broker.resolve(screen.preview, decision)

    @on(McpTrustPrompt.Resolved)
    async def _resolve_mcp_trust(self, message: McpTrustPrompt.Resolved) -> None:
        if self._mcp_trust_prompt is not message.prompt or not self._mcp_trust_queue:
            return
        config = self._mcp_trust_queue.pop(0)
        self._mcp_trust_prompt = None
        await message.prompt.remove()
        approved = False
        if message.approve and self._mcp_trust_repository is not None:
            try:
                self._mcp_trust_repository.approve(server_identity(self.workspace, config))
            except McpTrustStorageError:
                self._notice("MCP trust could not be saved; the server remains disabled.")
            else:
                approved = True
        if approved:
            self._mcp_ready_configs.append(config)
            self._mcp_states.pop(config.name, None)
        else:
            self._mcp_states[config.name] = McpServerState(
                config.name,
                config.transport,
                McpServerStatus.TRUST_DENIED,
                failure_code=McpFailureCode.TRUST_STORAGE if message.approve else None,
                public_summary="项目 Server 未获信任。",
            )
        self._show_next_mcp_trust()

    def _launch_mcp_discovery(self) -> None:
        resolved: list[ResolvedServerConfig] = []
        for config in self._mcp_ready_configs:
            value, diagnostic = resolve_server(config, self._mcp_environment)
            if value is not None:
                resolved.append(value)
                continue
            if diagnostic is not None:
                self._mcp_states[config.name] = McpServerState(
                    config.name,
                    config.transport,
                    McpServerStatus.CONFIG_ERROR,
                    failure_code=diagnostic.failure_code,
                    public_summary=diagnostic.message,
                )
        if not resolved:
            self._finish_mcp_startup(None, (), ())
            return
        self._set_status("Discovering MCP tools…")
        manager = self._mcp_manager_factory()
        self._mcp_manager = manager

        def discover() -> None:
            snapshot = manager.start(tuple(resolved))
            adapters = build_adapters(manager, tuple(resolved))
            if self._is_closing:
                manager.close()
                return
            self.call_from_thread(
                self._finish_mcp_startup,
                manager,
                tuple(resolved),
                adapters,
                snapshot.states,
            )

        Thread(target=discover, name="fakuicode-mcp-startup", daemon=True).start()

    def _finish_mcp_startup(
        self,
        manager: McpClientManager | None,
        resolved: tuple[ResolvedServerConfig, ...],
        adapters: tuple[McpToolAdapter, ...],
        states: tuple[McpServerState, ...] = (),
    ) -> None:
        del manager, resolved
        registered_by_server: dict[str, int] = {}
        for adapter in adapters:
            registered_by_server[adapter.binding.server_name] = (
                registered_by_server.get(adapter.binding.server_name, 0) + 1
            )
        for state in states:
            self._mcp_states[state.name] = replace(
                state, tool_count=registered_by_server.get(state.name, 0)
            )
        self._mcp_adapters = adapters
        if isinstance(self.session, AgentSessionController):
            registry = self.session.runner.tools
            if isinstance(registry, ToolRegistry):
                for adapter in adapters:
                    if not registry.is_known(adapter.definition.name):
                        registry.register(adapter)
        self._refresh_skills()
        connected = sum(
            state.status in {McpServerStatus.CONNECTED, McpServerStatus.RESTART_REQUIRED}
            for state in self._mcp_states.values()
        )
        failed = sum(state.status is McpServerStatus.FAILED for state in self._mcp_states.values())
        disabled = sum(state.status is McpServerStatus.DISABLED for state in self._mcp_states.values())
        denied = sum(state.status is McpServerStatus.TRUST_DENIED for state in self._mcp_states.values())
        extras = []
        if failed:
            extras.append(f"{failed} failed")
        if disabled:
            extras.append(f"{disabled} disabled")
        if denied:
            extras.append(f"{denied} trust denied")
        notice = f"Connected to {connected} MCP server(s), {len(adapters)} tools registered"
        if extras:
            notice += " · " + " · ".join(extras)
        self._notice(notice)
        self._restore_input("Ready")

    def _format_mcp_status(self) -> str:
        if not self._mcp_snapshot.has_configuration:
            return "No MCP servers configured."
        if self._mcp_manager is not None:
            registered_by_server: dict[str, int] = {}
            for adapter in self._mcp_adapters:
                registered_by_server[adapter.binding.server_name] = (
                    registered_by_server.get(adapter.binding.server_name, 0) + 1
                )
            for state in self._mcp_manager.snapshot().states:
                self._mcp_states[state.name] = replace(
                    state, tool_count=registered_by_server.get(state.name, 0)
                )
        if not self._mcp_states:
            return "MCP startup is still in progress."
        lines: list[str] = []
        for name in sorted(self._mcp_states):
            state = self._mcp_states[name]
            transport = state.transport.value if state.transport is not None else "-"
            line = f"{name} · {transport} · {state.status.value} · {state.tool_count} tools"
            if state.status is McpServerStatus.RESTART_REQUIRED:
                line += " · restart required"
            elif state.public_summary:
                line += f" · {state.public_summary[:120]}"
            lines.append(line)
        return "\n".join(lines)

    def _new_conversation(self, *, reload_instructions: bool = True) -> None:
        self._close_agent_session()
        if reload_instructions:
            self.instruction_snapshot = self._load_instruction_snapshot()
        if self.store is not None:
            self.conversation = self.store.create_conversation("New conversation", self.workspace, self.profile_name)
        self.session = self._make_session(self._provider)
        self._last_prompt = None
        if reload_instructions and self.is_mounted:
            self._mount_instruction_warning(self.query_one("#conversation", VerticalScroll))

    def _resume_conversation(self, prefix: str) -> None:
        if self.store is None:
            self._notice("Local session storage is unavailable.")
            return
        matches = [
            record
            for record in self.store.list_conversations()
            if record.id.startswith(prefix) and record.workspace.resolve() == self.workspace
        ]
        if len(matches) != 1:
            self._notice("Session id was not found or is ambiguous.")
            return
        conversation = matches[0]
        try:
            resume_reminder = _build_resume_gap_reminder(
                conversation.updated_at,
                self._clock_ns(),
            )
        except Exception:
            resume_reminder = None
        try:
            config = self.profiles.get(conversation.profile_name)
        except KeyError:
            self._notice(f"Profile '{conversation.profile_name}' is no longer configured.")
            return
        self.conversation = conversation
        self.profile_name = conversation.profile_name
        self.config = config
        self._close_agent_session()
        self.instruction_snapshot = self._load_instruction_snapshot()
        self._provider = self._provider_factory(self.config)
        self.query_one(PromptPanel).set_model(self.config.model)
        self.session = self._make_session(self._provider)
        if resume_reminder is not None and isinstance(self.session, AgentSessionController):
            self.session.set_resume_reminder(resume_reminder[1])
        self._last_prompt = next((message.content for message in reversed(self.session.history) if message.role == "user"), None)
        self.run_worker(
            self._replace_visible_history(
                f"Resumed {self.conversation.id[:8]}.",
                resume_reminder[0] if resume_reminder is not None else None,
            ),
            group="resume-history",
            exit_on_error=False,
        )

    async def _replace_visible_history(self, notice: str, supplemental_notice: str | None = None) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.remove_children(
            child for child in conversation.children if not isinstance(child, BrandPanel)
        )
        self._mount_instruction_warning(conversation)
        self._restore_visible_history(conversation)
        self._notice(notice)
        if supplemental_notice is not None:
            self._notice(supplemental_notice)

    def _delete_conversation(self, prefix: str) -> None:
        if self.store is None:
            self._notice("Local session storage is unavailable.")
            return
        matches = [record for record in self.store.list_conversations() if record.id.startswith(prefix)]
        if len(matches) != 1:
            self._notice("Session id was not found or is ambiguous.")
            return
        deleted = matches[0]
        try:
            result = delete_conversation_with_artifacts(self.store, deleted.id)
        except Exception:
            self._notice("Session deletion failed; the saved conversation was retained.")
            return
        if self.conversation is not None and deleted.id == self.conversation.id:
            self._new_conversation(reload_instructions=False)
        notice = f"Deleted {deleted.id[:8]}."
        if result.warning is not None:
            notice += f" {result.warning}"
        self._notice(notice)

    def _switch_profile(self, name: str) -> None:
        try:
            config = self.profiles.get(name)
        except KeyError:
            self._notice(f"Unknown profile '{name}'.")
            return
        self.config = config
        self.profile_name = name
        self._provider = self._provider_factory(config)
        self._new_conversation()
        self.query_one(PromptPanel).set_model(config.model)
        self._notice(f"Switched to {name} · {config.model}.")

    def _load_instruction_snapshot(self) -> InstructionSnapshot:
        if self._instruction_loader is None:
            return InstructionSnapshot.empty()
        try:
            return self._instruction_loader.load()
        except Exception:
            return InstructionSnapshot.failed()

    def _refresh_skills(self) -> None:
        if self.skill_manager is None:
            return
        snapshot = self.skill_manager.refresh()
        self._rebuild_command_registry()
        if not self.is_running:
            return
        omitted = self.skill_manager.catalog.omitted_names
        if omitted:
            omitted_key = ("catalog", ",".join(omitted), "catalog_omitted")
            if omitted_key not in self._shown_skill_diagnostics:
                self._shown_skill_diagnostics.add(omitted_key)
                self._notice(
                    "Skill warning · catalog_omitted · " + ", ".join(omitted)
                )
        for diagnostic in snapshot.diagnostics:
            key = (diagnostic.source.value, diagnostic.name, diagnostic.code)
            if key in self._shown_skill_diagnostics:
                continue
            self._shown_skill_diagnostics.add(key)
            self._notice(
                f"Skill warning · {diagnostic.source.value}/{diagnostic.name} · {diagnostic.code}"
            )

    def _capture_readonly_memory_snapshot(self) -> object | None:
        if self.memory_service is None:
            return None
        try:
            return self.memory_service.capture_turn_context().memory_snapshot
        except Exception:
            return None

    def _rebuild_command_registry(self) -> None:
        skills = ()
        if self.skill_manager is not None:
            skills = tuple(
                (skill.name, skill.description)
                for skill in self.skill_manager.snapshot.skills.values()
            )
        self._command_registry = compose_command_registry(skills)
        if self.is_running:
            self.query_one(PromptPanel).set_command_registry(self._command_registry)

    def _mount_instruction_warning(self, conversation: VerticalScroll) -> None:
        """Show one safe, structured instruction-loader warning block per load."""

        snapshot = self.instruction_snapshot
        if not snapshot.warning_count:
            return
        details = [
            " ".join(
                part
                for part in (
                    f"code={diagnostic.code}",
                    f"scope={diagnostic.scope}",
                    f"source={sanitize_instruction_metadata(diagnostic.source)}",
                    f"line={diagnostic.line}" if diagnostic.line is not None else "",
                )
                if part
            )
            for diagnostic in snapshot.diagnostics
        ]
        if snapshot.global_failure is not None:
            details.append(f"failure={snapshot.global_failure}")
        message = f"Project instructions: {snapshot.warning_count} warning(s)"
        if details:
            message += "\n" + "\n".join(details)
        conversation.mount(SystemNotice(message))

    def _mount_first_memory_notice(self, conversation: VerticalScroll) -> None:
        if self.memory_service is None:
            return
        try:
            if not self.memory_service.first_notice_needed():
                return
            conversation.mount(
                SystemNotice(
                    "Automatic memory is enabled: the current model may extract useful "
                    "notes in the background, saved locally. Use /memory off to disable it."
                )
            )
            self.memory_service.confirm_first_notice()
        except Exception:
            return

    def _handle_memory_command(self, argument: str | None) -> None:
        service = self.memory_service
        if service is None:
            self._notice("Automatic memory is unavailable.")
            return
        try:
            if argument is None:
                self._notice(_format_memory_status(service.status()))
                return
            if argument in {"on", "off"}:
                enabled = argument == "on"
                service.set_enabled(enabled)
                state = "enabled" if enabled else "disabled"
                self._notice(f"Automatic memory {state} for future turns.")
                return
            if argument == "forget":
                self._open_memory_picker()
                return
            prefix = "forget "
            if argument.startswith(prefix):
                result = service.forget(argument[len(prefix) :])
                if result.success:
                    self._notice("Memory entry forgotten.")
                else:
                    self._notice("Memory entry was not found in the current scopes.")
                return
        except Exception:
            self._notice("Automatic memory operation failed safely.")

    def _show_memory_diagnostics(self) -> None:
        if self.memory_service is None:
            return
        try:
            codes = self.memory_service.consume_diagnostic_codes()
        except Exception:
            if self._memory_diagnostic_error_reported:
                return
            self._memory_diagnostic_error_reported = True
            codes = ("unavailable",)
        else:
            self._memory_diagnostic_error_reported = False
        if codes:
            safe_codes = ", ".join(dict.fromkeys(code[:64] for code in codes))
            self._notice(f"Automatic memory warning: {safe_codes}.")

    def _open_model_picker(self) -> None:
        """Open a local-only picker that returns a Profile name or cancellation."""
        choices = tuple(
            ProfileChoice(profile_name=name, model_name=config.model) for name, config in self.profiles.profiles.items()
        )
        self.push_screen(ModelPicker(choices, self.profile_name), self._apply_model_picker_selection)

    def _open_session_picker(self) -> None:
        choices = self._session_choices()
        if choices is None:
            return
        current_id = self.conversation.id if self.conversation is not None else None
        self.push_screen(SessionPicker(choices, current_id), self._apply_session_picker_selection)

    def _open_session_delete_picker(self) -> None:
        choices = self._session_choices()
        if choices is None:
            return
        current_id = self.conversation.id if self.conversation is not None else None
        self.push_screen(
            SessionPicker(choices, current_id, purpose="delete"),
            self._confirm_session_deletion,
        )

    def _session_choices(self) -> tuple[SessionChoice, ...] | None:
        if self.store is None:
            self._notice("Local session storage is unavailable.")
            return None
        records = [
            record
            for record in self._list_titled_conversations()
            if record.workspace.resolve() == self.workspace
        ]
        if not records:
            self._notice("No saved conversations.")
            return None
        choices_list: list[SessionChoice] = []
        for record in records:
            try:
                message_count = self.store.visible_message_count(record.id)
            except Exception:
                message_count = None
            choices_list.append(
                SessionChoice(
                    record.id,
                    record.title,
                    record.profile_name,
                    record.updated_at,
                    message_count,
                )
            )
        return tuple(choices_list)

    def _ensure_conversation_title(
        self,
        record: ConversationRecord,
        candidate: str,
    ) -> ConversationRecord:
        if self.store is None:
            return record
        try:
            return self.store.ensure_conversation_title(record.id, candidate)
        except Exception:
            return record

    def _list_titled_conversations(self) -> list[ConversationRecord]:
        assert self.store is not None
        try:
            self.store.backfill_default_conversation_titles()
        except Exception:
            pass
        return self.store.list_conversations()

    def _open_memory_picker(self) -> None:
        service = self.memory_service
        if service is None:
            self._notice("Automatic memory is unavailable.")
            return
        try:
            choices = service.list_visible_entries()
        except Exception:
            self._notice("Automatic memory operation failed safely.")
            return
        if not choices:
            self._notice("No memory entries are available in the current scopes.")
            return
        picker_choices = tuple(
            MemoryChoice(item.id, item.scope, item.category, item.summary)
            for item in choices
        )
        self.push_screen(MemoryPicker(picker_choices), self._confirm_memory_forget)

    def _confirm_memory_forget(self, entry_id: str | None) -> None:
        if entry_id is None:
            return
        self.push_screen(
            ConfirmationScreen("Forget the selected memory entry?", "Forget memory"),
            lambda confirmed: self._apply_memory_forget_confirmation(entry_id, confirmed),
        )

    def _apply_memory_forget_confirmation(self, entry_id: str, confirmed: bool) -> None:
        if not confirmed:
            return
        service = self.memory_service
        if service is None:
            self._notice("Automatic memory is unavailable.")
            return
        try:
            result = service.forget(entry_id)
        except Exception:
            self._notice("Automatic memory operation failed safely.")
            return
        if result.success:
            self._notice("Memory entry forgotten.")
        else:
            self._notice("Memory entry was not found in the current scopes.")

    def _confirm_session_deletion(self, conversation_id: str | None) -> None:
        if conversation_id is None:
            return
        title = "the selected conversation"
        if self.store is not None:
            try:
                title = f'"{self.store.get_conversation(conversation_id).title}"'
            except Exception:
                pass
        self.push_screen(
            ConfirmationScreen(f"Delete {title}? This cannot be undone.", "Delete conversation"),
            lambda confirmed: self._apply_session_delete_confirmation(conversation_id, confirmed),
        )

    def _apply_session_delete_confirmation(self, conversation_id: str, confirmed: bool) -> None:
        if confirmed:
            self._delete_conversation(conversation_id)

    def _open_permission_settings(self) -> None:
        manager = self._permission_manager()
        if manager is None:
            self._notice("Permissions require an agent-capable provider.")
            return
        self.push_screen(
            PermissionSettingsScreen(manager.mode, manager.snapshot),
            self._apply_permission_settings,
        )

    def _apply_permission_settings(self, action: PermissionSettingsAction | None) -> None:
        if action is None:
            return
        manager = self._permission_manager()
        if manager is None:
            return
        if action.mode is not None:
            try:
                manager.set_mode(action.mode)
            except ValueError as error:
                self._notice(str(error))
            else:
                self._notice(f"Permission mode for this session: {action.mode.value}.")
        if action.project_trusted is not None:
            try:
                self._permission_snapshot = manager.set_project_trusted(action.project_trusted)
            except PermissionPersistenceError as error:
                self._notice(str(error))
            else:
                state = "trusted" if action.project_trusted else "untrusted"
                self._notice(f"Project shared permission rules are now {state}.")

    def _apply_model_picker_selection(self, profile_name: str | None) -> None:
        if profile_name is not None:
            self._switch_profile(profile_name)

    def _apply_session_picker_selection(self, conversation_id: str | None) -> None:
        if conversation_id is not None:
            self._resume_conversation(conversation_id)

    def _permission_manager(self) -> PermissionManager | None:
        if not isinstance(self.session, AgentSessionController):
            return None
        manager = getattr(self.session.runner.tools, "permission_manager", None)
        return manager if isinstance(manager, PermissionManager) else None

    def _permission_status(self) -> str:
        manager = self._permission_manager()
        if manager is None:
            return "Permissions: unavailable"
        if manager.snapshot.locked:
            return "Permissions: locked/strict"
        trust = "trusted" if manager.snapshot.project_trusted else "untrusted"
        return f"Permissions: {manager.mode.value} · Project: {trust}"

    def _notice(self, content: str) -> None:
        self.query_one("#conversation", VerticalScroll).mount(SystemNotice(content))
        self._set_status(content.splitlines()[0])
        self._schedule_stream_follow()

    def _execute_saved_plan(self) -> None:
        if not isinstance(self.session, AgentSessionController):
            self._notice("Plan execution requires an agent-capable provider.")
            return
        try:
            plan = self.session.prepare_plan_execution()
        except ValueError as error:
            self._notice(str(error))
            self._restore_input("Ready")
            return
        self._begin_turn(
            "Execute the saved plan. Continue using tools until it is complete:\n\n" + plan,
            self.query_one(PromptEditor),
        )

    def _show_inline_prompt(self, prompt: PermissionPrompt | PlanExecutionPrompt) -> None:
        self._dismiss_inline_prompt()
        self._inline_prompt = prompt
        self.query_one("#conversation", VerticalScroll).mount(prompt)
        self._schedule_stream_follow()

    @on(events.DescendantFocus)
    def _keep_choice_prompt_focused(self, event: events.DescendantFocus) -> None:
        prompt = self._hook_trust_prompt or self._mcp_trust_prompt or self._inline_prompt
        if prompt is None or not prompt.is_mounted:
            return
        options = prompt.query_one(OptionList)
        if event.widget is not options:
            options.focus()

    def _dismiss_inline_prompt(self) -> None:
        prompt = self._inline_prompt
        self._inline_prompt = None
        if prompt is not None and prompt.is_mounted:
            prompt.remove()

    def _close_agent_session(self) -> None:
        self._dismiss_inline_prompt()
        current = getattr(self, "session", None)
        if isinstance(current, AgentSessionController):
            manager = getattr(current.runner.tools, "permission_manager", None)
            if isinstance(manager, PermissionManager):
                self._permission_snapshot = manager.snapshot
            current.close()
        self._approval_broker = None
        self._active_permission_request_id = None

    def _start_stream_turn(
        self,
        text: str,
        cancel_event: Event,
        *,
        skill_invocation: tuple[str, str | None] | None = None,
    ) -> None:
        Thread(
            target=self._stream_turn,
            args=(text, cancel_event, skill_invocation),
            daemon=True,
        ).start()

    def _begin_compact(self) -> None:
        if not isinstance(self.session, AgentSessionController):
            self._notice("Context compaction requires an agent-capable provider.")
            return
        if self._active_turn is not None or self._compact_active:
            self._notice("Another model operation is already running.")
            return
        self._compact_active = True
        self._cancel_event = Event()
        self.query_one(PromptEditor).disabled = True
        self._set_status("Compacting context…")
        Thread(target=self._compact_context, args=(self._cancel_event,), daemon=True).start()

    def _compact_context(self, cancel_event: Event) -> None:
        try:
            assert isinstance(self.session, AgentSessionController)
            status = self.session.compact(cancel_event=cancel_event)
        except RequestCancelled:
            self._notify_compact_finished(None, "Request cancelled.")
        except Exception:
            self._notify_compact_finished(None, "Context compaction failed.")
        else:
            self._notify_compact_finished(status, None)

    def _notify_compact_finished(
        self,
        status: ContextStatus | None,
        error: str | None,
    ) -> None:
        if self._is_closing:
            return
        try:
            self.call_from_thread(self._finish_compact, status, error)
        except RuntimeError:
            return

    def _finish_compact(
        self,
        status: ContextStatus | None,
        error: str | None,
    ) -> None:
        self._compact_active = False
        self._cancel_event = None
        if error is not None:
            self._restore_input(error)
            return
        assert status is not None
        rendered = _format_context_status(status)
        self._notice(rendered)
        self._restore_input(rendered)

    def _stream_turn(
        self,
        text: str,
        cancel_event: Event,
        skill_invocation: tuple[str, str | None] | None = None,
    ) -> None:
        try:
            if isinstance(self.session, AgentSessionController):
                events = self.session.send(
                    text,
                    cancel_event=cancel_event,
                    skill_invocation=skill_invocation,
                )
            else:
                events = self.session.send(text, cancel_event=cancel_event)
            for event in events:
                self._pending_stream_events.put(event)
        except RequestCancelled as error:
            self._notify_stream_finished(str(error))
        except ProviderError as error:
            self._notify_stream_finished(str(error))
        except Exception:
            self._notify_stream_finished("Unexpected application failure.")
        else:
            self._notify_stream_finished()

    def action_cancel(self) -> None:
        if self._skill_install_active and self._skill_install_cancel_event is not None:
            self._skill_install_cancel_event.set()
            self._set_status("正在取消 Skill 安装…")
            return
        if (self._active_turn is None and not self._compact_active) or self._cancel_event is None:
            return
        self._cancel_event.set()
        if isinstance(self.session, AgentSessionController):
            self.session.cancel()
        self._set_status("Cancelling…")

    def _notify_stream_finished(self, error: str | None = None) -> None:
        if self._is_closing:
            return
        try:
            self.call_from_thread(self._mark_stream_finished, error)
        except RuntimeError:
            # A daemon provider thread may finish just after the TUI event loop closes.
            return

    def _mark_stream_finished(self, error: str | None = None) -> None:
        self._stream_finished = True
        if error is not None:
            self._stream_error = error
        self._drain_stream_events()

    def _drain_stream_events(self) -> None:
        self._show_next_skill_install()
        self._show_next_skill_trust()
        self._show_memory_diagnostics()
        self._drain_permission_request()
        handled_events = False
        for _ in range(256):
            try:
                event = self._pending_stream_events.get_nowait()
            except Empty:
                break
            self._handle_stream_event(event)
            handled_events = True

        if handled_events:
            self._schedule_stream_follow()

        if not self._stream_finished:
            return

        if self._pending_stream_events.empty():
            error = self._stream_error
            self._stream_finished = False
            self._stream_error = None
            if error is None:
                self._finish_success()
            else:
                self._finish_error(error)

    def _drain_permission_request(self) -> None:
        broker = self._approval_broker
        if broker is None or self._active_permission_request_id is not None or self._is_closing:
            return
        request = broker.next_request()
        if request is None:
            return
        self._active_permission_request_id = request.request_id
        self._set_status(f"等待权限确认 · {request.tool_name}")
        self._show_inline_prompt(PermissionPrompt(request))

    @on(PermissionPrompt.Resolved)
    def _resolve_permission(self, message: PermissionPrompt.Resolved) -> None:
        request_id = message.prompt.request.request_id
        if self._active_permission_request_id != request_id:
            return
        self._dismiss_inline_prompt()
        self._active_permission_request_id = None
        broker = self._approval_broker
        if message.cancel_turn:
            # Cancel before unblocking the tool worker so it cannot start
            # another model round after receiving this denial.
            self.action_cancel()
        if broker is not None:
            broker.resolve(request_id, message.choice)
        if not message.cancel_turn:
            self._set_status("继续执行…")

    @on(PlanExecutionPrompt.Resolved)
    def _resolve_plan_execution(self, message: PlanExecutionPrompt.Resolved) -> None:
        if self._inline_prompt is not message.prompt:
            return
        self._dismiss_inline_prompt()
        if message.execute:
            self._execute_saved_plan()
        else:
            self._restore_input("Ready")

    def _handle_stream_event(self, event: StreamEvent | AgentStreamEvent) -> None:
        if event.kind == "context_status" and event.context_status is not None:
            self._notice(_format_context_status(event.context_status))
            return
        turn = self._active_turn
        if turn is None:
            return
        if event.kind == "text_delta":
            turn.append_text(event.text)
        elif event.kind == "thinking_start":
            turn.start_thinking()
        elif event.kind == "thinking_delta":
            turn.append_thinking(event.text)
        elif event.kind == "progress" and event.progress is not None:
            label = "Planning" if isinstance(self.session, AgentSessionController) and self.session.mode == "plan" else "Working"
            self._set_status(f"{label} · Round {event.progress.round_number} · {event.progress.phase}")
        elif event.kind == "usage":
            usage = self.session.token_usage if isinstance(self.session, AgentSessionController) else None
            if usage is not None:
                self._set_status(f"Tokens {usage.input_tokens}/{usage.output_tokens}")
        elif event.kind == "completed":
            self.run_worker(self._finalize_turn(turn), group="markdown-finalization", exit_on_error=False)
        elif event.kind in {"cancelled", "error"}:
            self._stream_error = event.text or event.kind
        elif event.kind == "tool_call" and event.tool_call is not None:
            self._tool_calls[event.tool_call.id] = event.tool_call
            turn.show_tool_call(event.tool_call)
            self._set_status(f"Running {event.tool_call.name}…")
        elif event.kind == "tool_result" and event.tool_result is not None:
            call = self._tool_calls.get(event.tool_result.call_id)
            if call is not None:
                turn.show_tool_result(call, event.tool_result)
            self._set_status(event.tool_result.summary)

    async def _finalize_turn(self, turn: AssistantTurn) -> None:
        await turn.finalize()
        self._schedule_stream_follow()

    def _schedule_stream_follow(self) -> None:
        if not self._follow_stream or self._stream_follow_scheduled:
            return
        self._stream_follow_scheduled = True
        self.call_later(self._follow_conversation_end)

    def _follow_conversation_end(self) -> None:
        self._stream_follow_scheduled = False
        if self._follow_stream:
            try:
                conversation = self.query_one("#conversation", VerticalScroll)
            except NoMatches:
                return
            conversation.refresh(layout=True)
            conversation.scroll_end(animate=False)

    @on(Collapsible.Toggled)
    def _pause_stream_following_for_expanded_thinking(self, event: Collapsible.Toggled) -> None:
        if not event.collapsible.collapsed:
            self._follow_stream = False

    @on(ConversationView.UserScrolled)
    def _handle_manual_conversation_scroll(self, message: ConversationView.UserScrolled) -> None:
        if message.direction < 0:
            self._follow_stream = False
            return
        self._resume_stream_following_if_at_bottom()

    def _resume_stream_following_if_at_bottom(self) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        if conversation.scroll_y >= conversation.max_scroll_y - 0.5:
            self._follow_stream = True
            self._schedule_stream_follow()

    def _finish_success(self) -> None:
        self._active_turn = None
        if (
            isinstance(self.session, AgentSessionController)
            and self.session.mode == "plan"
            and self.session.saved_plan
        ):
            self._set_status("Plan mode · 计划已就绪，请选择是否执行")
            self._show_inline_prompt(PlanExecutionPrompt())
            return
        self._restore_input("Ready")

    def _finish_error(self, message: str) -> None:
        if self._active_turn is not None:
            self._active_turn.show_error(message)
        self._active_turn = None
        self._restore_input(f"Error: {message}")

    def _restore_input(self, status: str) -> None:
        editor = self.query_one(PromptEditor)
        editor.disabled = False
        editor.focus()
        if status == "Ready" and isinstance(self.session, AgentSessionController):
            if self.session.mode == "plan":
                status = "Plan mode · ready for /do"
            else:
                usage = self.session.token_usage
                status = "Ready · tokens unavailable" if usage is None else f"Ready · tokens {usage.input_tokens}/{usage.output_tokens}"
            status = f"{status} · {self._permission_status()}"
        self._set_status(status)

    def _set_status(self, status: str) -> None:
        mode = "PLAN" if self.get_agent_mode() == "plan" else "DEFAULT"
        self.query_one("#status", Static).update(Text(f"[{mode}] {status}"))
