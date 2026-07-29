"""Runtime assembly boundaries kept separate from Textual event handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Event

from fakuicode.commands import (
    DEFAULT_COMMAND_REGISTRY,
    RESERVED_COMMAND_NAMES,
    CommandRegistry,
    compose_command_registry,
)
from fakuicode.hooks.config import HookConfigRepository, HookPaths
from fakuicode.hooks.models import HookConfigSnapshot
from fakuicode.hooks.runtime import HookDiagnostic, HookEngine
from fakuicode.instructions import InstructionLoader
from fakuicode.mcp.adapter import McpToolAdapter
from fakuicode.memory.service import MemoryService
from fakuicode.models import ProfileSet, ProviderConfig
from fakuicode.permissions.config import PermissionConfigRepository, PermissionConfigSnapshot
from fakuicode.permissions.manager import ApprovalBroker, PermissionManager
from fakuicode.permissions.safety import DangerousCommandGuard
from fakuicode.providers.base import AgentRequest, ChatProvider
from fakuicode.providers.invocation import is_agent_provider, provider_supports_system_context
from fakuicode.session import AgentSessionController, SessionController
from fakuicode.skills import IsolatedSkillExecutor, SkillDiscovery, SkillManager
from fakuicode.skills.broker import SkillTrustBroker
from fakuicode.skills.install import SkillInstaller, SkillPackageFetcher
from fakuicode.skills.install_broker import SkillInstallBroker
from fakuicode.skills.trust import SkillTrustRepository
import fakuicode.skills as skill_package
from fakuicode.storage import ConversationRecord, ConversationStore
from fakuicode.subagents import AgentCatalog, ChildRuntimeFactory, TaskManager
from fakuicode.subagents.tools import (
    AgentTool,
    SendMessageTool,
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
)
from fakuicode.teams.config import TeamFeatureConfig, coordinator_is_enabled
from fakuicode.teams.control_tools import (
    TeamFinalizePrepareTool,
    TeamFinalizeTool,
    TeamIntegrateTaskTool,
    TeamMemberAssignTool,
    TeamMemberResumeTool,
    TeamMemberStartTool,
    TeamMemberStopTool,
    TeamPlanReviewTool,
)
from fakuicode.teams.coordinator import apply_coordinator_scope
from fakuicode.teams.git import TeamGitCoordinator
from fakuicode.teams.models import TeamStatus, TeamTask
from fakuicode.teams.runtime import TeamRuntimeManager
from fakuicode.teams.service import TeamService
from fakuicode.teams.storage import TeamStore
from fakuicode.teams.tools import (
    TeamCreateTool,
    TeamInboxListTool,
    TeamMessageSendTool,
    TeamTaskCreateTool,
    TeamTaskDeleteTool,
    TeamTaskGetTool,
    TeamTaskListTool,
    TeamTaskUpdateTool,
)
from fakuicode.tool_scheduler import ReadOnlyToolScheduler
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry
from fakuicode.worktrees.manager import WorktreeManager
from fakuicode.worktrees.models import ChildExecutionContext


@dataclass
class RuntimeBundle:
    """Resources that share one session lifecycle."""

    session: SessionController | AgentSessionController
    approval_broker: ApprovalBroker | None = None
    task_manager: TaskManager | None = None
    agent_tool: AgentTool | None = None
    skill_manager: SkillManager | None = None
    command_registry: CommandRegistry = DEFAULT_COMMAND_REGISTRY
    team_feature: TeamFeatureRuntime | None = None


class TeamFeatureRuntime:
    """Own Team service/runtime wiring for one lead conversation."""

    def __init__(
        self,
        *,
        config: TeamFeatureConfig,
        home: Path | None,
        environment: Mapping[str, str],
        worktree_manager: WorktreeManager | None,
        conversation: ConversationRecord | None,
        agent_catalog: AgentCatalog,
        profile_name: str,
        on_coordinator_activated: Callable[[], None],
    ) -> None:
        self.config = config
        self.home = home
        self.environment = environment
        self.worktree_manager = worktree_manager
        self.conversation = conversation
        self.agent_catalog = agent_catalog
        self.profile_name = profile_name
        self.on_coordinator_activated = on_coordinator_activated
        self.service: TeamService | None = None
        self.runtime: TeamRuntimeManager | None = None
        self.git: TeamGitCoordinator | None = None
        self.coordinator_active = False

    def install(
        self,
        registry: ToolRegistry,
        child_runtime: ChildRuntimeFactory,
        task_manager: TaskManager,
    ) -> None:
        """Attach Team tools only when the session has a safe Git target."""

        self.coordinator_active = False
        manager = self.worktree_manager
        conversation = self.conversation
        if manager is None or conversation is None or self.home is None:
            return
        git = manager.git
        timeout = manager.limits.metadata_timeout_seconds
        branch_result = git.run(
            manager.repo_root,
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            timeout=timeout,
            check=False,
        )
        if branch_result.returncode != 0 or not branch_result.stdout:
            return
        target_sha = git.run(
            manager.repo_root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            timeout=timeout,
        ).stdout
        repository_fingerprint = sha256(
            str(manager.repo_root).casefold().encode("utf-8")
        ).hexdigest()
        service = TeamService(
            TeamStore(self.home),
            lead_conversation_id=conversation.id,
            repository_fingerprint=repository_fingerprint,
            target_branch=branch_result.stdout,
            target_sha=target_sha,
            lead_profile=self.profile_name,
        )
        git_coordinator = TeamGitCoordinator(service, manager)
        team_runtime = TeamRuntimeManager(
            service,
            self.agent_catalog,
            child_runtime,
            task_manager,
            git_coordinator=git_coordinator,
        )
        self.service = service
        self.runtime = team_runtime
        self.git = git_coordinator

        def activate_team_tools(_team: object = None) -> None:
            actor = service.actor()
            registry.permission_manager.grant_session_capability(
                actor.workflow_capability
            )

            def resume_approved(task: TeamTask) -> None:
                if not task.plan_approved or task.assignee_id is None:
                    return
                member = next(
                    item
                    for item in service.store.list_members(actor.team_id)
                    if item.member_id == task.assignee_id
                )
                team_runtime.resume_member(
                    actor,
                    member_name=member.name,
                    prompt="计划已获 Lead 批准。读取审批消息并按批准计划实施。",
                    description=f"恢复已批准任务：{task.title}",
                )

            tools = (
                TeamTaskCreateTool(service, actor),
                TeamTaskGetTool(service, actor),
                TeamTaskListTool(service, actor),
                TeamTaskUpdateTool(service, actor),
                TeamTaskDeleteTool(service, actor),
                TeamMessageSendTool(
                    service,
                    actor,
                    delivery_notifier=team_runtime.notify_messages,
                ),
                TeamInboxListTool(service, actor),
                TeamMemberStartTool(actor, team_runtime),
                TeamMemberAssignTool(actor, team_runtime),
                TeamMemberResumeTool(actor, team_runtime),
                TeamMemberStopTool(actor, team_runtime),
                TeamPlanReviewTool(
                    service,
                    actor,
                    on_reviewed=resume_approved,
                ),
                TeamIntegrateTaskTool(actor, git_coordinator),
                TeamFinalizePrepareTool(actor, git_coordinator),
                TeamFinalizeTool(actor, git_coordinator),
            )
            for tool in tools:
                if not registry.is_known(tool.definition.name):
                    registry.register(tool)
            if coordinator_is_enabled(self.config, self.environment):
                self.coordinator_active = True
                apply_coordinator_scope(
                    registry,
                    {tool.definition.name for tool in tools},
                )
                self.on_coordinator_activated()

        matching = [
            team
            for team in service.store.list_teams()
            if team.status is TeamStatus.ACTIVE
            and team.lead_conversation_id == conversation.id
            and team.repository_fingerprint == repository_fingerprint
        ]
        if matching:
            selected = max(matching, key=lambda team: team.created_at)
            service.attach_team(selected.name)
            activate_team_tools()
            return
        registry.register(TeamCreateTool(service, on_created=activate_team_tools))


@dataclass(frozen=True)
class SessionFactoryDependencies:
    workspace: Path
    config: ProviderConfig
    profiles: ProfileSet
    profile_name: str
    provider_factory: Callable[[ProviderConfig], ChatProvider]
    store: ConversationStore | None
    conversation: ConversationRecord | None
    permission_snapshot: PermissionConfigSnapshot
    permission_repository: PermissionConfigRepository | None
    hook_snapshot: HookConfigSnapshot
    hook_repository: HookConfigRepository | None
    hook_engine: HookEngine
    hook_diagnostic_sink: Callable[[HookDiagnostic], None]
    memory_service: MemoryService | None
    read_only_scheduler: ReadOnlyToolScheduler
    worktree_manager: WorktreeManager | None
    agent_catalog: AgentCatalog
    team_feature: TeamFeatureRuntime
    mcp_adapters: tuple[McpToolAdapter, ...]
    skill_user_root: Path | None
    skill_trust_repository: SkillTrustRepository | None
    skill_fetcher: SkillPackageFetcher | None
    skill_trust_broker: SkillTrustBroker
    skill_install_broker: SkillInstallBroker
    enable_subagent_background: bool
    subagent_auto_background_seconds: float
    subagent_max_concurrent: int
    project_instructions: str
    active_instructions: Callable[[], str]
    create_child_provider: Callable[[ProviderConfig], object]
    parent_request_provider: Callable[[], AgentRequest | None]
    capture_readonly_memory_snapshot: Callable[[], object | None]
    cancel_event_provider: Callable[[], Event | None]
    on_skills_changed: Callable[[], None]


class SessionFactory:
    """Build one chat or agent runtime without depending on Textual widgets."""

    def __init__(self, dependencies: SessionFactoryDependencies) -> None:
        self.dependencies = dependencies

    def create(self, provider: ChatProvider) -> RuntimeBundle:
        if not is_agent_provider(provider):
            return RuntimeBundle(
                session=SessionController(provider),
                team_feature=self.dependencies.team_feature,
            )

        dependencies = self.dependencies
        broker = ApprovalBroker()
        permissions = PermissionManager(
            dependencies.permission_snapshot,
            DangerousCommandGuard(dependencies.workspace),
            approval_handler=broker,
            repository=dependencies.permission_repository,
        )
        registry = ToolRegistry(
            WorkspacePolicy(dependencies.workspace),
            permission_manager=permissions,
            hook_engine=dependencies.hook_engine,
        )
        task_manager = TaskManager(
            max_concurrent=dependencies.subagent_max_concurrent
        )
        try:
            child_registry = self._child_registry_factory()
            child_runtime = ChildRuntimeFactory(
                store=dependencies.store,
                parent_conversation_id=(
                    dependencies.conversation.id
                    if dependencies.conversation is not None
                    else None
                ),
                workspace=dependencies.workspace,
                profiles=dependencies.profiles,
                active_profile_name=dependencies.profile_name,
                provider_factory=dependencies.create_child_provider,
                tool_registry_factory=child_registry,
                parent_permissions=permissions,
                approval_handler=broker,
                project_instructions=dependencies.project_instructions,
                parent_request_provider=dependencies.parent_request_provider,
                worktree_manager=dependencies.worktree_manager,
                project_instruction_provider=lambda child_workspace: InstructionLoader(
                    child_workspace
                ).load().text,
                memory_service=dependencies.memory_service,
                read_only_scheduler=dependencies.read_only_scheduler,
            )
            dependencies.team_feature.install(
                registry,
                child_runtime,
                task_manager,
            )
            agent_tool = AgentTool(
                dependencies.agent_catalog,
                child_runtime,
                task_manager,
                inline_timeout_seconds=dependencies.subagent_auto_background_seconds,
                background_enabled=dependencies.enable_subagent_background,
            )
            registry.register_system(agent_tool)
            registry.register_system(TaskListTool(task_manager))
            registry.register_system(TaskGetTool(task_manager))
            registry.register_system(TaskStopTool(task_manager))
            registry.register_system(SendMessageTool(task_manager))
            for adapter in dependencies.mcp_adapters:
                registry.register(adapter)

            manager = self._skill_manager(provider, registry, permissions, broker, child_registry)
            command_registry = _command_registry(manager)
            session = AgentSessionController(
                provider,  # type: ignore[arg-type]
                registry,
                store=dependencies.store,
                conversation_id=(
                    dependencies.conversation.id
                    if dependencies.conversation is not None
                    else None
                ),
                custom_instructions=dependencies.project_instructions,
                memory_service=dependencies.memory_service,
                skill_manager=manager,
                read_only_scheduler=dependencies.read_only_scheduler,
            )
            return RuntimeBundle(
                session=session,
                approval_broker=broker,
                task_manager=task_manager,
                agent_tool=agent_tool,
                skill_manager=manager,
                command_registry=command_registry,
                team_feature=dependencies.team_feature,
            )
        except BaseException:
            task_manager.close()
            raise

    def _child_registry_factory(self) -> Callable[..., ToolRegistry]:
        dependencies = self.dependencies

        def child_registry(
            child_permissions: PermissionManager,
            execution_context: ChildExecutionContext | None = None,
        ) -> ToolRegistry:
            child_workspace = (
                execution_context.execution_workspace
                if execution_context is not None
                else dependencies.workspace
            )
            child_policy = WorkspacePolicy(
                child_workspace,
                mappings=(
                    execution_context.mappings
                    if execution_context is not None
                    else ()
                ),
            )
            child_hook_rules = dependencies.hook_snapshot.rules
            if execution_context is not None:
                parent_paths = (
                    dependencies.hook_repository.paths
                    if dependencies.hook_repository is not None
                    else HookPaths.for_workspace(dependencies.workspace)
                )
                child_snapshot = HookConfigRepository(
                    HookPaths(
                        parent_paths.user,
                        child_workspace / ".fakuicode" / "hooks.yaml",
                        parent_paths.trust,
                    ),
                    child_workspace,
                    project_trusted=False,
                ).load()
                child_hook_rules = tuple(
                    rule
                    for rule in child_snapshot.rules
                    if rule not in child_snapshot.project_rules
                )
                if (
                    dependencies.hook_snapshot.project_trusted
                    and child_snapshot.project_fingerprint
                    == dependencies.hook_snapshot.project_fingerprint
                ):
                    child_hook_rules += child_snapshot.project_rules
            child_hooks = HookEngine(
                child_hook_rules,
                diagnostic_sink=dependencies.hook_diagnostic_sink,
                workspace=child_workspace,
            )
            return ToolRegistry(
                child_policy,
                permission_manager=child_permissions,
                hook_engine=child_hooks,
            )

        return child_registry

    def _skill_manager(
        self,
        provider: ChatProvider,
        registry: ToolRegistry,
        permissions: PermissionManager,
        broker: ApprovalBroker,
        child_registry: Callable[..., ToolRegistry],
    ) -> SkillManager | None:
        dependencies = self.dependencies
        if not provider_supports_system_context(provider):
            return None
        user_root = (
            dependencies.skill_user_root
            or dependencies.workspace / ".fakuicode" / "__user_skills_disabled__"
        )
        builtin_root = Path(skill_package.__file__).parent / "builtin"
        discovery = SkillDiscovery(
            dependencies.workspace / ".fakuicode" / "skills",
            user_root,
            builtin_root,
            reserved_commands=RESERVED_COMMAND_NAMES,
        )
        manager = SkillManager(
            discovery,
            registry,
            context_window=dependencies.config.context_window,
            trust_repository=dependencies.skill_trust_repository,
            trust_handler=lambda request: dependencies.skill_trust_broker.request(
                request,
                cancel_event=dependencies.cancel_event_provider(),
            ),
        )
        manager.refresh()

        def refresh_after_install() -> object:
            snapshot = manager.refresh()
            dependencies.on_skills_changed()
            return snapshot

        manager.installer = SkillInstaller(
            dependencies.workspace,
            user_root,
            fetcher=dependencies.skill_fetcher,
            refresh=refresh_after_install,
            builtin_root=builtin_root,
        )
        manager.install_confirmation = (
            lambda preview, cancel: dependencies.skill_install_broker.request(
                preview,
                cancel_event=cancel,
            )
        )

        def skill_child_registry() -> ToolRegistry:
            child_permissions = permissions.spawn_child(approval_handler=broker)
            child = child_registry(child_permissions)
            for adapter in dependencies.mcp_adapters:
                child.register(adapter)
            return child

        executor = None
        if dependencies.store is not None and dependencies.conversation is not None:
            executor = IsolatedSkillExecutor(
                store=dependencies.store,
                parent_conversation_id=dependencies.conversation.id,
                workspace=dependencies.workspace,
                profiles=dependencies.profiles,
                active_profile_name=dependencies.profile_name,
                parent_messages=lambda: manager.parent_messages,
                provider_factory=dependencies.provider_factory,
                tool_registry_factory=skill_child_registry,
                custom_instructions=dependencies.active_instructions(),
                readonly_memory_snapshot=dependencies.capture_readonly_memory_snapshot,
                read_only_scheduler=dependencies.read_only_scheduler,
            )
        if executor is not None:
            manager.isolated_runner = lambda name, arguments, cancel: executor.run(
                manager.snapshot.skills[name], arguments, cancel
            )
        return manager


def _command_registry(manager: SkillManager | None) -> CommandRegistry:
    if manager is None:
        return DEFAULT_COMMAND_REGISTRY
    return compose_command_registry(
        tuple(
            (skill.name, skill.description)
            for skill in manager.snapshot.skills.values()
        )
    )
