"""Skill activation lifecycle and the system-level load_skill tool."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from threading import Event

from fakuicode.errors import ToolExecutionError
from fakuicode.instructions import sanitize_instruction_metadata
from fakuicode.models import AgentMessage, AgentMode, ToolDefinition
from fakuicode.skills.catalog import render_skill_catalog
from fakuicode.skills.discovery import SkillDiscovery
from fakuicode.skills.models import (
    ActiveSkill,
    SkillCatalog,
    SkillExecution,
    SkillDefinition,
    SkillInvocation,
    SkillSnapshot,
)
from fakuicode.skills.install import (
    SkillInstallDecision,
    SkillInstallError,
    SkillInstallPreview,
    SkillInstallRequest,
    SkillInstaller,
)
from fakuicode.skills.tool import SkillScriptTool
from fakuicode.skills.trust import (
    SkillTrustRepository,
    SkillTrustRequest,
    SkillTrustStorageError,
    build_skill_trust_request,
    skill_identity,
)
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments
from fakuicode.tools.registry import ToolRegistry


ActivationValidator = Callable[[tuple[ActiveSkill, ...]], None]
ActivationObserver = Callable[[ActiveSkill], None]
IsolatedRunner = Callable[[str, str, Event | None], ToolExecution]
TrustHandler = Callable[[SkillTrustRequest], bool]
InstallConfirmationHandler = Callable[[SkillInstallPreview, Event | None], SkillInstallDecision]


class SkillManager:
    def __init__(
        self,
        discovery: SkillDiscovery,
        tools: ToolRegistry,
        *,
        context_window: int,
        isolated_runner: IsolatedRunner | None = None,
        trust_repository: SkillTrustRepository | None = None,
        trust_handler: TrustHandler | None = None,
        installer: SkillInstaller | None = None,
        install_confirmation: InstallConfirmationHandler | None = None,
    ) -> None:
        self.discovery = discovery
        self.tools = tools
        self.context_window = context_window
        self.isolated_runner = isolated_runner
        self.trust_repository = trust_repository
        self.trust_handler = trust_handler
        self.installer = installer
        self.install_confirmation = install_confirmation
        self.activation_validator: ActivationValidator | None = None
        self.on_activation: ActivationObserver | None = None
        self.snapshot = SkillSnapshot({}, ())
        self._active: OrderedDict[str, ActiveSkill] = OrderedDict()
        self.mode: AgentMode = "execute"
        self.parent_messages: tuple[AgentMessage, ...] = ()
        self.tools.register_system(LoadSkillTool(self))
        self.tools.register_system(InstallSkillTool(self))

    @property
    def active(self) -> tuple[ActiveSkill, ...]:
        return tuple(self._active.values())

    @property
    def active_prompt(self) -> str:
        sections = []
        for skill in self._active.values():
            stale = "；能力包已更新，专属工具已撤销，需重新激活" if skill.stale else ""
            package_path = (
                f"；包根目录：{sanitize_instruction_metadata(str(skill.package_path))}"
                if skill.package_path is not None
                else ""
            )
            sections.append(
                f"### Skill: {skill.name}\n"
                f"来源：{skill.source.value}；指纹：{skill.fingerprint}{package_path}{stale}\n\n"
                f"{skill.rendered_body}"
            )
        return "\n\n".join(sections)

    @property
    def catalog_text(self) -> str:
        return self.catalog.text

    @property
    def catalog(self) -> SkillCatalog:
        return render_skill_catalog(self.snapshot, context_window=self.context_window)

    def refresh(self) -> SkillSnapshot:
        active_runtime_names = {
            tool_name for active in self._active.values() for tool_name in active.runtime_tool_names
        }
        snapshot = self.discovery.refresh(set(self.tools.all_names()).difference(active_runtime_names))
        updated: OrderedDict[str, ActiveSkill] = OrderedDict()
        for name, active in self._active.items():
            current = snapshot.skills.get(name)
            if current is None or current.fingerprint != active.fingerprint:
                for tool_name in active.runtime_tool_names:
                    if tool_name in self.tools.all_names():
                        self.tools.unregister(tool_name)
                updated[name] = ActiveSkill(
                    active.name,
                    active.rendered_body,
                    active.source,
                    active.arguments,
                    active.fingerprint,
                    active.visible_tools,
                    active.runtime_tool_names,
                    True,
                    active.package_path,
                )
            else:
                updated[name] = active
        self.snapshot = snapshot
        self._active = updated
        self._apply_visibility()
        return snapshot

    def invoke(
        self,
        name: str,
        arguments: str | None,
        *,
        model_initiated: bool = False,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        skill = self.snapshot.skills.get(name)
        if skill is None:
            return ToolExecution(False, f"Skill '{name}' 不存在或当前不可用。", "skill unavailable")
        if model_initiated and skill.invocation is SkillInvocation.MANUAL:
            return ToolExecution(False, f"Skill '{name}' is manual and cannot be loaded by the model.", "manual skill refused")
        if skill.execution is SkillExecution.ISOLATED:
            if self.mode == "plan":
                return ToolExecution(False, "计划模式不能启动独立 Skill。", "isolated skill blocked in plan mode")
            if self.isolated_runner is None:
                return ToolExecution(False, "独立 Skill 运行器尚未配置。", "isolated skill unavailable")
            trust_result = self._ensure_trusted(skill)
            if trust_result is not None:
                return trust_result
            return self.isolated_runner(name, arguments or "", cancel_event)

        candidate = ActiveSkill(
            skill.name,
            skill.render(arguments),
            skill.source,
            arguments or "",
            skill.fingerprint,
            skill.visible_tools,
            skill.runtime_tool_names,
            False,
            skill.package_path.resolve(),
        )
        prospective = OrderedDict(self._active)
        prospective[skill.name] = candidate
        try:
            if self.activation_validator is not None:
                self.activation_validator(tuple(prospective.values()))
        except ValueError as error:
            return ToolExecution(False, f"Skill 激活失败：{error}", "skill activation rejected")
        trust_result = self._ensure_trusted(skill)
        if trust_result is not None:
            return trust_result
        registered: list[str] = []
        try:
            for spec in skill.tools:
                runtime_name = f"skill__{skill.name}__{spec.name}"
                if runtime_name in self.tools.all_names():
                    continue
                self.tools.register(SkillScriptTool(self.tools.policy.workspace, skill, spec))
                registered.append(runtime_name)
        except Exception:
            for runtime_name in registered:
                self.tools.unregister(runtime_name)
            return ToolExecution(False, "Skill 专属工具注册失败。", "skill tool registration failed")
        try:
            if self.on_activation is not None:
                self.on_activation(candidate)
        except Exception:
            for runtime_name in registered:
                self.tools.unregister(runtime_name)
            return ToolExecution(False, "Skill 激活状态保存失败。", "skill activation persistence failed")
        self._active = prospective
        self._apply_visibility()
        return ToolExecution(True, f"Skill '{name}' 已激活。", f"activated skill {name}")

    def install(
        self,
        request: SkillInstallRequest,
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        if self.mode == "plan":
            return ToolExecution(False, "计划模式不能安装 Skill。", "skill installation blocked in plan mode")
        if self.installer is None or self.install_confirmation is None:
            return ToolExecution(False, "Skill 安装器尚未配置。", "skill installer unavailable")
        try:
            result = self.installer.install(
                request,
                confirm=lambda preview: self.install_confirmation(preview, cancel_event),
                cancel_event=cancel_event,
            )
        except SkillInstallError as error:
            return ToolExecution(False, f"Skill 安装失败：{error}", "skill installation failed")
        metadata = None
        if result.target_path is not None:
            metadata = {"skill": result.name, "path": str(result.target_path)}
        return ToolExecution(
            result.success,
            result.output,
            f"installed skill {result.name}" if result.success else "skill installation cancelled",
            metadata=metadata,
        )

    def clear(self) -> None:
        for skill in self._active.values():
            for tool_name in skill.runtime_tool_names:
                if tool_name in self.tools.all_names():
                    self.tools.unregister(tool_name)
        self._active.clear()
        self.tools.set_visible_tools(None)

    def close(self) -> None:
        close = getattr(self.installer, "close", None)
        if callable(close):
            close()

    def restore(self, active: tuple[ActiveSkill, ...]) -> None:
        restored: OrderedDict[str, ActiveSkill] = OrderedDict()
        registered: list[str] = []
        try:
            for item in active:
                current = self.snapshot.skills.get(item.name)
                stale = current is None or current.fingerprint != item.fingerprint
                restored[item.name] = ActiveSkill(
                    item.name,
                    item.rendered_body,
                    item.source,
                    item.arguments,
                    item.fingerprint,
                    item.visible_tools,
                    item.runtime_tool_names,
                    stale,
                    item.package_path or (current.package_path.resolve() if current is not None else None),
                )
                if not stale and current is not None and self._is_pretrusted(current):
                    for spec in current.tools:
                        runtime_name = f"skill__{current.name}__{spec.name}"
                        if runtime_name not in self.tools.all_names():
                            self.tools.register(SkillScriptTool(self.tools.policy.workspace, current, spec))
                            registered.append(runtime_name)
                elif not stale and current is not None and current.tools:
                    restored[item.name] = ActiveSkill(
                        item.name,
                        item.rendered_body,
                        item.source,
                        item.arguments,
                        item.fingerprint,
                        item.visible_tools,
                        item.runtime_tool_names,
                        True,
                        item.package_path or current.package_path.resolve(),
                    )
        except Exception:
            for runtime_name in registered:
                self.tools.unregister(runtime_name)
            raise
        self._active = restored
        self._apply_visibility()

    def set_mode(self, mode: AgentMode) -> None:
        self.mode = mode

    def set_parent_messages(self, messages: Sequence[AgentMessage]) -> None:
        self.parent_messages = tuple(messages)

    def _apply_visibility(self) -> None:
        if not self._active:
            self.tools.set_visible_tools(None)
            return
        visible: set[str] = set()
        for skill in self._active.values():
            visible.update(skill.visible_tools)
            if not skill.stale:
                visible.update(skill.runtime_tool_names)
        self.tools.set_visible_tools(visible)

    def _ensure_trusted(self, skill: SkillDefinition) -> ToolExecution | None:
        if not skill.tools or skill.source.value == "builtin" or (
            skill.source.value == "user" and skill.install_receipt is None
        ):
            return None
        if self.trust_repository is None:
            return ToolExecution(False, "项目 Skill 脚本缺少信任存储。", "skill trust unavailable")
        identity = skill_identity(self.tools.policy.workspace, skill)
        if self.trust_repository.is_trusted(identity):
            return None
        if self.trust_repository.diagnostic is not None:
            return ToolExecution(False, self.trust_repository.diagnostic, "skill trust unavailable")
        request = build_skill_trust_request(skill)
        if self.trust_handler is None or not self.trust_handler(request):
            return ToolExecution(False, "项目 Skill 脚本未获信任。", "skill trust denied")
        try:
            self.trust_repository.approve(identity)
        except SkillTrustStorageError:
            return ToolExecution(False, "Skill 信任记录保存失败。", "skill trust unavailable")
        return None

    def _is_pretrusted(self, skill: SkillDefinition) -> bool:
        if not skill.tools or skill.source.value == "builtin" or (
            skill.source.value == "user" and skill.install_receipt is None
        ):
            return True
        if self.trust_repository is None:
            return False
        return self.trust_repository.is_trusted(
            skill_identity(self.tools.policy.workspace, skill)
        )


class LoadSkillTool:
    def __init__(self, manager: SkillManager) -> None:
        self.manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "load_skill",
            "按名称加载一个可自动调用的 Skill；共享 Skill 会固定其完整 SOP，独立 Skill 会在隐藏子会话运行。",
            {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "string"},
                },
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        # Host-control only; Plan safety is enforced by SkillManager and the visible tool intersection.
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if not set(arguments).issubset({"name", "arguments"}) or "name" not in arguments:
            raise ToolExecutionError("load_skill requires name and optional arguments.")
        name = arguments.get("name")
        value = arguments.get("arguments", "")
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ToolExecutionError("load_skill arguments are invalid.")
        return ToolPreparation(freeze_arguments({"name": name, "arguments": value}), name)

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        return self.manager.invoke(
            str(arguments["name"]),
            str(arguments["arguments"]),
            model_initiated=True,
            cancel_event=cancel_event,
        )


class InstallSkillTool:
    def __init__(self, manager: SkillManager) -> None:
        self.manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "install_skill",
            "仅当用户明确要求安装公共 Skill 时使用。解析受支持的 skills.sh 或 GitHub HTTPS 来源，展示宿主确认后安装；不得改用 curl、npx 或 run_command 下载 Skill。",
            {
                "type": "object",
                "required": ["source"],
                "properties": {
                    "source": {"type": "string"},
                    "skill": {"type": "string"},
                    "scope": {"type": "string", "enum": ["project", "user"]},
                    "preset": {
                        "type": "string",
                        "enum": ["instruction", "read-only", "coding"],
                    },
                    "replace": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        from fakuicode.skills.install import SkillInstallPreset, SkillInstallScope

        allowed = {"source", "skill", "scope", "preset", "replace"}
        if set(arguments).difference(allowed) or "source" not in arguments:
            raise ToolExecutionError("install_skill arguments are invalid.")
        source = arguments.get("source")
        skill = arguments.get("skill")
        scope = arguments.get("scope", "project")
        preset = arguments.get("preset")
        replace = arguments.get("replace", False)
        if (
            not isinstance(source, str)
            or not source.strip()
            or (skill is not None and (not isinstance(skill, str) or not skill))
            or not isinstance(scope, str)
            or (preset is not None and not isinstance(preset, str))
            or not isinstance(replace, bool)
        ):
            raise ToolExecutionError("install_skill arguments are invalid.")
        try:
            normalized_scope = SkillInstallScope(scope)
            normalized_preset = SkillInstallPreset(preset) if preset is not None else None
        except ValueError as error:
            raise ToolExecutionError("install_skill arguments are invalid.") from error
        request = SkillInstallRequest(
            source.strip(),
            skill,
            normalized_scope,
            normalized_preset,
            replace,
        )
        return ToolPreparation(freeze_arguments({"request": request}), source.strip())

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        request = arguments.get("request")
        if not isinstance(request, SkillInstallRequest):
            raise ToolExecutionError("install_skill arguments are invalid.")
        return self.manager.install(request, cancel_event=cancel_event)
