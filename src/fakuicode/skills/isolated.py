"""Hidden child-conversation execution for isolated Skills."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from fakuicode.errors import RequestCancelled
from fakuicode.instructions import sanitize_instruction_metadata
from fakuicode.models import AgentMessage, ProfileSet
from fakuicode.session import AgentSessionController
from fakuicode.skills.models import SkillDefinition
from fakuicode.skills.tool import SkillScriptTool
from fakuicode.storage import ConversationStore
from fakuicode.tool_scheduler import ReadOnlyToolScheduler
from fakuicode.subagents.runtime import run_controller_to_completion
from fakuicode.tools.base import ToolExecution
from fakuicode.tools.registry import ToolRegistry


def select_recent_user_turns(messages: Sequence[AgentMessage], count: int) -> tuple[AgentMessage, ...]:
    if count <= 0:
        return ()
    turns: list[list[AgentMessage]] = []
    current: list[AgentMessage] = []
    for message in messages:
        is_user_prompt = message.role == "user" and bool(message.content) and not message.tool_results
        if is_user_prompt:
            if current:
                turns.append(current)
            current = [message]
        elif current:
            current.append(message)
    if current:
        turns.append(current)
    return tuple(message for turn in turns[-count:] for message in turn)


@dataclass
class _PromptSkillContext:
    active_prompt: str
    catalog_text: str = ""

    def set_mode(self, mode: str) -> None:
        del mode


class IsolatedSkillExecutor:
    def __init__(
        self,
        *,
        store: ConversationStore,
        parent_conversation_id: str,
        workspace: Path,
        profiles: ProfileSet,
        active_profile_name: str,
        parent_messages: Callable[[], Sequence[AgentMessage]],
        provider_factory: Callable[[object], object],
        tool_registry_factory: Callable[[], ToolRegistry],
        custom_instructions: str = "",
        readonly_memory_snapshot: object | Callable[[], object | None] | None = None,
        read_only_scheduler: ReadOnlyToolScheduler | None = None,
    ) -> None:
        self.store = store
        self.parent_conversation_id = parent_conversation_id
        self.workspace = workspace
        self.profiles = profiles
        self.active_profile_name = active_profile_name
        self.parent_messages = parent_messages
        self.provider_factory = provider_factory
        self.tool_registry_factory = tool_registry_factory
        self.custom_instructions = custom_instructions
        self.readonly_memory_snapshot = readonly_memory_snapshot
        self.read_only_scheduler = read_only_scheduler

    def run(
        self,
        skill: SkillDefinition,
        arguments: str,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        profile_name = self.active_profile_name if skill.profile == "inherit" else skill.profile
        try:
            config = self.profiles.get(profile_name)
        except KeyError:
            return ToolExecution(False, f"Skill 指定的 Profile '{profile_name}' 不存在。", "isolated profile unavailable")
        child = self.store.create_conversation(
            f"Skill: {skill.name}",
            self.workspace,
            profile_name,
            conversation_type="skill",
            parent_conversation_id=self.parent_conversation_id,
            skill_name=skill.name,
        )
        inherited = select_recent_user_turns(self.parent_messages(), skill.history_turns)
        first_sequence, last_sequence = self._parent_range(skill.history_turns)
        self.store.append_event(
            child.id,
            "system",
            "",
            metadata={
                "skill_run": skill.name,
                "parent_conversation_id": self.parent_conversation_id,
                "history_turns": skill.history_turns,
                "inherited_message_count": len(inherited),
                "parent_range": [first_sequence, last_sequence],
                "profile": profile_name,
                "status": "active",
            },
        )
        terminal = "error"
        session: AgentSessionController | None = None
        try:
            tools = self.tool_registry_factory()
            for spec in skill.tools:
                tools.register(SkillScriptTool(self.workspace, skill, spec))
            tools.set_visible_tools({*skill.visible_tools, *skill.runtime_tool_names})
            prompt = _PromptSkillContext(
                f"### Skill: {skill.name}\n"
                f"来源：{skill.source.value}；指纹：{skill.fingerprint}；"
                f"包根目录：{sanitize_instruction_metadata(str(skill.package_path.resolve()))}\n\n"
                f"{skill.render(arguments)}"
            )
            session = AgentSessionController(
                self.provider_factory(config),
                tools,
                store=self.store,
                conversation_id=child.id,
                custom_instructions=self.custom_instructions,
                skill_manager=prompt,
                readonly_memory_snapshot=(
                    self.readonly_memory_snapshot()
                    if callable(self.readonly_memory_snapshot)
                    else self.readonly_memory_snapshot
                ),
                retry_provider_errors=False,
                read_only_scheduler=self.read_only_scheduler,
            )
            for message in inherited:
                self._append_inherited_message(child.id, message)
            session.history = list(session.context_manager.active_messages())
            outcome = run_controller_to_completion(
                session,
                f"执行 Skill '{skill.name}'。参数已在 Skill 指令中提供。",
                cancel_event=cancel_event,
            )
            terminal = "error" if outcome.status == "failed" else outcome.status
            answer = outcome.text
            if terminal == "completed" and answer:
                return ToolExecution(
                    True,
                    answer,
                    f"isolated skill {skill.name} completed",
                    metadata={"child_conversation_id": child.id, "skill": skill.name, "profile": profile_name, "status": terminal},
                )
            if terminal == "completed":
                terminal = "error"
            if terminal == "cancelled":
                return ToolExecution(
                    False,
                    "独立 Skill 已取消。",
                    "isolated skill cancelled",
                    metadata={"child_conversation_id": child.id, "skill": skill.name, "profile": profile_name, "status": terminal},
                )
            return ToolExecution(
                False,
                answer or "独立 Skill 执行失败。",
                "isolated skill failed",
                metadata={"child_conversation_id": child.id, "skill": skill.name, "profile": profile_name, "status": terminal},
            )
        except RequestCancelled:
            terminal = "cancelled"
            return ToolExecution(
                False,
                "独立 Skill 已取消。",
                "isolated skill cancelled",
                metadata={"child_conversation_id": child.id, "skill": skill.name, "profile": profile_name, "status": terminal},
            )
        except Exception:
            return ToolExecution(
                False,
                "独立 Skill 执行失败。",
                "isolated skill failed",
                metadata={"child_conversation_id": child.id, "skill": skill.name, "profile": profile_name, "status": "error"},
            )
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                self.store.update_conversation_status(child.id, terminal)

    def _parent_range(self, history_turns: int) -> tuple[int, int]:
        if history_turns <= 0:
            return (0, 0)
        boundary = self.store.latest_clear_sequence(self.parent_conversation_id)
        events = self.store.load_events(self.parent_conversation_id, after_sequence=boundary)
        user_sequences = [event.sequence for event in events if event.kind == "user"]
        if not user_sequences:
            return (0, 0)
        first = user_sequences[max(0, len(user_sequences) - history_turns)]
        return (first, events[-1].sequence)

    def _append_inherited_message(self, child_id: str, message: AgentMessage) -> None:
        inherited = {"inherited_from_parent": self.parent_conversation_id}
        if message.role == "assistant":
            metadata = dict(inherited)
            if message.tool_calls:
                metadata["tool_calls"] = [
                    {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                    for call in message.tool_calls
                ]
            self.store.append_event(child_id, "assistant", message.content, metadata=metadata)
            for call in message.tool_calls:
                self.store.append_event(
                    child_id,
                    "tool_call",
                    call.name,
                    call_id=call.id,
                    metadata={**inherited, "arguments": dict(call.arguments)},
                )
            return
        if message.tool_results:
            for result in message.tool_results:
                self.store.append_event(
                    child_id,
                    "tool_result",
                    result.output,
                    call_id=result.call_id,
                    metadata={
                        **inherited,
                        "tool_name": result.tool_name,
                        "success": result.success,
                        "summary": result.summary,
                    },
                )
            return
        self.store.append_event(child_id, "user", message.content, metadata=inherited)
