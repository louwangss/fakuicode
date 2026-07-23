from __future__ import annotations

from fakuicode.agent import AgentRunner
from fakuicode.hooks.models import HookEvent, HookSource, HookRule, PromptAction
from fakuicode.hooks.runtime import HookEngine
from fakuicode.context_manager import ContextPreparationResult
from fakuicode.models import AgentMessage, AgentStreamEvent


class CaptureEngine(HookEngine):
    def __init__(self, rules: tuple[HookRule, ...] = ()) -> None:
        super().__init__(rules)
        self.calls: list[tuple[HookEvent, object]] = []

    def dispatch(self, event: HookEvent, payload, *, plan_mode: bool = False):
        self.calls.append((event, payload))
        return super().dispatch(event, payload, plan_mode=plan_mode)


class RecordingEngine(CaptureEngine):
    def __init__(self) -> None:
        super().__init__(
            (
                HookRule(
                    "turn-guidance",
                    HookEvent.TURN_START,
                    PromptAction("Turn Hook sentinel."),
                    HookSource.USER,
                ),
                HookRule(
                    "request-guidance",
                    HookEvent.PRE_MODEL_REQUEST,
                    PromptAction("Request Hook sentinel."),
                    HookSource.USER,
                ),
            )
        )


class Tools:
    def __init__(self, hooks: HookEngine) -> None:
        self.hook_engine = hooks

    def begin_request(self) -> None:
        pass

    def definitions(self, *, read_only_only: bool = False):
        return []

    def is_known(self, name: str) -> bool:
        return False

    def is_read_only(self, name: str) -> bool:
        return False


class Provider:
    def __init__(self) -> None:
        self.requests = []

    def stream_agent(self, messages, tools, *, cancel_event=None, system_instruction="", request=None):
        del messages, tools, cancel_event, system_instruction
        self.requests.append(request)
        yield AgentStreamEvent("text_delta", "done")
        yield AgentStreamEvent("completed")

    def cancel(self) -> None:
        pass


def test_agent_runner_dispatches_turn_and_message_lifecycle_and_injects_prompts() -> None:
    hooks = RecordingEngine()
    provider = Provider()

    events = list(AgentRunner(provider, Tools(hooks)).run([AgentMessage("user", "work")]))

    assert events[-1].kind == "completed"
    assert hooks.calls == [
        (
            HookEvent.TURN_START,
            {"turn": {"message_count": 1, "outcome": "started"}},
        ),
        (
            HookEvent.PRE_MODEL_REQUEST,
            {"message": {"round": 1, "history_count": 1, "outcome": "pending"}},
        ),
        (
            HookEvent.POST_MODEL_RESPONSE,
            {
                "message": {
                    "round": 1,
                    "outcome": "completed",
                    "tool_call_count": 0,
                    "text": "done",
                }
            },
        ),
        (
            HookEvent.TURN_END,
            {"turn": {"message_count": 1, "outcome": "completed"}},
        ),
    ]
    assert "Turn Hook sentinel." in provider.requests[0].system_supplement
    assert "Request Hook sentinel." in provider.requests[0].system_supplement


def test_plan_mode_passes_read_only_boundary_to_lifecycle_hooks() -> None:
    calls: list[tuple[HookEvent, bool]] = []

    class Hooks(HookEngine):
        def __init__(self) -> None:
            super().__init__(())

        def dispatch(self, event: HookEvent, payload, *, plan_mode: bool = False):
            calls.append((event, plan_mode))
            return super().dispatch(event, payload, plan_mode=plan_mode)

    list(AgentRunner(Provider(), Tools(Hooks())).run([AgentMessage("user", "plan")], mode="plan"))

    assert calls
    assert all(plan_mode for _, plan_mode in calls)


def test_compaction_prompt_is_added_to_the_current_provider_request() -> None:
    hooks = HookEngine(
        (
            HookRule(
                "after-compact",
                HookEvent.POST_COMPACT,
                PromptAction("Re-read details after compaction."),
                HookSource.USER,
            ),
        )
    )
    provider = Provider()

    class Manager:
        def prepare_request(self, request):
            hooks.dispatch(
                HookEvent.POST_COMPACT,
                {"compact": {"trigger": "automatic", "outcome": "compacted"}},
            )
            return ContextPreparationResult(request)

    events = list(
        AgentRunner(provider, Tools(hooks), context_manager=Manager()).run(
            [AgentMessage("user", "work")]
        )
    )

    assert events[-1].kind == "completed"
    assert "Re-read details after compaction." in provider.requests[0].system_supplement


def test_internal_request_building_does_not_consume_pending_hook_context() -> None:
    hooks = HookEngine(
        (
            HookRule(
                "next-request",
                HookEvent.POST_MODEL_RESPONSE,
                PromptAction("Pending Hook sentinel."),
                HookSource.USER,
            ),
        )
    )
    runner = AgentRunner(Provider(), Tools(hooks))
    hooks.dispatch(HookEvent.POST_MODEL_RESPONSE, {})

    first = runner.build_request([AgentMessage("user", "inspect")])
    second = runner.build_request([AgentMessage("user", "inspect")])

    assert "Pending Hook sentinel." in first.system_supplement
    assert "Pending Hook sentinel." in second.system_supplement
    assert hooks.consume_pending_prompts() == ("Pending Hook sentinel.",)


def test_session_and_clear_lifecycle_payload_contract(tmp_path) -> None:
    from fakuicode.session import AgentSessionController
    from fakuicode.storage import ConversationStore
    from fakuicode.tools.policy import WorkspacePolicy

    hooks = CaptureEngine()
    tools = Tools(hooks)
    tools.policy = WorkspacePolicy(tmp_path)
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create_conversation("contract", tmp_path, "default")
    session = AgentSessionController(
        Provider(),
        tools,
        store=store,
        conversation_id=conversation.id,
    )

    session.clear_context()
    session.close()

    assert hooks.calls == [
        (
            HookEvent.SESSION_START,
            {
                "session": {
                    "conversation_id": conversation.id,
                    "outcome": "started",
                }
            },
        ),
        (
            HookEvent.CONTEXT_CLEARED,
            {
                "context": {
                    "conversation_id": conversation.id,
                    "outcome": "completed",
                }
            },
        ),
        (
            HookEvent.SESSION_END,
            {
                "session": {
                    "conversation_id": conversation.id,
                    "outcome": "completed",
                }
            },
        ),
    ]


def test_tool_lifecycle_payload_contract(tmp_path) -> None:
    from fakuicode.models import ToolCall
    from fakuicode.tools.policy import WorkspacePolicy
    from fakuicode.tools.registry import ToolRegistry

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.txt").write_text("content", encoding="utf-8")
    hooks = CaptureEngine()
    registry = ToolRegistry(WorkspacePolicy(workspace), hook_engine=hooks)

    result = registry.execute(ToolCall("call-1", "read_file", {"path": "value.txt"}))

    assert result.success is True
    assert hooks.calls == [
        (
            HookEvent.PRE_TOOL_USE,
            {
                "tool": {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": {"path": "value.txt"},
                    "target": "value.txt",
                    "read_only": True,
                }
            },
        ),
        (
            HookEvent.POST_TOOL_USE,
            {
                "tool": {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": {"path": "value.txt"},
                    "target": "value.txt",
                    "read_only": True,
                    "outcome": "ok",
                    "summary": "read value.txt",
                    "duration_seconds": None,
                }
            },
        ),
    ]
