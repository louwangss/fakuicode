from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from io import StringIO
from pathlib import Path
import subprocess
from threading import Event
from time import sleep

import pytest
from rich.console import Console
from textual.widgets import Collapsible, Footer, Header, Markdown, OptionList, Static


def render_plain(renderable: object, *, width: int = 120) -> str:
    console = Console(width=width, record=True, file=StringIO())
    console.print(renderable)
    return console.export_text()


class FakeProvider:
    def __init__(self, responses: list[list[object] | Exception]) -> None:
        self.responses = responses
        self.calls: list[Sequence[object]] = []

    def stream_chat(self, messages: Sequence[object]) -> Iterator[object]:
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        yield from response


class BlockingProvider:
    """Keeps its worker active until the test explicitly releases it."""

    def __init__(self) -> None:
        self.calls: list[Sequence[object]] = []
        self.started = Event()
        self.release = Event()

    def stream_chat(self, messages: Sequence[object]) -> Iterator[object]:
        from fakuicode.models import StreamEvent

        self.calls.append(messages)
        self.started.set()
        self.release.wait()
        yield StreamEvent("text_delta", "done")
        yield StreamEvent("completed")


class PausedThinkingProvider:
    """Streams extra thinking only after the test expands its panel."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def stream_chat(self, messages: Sequence[object]) -> Iterator[object]:
        from fakuicode.models import StreamEvent

        yield StreamEvent("thinking_start")
        yield StreamEvent("thinking_delta", "initial reasoning " * 4)
        self.started.set()
        self.release.wait()
        yield StreamEvent("thinking_delta", "more reasoning " * 100)
        yield StreamEvent("thinking_end")
        yield StreamEvent("text_delta", "answer")
        yield StreamEvent("completed")


class PausedTextProvider:
    """Produces a tall response, then keeps the stream explicitly open."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def stream_chat(self, messages: Sequence[object]) -> Iterator[object]:
        from fakuicode.models import StreamEvent

        for index in range(120):
            yield StreamEvent("text_delta", f"line {index}\n")
        self.started.set()
        self.release.wait()
        yield StreamEvent("completed")


class SlowTextProvider:
    """Emits text gradually to exercise live layout changes during a stream."""

    def __init__(self) -> None:
        self.started = Event()
        self.finished = Event()

    def stream_chat(self, messages: Sequence[object]) -> Iterator[object]:
        from fakuicode.models import StreamEvent

        for index in range(120):
            yield StreamEvent("text_delta", f"line {index}\n")
            if index == 40:
                self.started.set()
            sleep(0.01)
        self.finished.set()
        yield StreamEvent("completed")


class ControlledTextProvider:
    """Pauses between streamed batches so scroll ownership can be verified."""

    def __init__(self) -> None:
        self.initial_ready = Event()
        self.second_batch_ready = Event()
        self.release_second_batch = Event()
        self.release_final_batch = Event()

    def stream_chat(self, messages: Sequence[object]) -> Iterator[object]:
        from fakuicode.models import StreamEvent

        for index in range(120):
            yield StreamEvent("text_delta", f"line {index}\n")
        self.initial_ready.set()
        self.release_second_batch.wait()

        for index in range(120, 180):
            yield StreamEvent("text_delta", f"line {index}\n")
        self.second_batch_ready.set()
        self.release_final_batch.wait()

        for index in range(180, 240):
            yield StreamEvent("text_delta", f"line {index}\n")
        yield StreamEvent("completed")


class AgentTextProvider:
    """Minimal native-tool provider used to verify the TUI chooses the agent path."""

    def __init__(self) -> None:
        self.calls: list[Sequence[object]] = []
        self.requests: list[object] = []

    def stream_agent(
        self,
        messages: Sequence[object],
        tools: Sequence[object],
        *,
        cancel_event: Event | None = None,
        request: object | None = None,
    ) -> Iterator[object]:
        from fakuicode.models import AgentStreamEvent

        self.calls.append(messages)
        self.requests.append(request)
        yield AgentStreamEvent("text_delta", "agent answer")
        yield AgentStreamEvent("completed")


class RequestCaptureAgentProvider:
    def __init__(self) -> None:
        self.config = make_config()
        self.requests: list[object] = []

    def stream_agent(
        self,
        messages: Sequence[object],
        tools: Sequence[object],
        *,
        request: object | None = None,
        cancel_event: Event | None = None,
    ) -> Iterator[object]:
        from fakuicode.models import AgentStreamEvent

        del messages, tools, cancel_event
        self.requests.append(request)
        yield AgentStreamEvent("text_delta", "fresh answer")
        yield AgentStreamEvent("completed")


class FakeSkillFetcher:
    def fetch(self, source: object, *, cancel_event: Event | None = None) -> object:
        from fakuicode.skills.install import RemoteSkillPackage

        del cancel_event
        return RemoteSkillPackage(
            source=source,
            name="frontend-design",
            revision="a" * 40,
            skill_path="skills/frontend-design",
            files={
                "SKILL.md": (
                    "---\n"
                    "name: frontend-design\n"
                    "description: Build distinctive frontend interfaces\n"
                    "license: Complete terms in LICENSE.txt\n"
                    "---\n"
                    "Build a polished interface for $ARGUMENTS.\n"
                ).encode(),
                "LICENSE.txt": b"upstream license\n",
            },
        )


class InstallSkillAgentProvider:
    def __init__(self) -> None:
        self.calls: list[Sequence[object]] = []
        self.tool_sets: list[Sequence[object]] = []

    def stream_agent(
        self,
        messages: Sequence[object],
        tools: Sequence[object],
        *,
        cancel_event: Event | None = None,
        request: object | None = None,
    ) -> Iterator[object]:
        from fakuicode.models import AgentStreamEvent, ToolCall

        del cancel_event, request
        self.calls.append(messages)
        self.tool_sets.append(tools)
        if len(self.calls) == 1:
            yield AgentStreamEvent(
                "tool_call",
                tool_call=ToolCall(
                    "install-1",
                    "install_skill",
                    {"source": "https://www.skills.sh/anthropics/skills/frontend-design"},
                ),
            )
            yield AgentStreamEvent("completed")
            return
        assert messages[-1].tool_results[0].success is True
        yield AgentStreamEvent("text_delta", "frontend-design 已安装，可通过 /frontend-design 使用。")
        yield AgentStreamEvent("completed")


class FakeMemoryService:
    """Application-level memory double with only safe, public state."""

    def __init__(self) -> None:
        self.enabled = True
        self.notice_needed = True
        self.confirmed = 0
        self.closed: list[bool] = []
        self.forgotten: list[str] = []
        self.diagnostic_codes: tuple[str, ...] = ()

    def first_notice_needed(self) -> bool:
        return self.enabled and self.notice_needed

    def confirm_first_notice(self) -> None:
        self.confirmed += 1
        self.notice_needed = False

    def set_enabled(self, enabled: bool) -> object:
        self.enabled = enabled
        return object()

    def capture_turn_context(self, *, reminder: str = "") -> object:
        from fakuicode.memory.models import AgentTurnContext

        return AgentTurnContext(first_request_reminder=reminder, settings_generation=1)

    def detail_tool(self, snapshot: object) -> None:
        del snapshot
        return None

    def schedule_completed_turn(self, turn: object, snapshot: object) -> bool:
        del turn, snapshot
        return False

    def status(self) -> object:
        from fakuicode.memory.service import MemoryStatus

        return MemoryStatus(
            self.enabled,
            1,
            2,
            3,
            4,
            ("- [user_preference] concise replies (11111111-1111-4111-8111-111111111111)",),
            "committed",
            "2026-07-21T10:00:00Z",
        )

    def list_visible_entries(self) -> tuple[object, ...]:
        from fakuicode.memory.service import MemoryListItem

        return (
            MemoryListItem(
                "3f67a8d1-3853-4e09-989a-934cbf641629",
                "user",
                "user_preference",
                "concise replies",
            ),
        )

    def forget(self, entry_id: str) -> object:
        from fakuicode.memory.models import CommitResult

        self.forgotten.append(entry_id)
        return CommitResult(entry_id.endswith("9"), "committed" if entry_id.endswith("9") else "not_found")

    def consume_diagnostic_codes(self) -> tuple[str, ...]:
        codes = self.diagnostic_codes
        self.diagnostic_codes = ()
        return codes

    def close(self, *, wait: bool = False) -> None:
        self.closed.append(wait)


def make_config() -> object:
    from fakuicode.models import ProviderConfig

    return ProviderConfig("anthropic", "claude-test", "https://api.example.test/v1", "test-key")


def make_git_repository(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    for args in (
        ("init",),
        ("config", "user.name", "Team Test"),
        ("config", "user.email", "team@example.test"),
    ):
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return repo


def test_app_assembles_sessions_through_an_explicit_runtime_bundle(tmp_path: Path) -> None:
    from fakuicode.session import AgentSessionController, SessionController
    from fakuicode.tui.app import FakuicodeApp
    from fakuicode.tui.runtime import RuntimeBundle, TeamFeatureRuntime

    agent_app = FakuicodeApp(
        make_config(),
        provider=AgentTextProvider(),
        workspace=tmp_path,
    )

    assert isinstance(agent_app._runtime_bundle, RuntimeBundle)
    assert isinstance(agent_app._team_feature, TeamFeatureRuntime)
    assert isinstance(agent_app.session, AgentSessionController)
    assert agent_app._runtime_bundle.session is agent_app.session
    assert agent_app._runtime_bundle.task_manager is agent_app._task_manager
    agent_app._close_agent_session()

    chat_app = FakuicodeApp(
        make_config(),
        provider=FakeProvider([]),
        workspace=tmp_path,
    )

    assert isinstance(chat_app._runtime_bundle, RuntimeBundle)
    assert isinstance(chat_app.session, SessionController)
    assert chat_app._runtime_bundle.task_manager is None


def test_team_create_tool_is_wired_only_with_explicit_persistence_root(
    tmp_path: Path,
) -> None:
    from fakuicode.storage import ConversationStore
    from fakuicode.tui.app import FakuicodeApp

    repo = make_git_repository(tmp_path)
    app = FakuicodeApp(
        make_config(),
        provider=AgentTextProvider(),
        store=ConversationStore(tmp_path / "history.sqlite3"),
        workspace=repo,
        team_home=tmp_path / "teams",
    )

    assert app.session.runner.tools.is_known("team_create")
    assert app._team_service is not None


def test_team_activation_grants_only_the_current_workflow_capability(
    tmp_path: Path,
) -> None:
    from fakuicode.storage import ConversationStore
    from fakuicode.tui.app import FakuicodeApp
    from fakuicode.models import ToolCall

    repo = make_git_repository(tmp_path)
    app = FakuicodeApp(
        make_config(),
        provider=AgentTextProvider(),
        store=ConversationStore(tmp_path / "history.sqlite3"),
        workspace=repo,
        team_home=tmp_path / "teams",
    )
    registry = app.session.runner.tools

    created = registry._tools["team_create"].execute({"name": "alpha"})
    actor = app._team_service.actor()

    assert created.success is True
    assert registry.permission_manager.session_capabilities == (
        actor.workflow_capability,
    )
    cancelled = Event()
    cancelled.set()
    task_result = registry.execute(
        ToolCall(
            "task-create",
            "team_task_create",
            {"title": "读取文档", "description": "", "kind": "read_only"},
        ),
        cancel_event=cancelled,
    )
    finalize_result = registry.execute(
        ToolCall(
            "finalize",
            "team_finalize",
            {"confirmation_token": "untrusted-model-token"},
        ),
        cancel_event=cancelled,
    )

    assert task_result.success is True
    assert finalize_result.success is False
    assert finalize_result.summary == "permission denied"


def test_coordinator_dual_gate_applies_ceiling_and_instructions(
    tmp_path: Path,
) -> None:
    from fakuicode.storage import ConversationStore
    from fakuicode.teams.config import TeamFeatureConfig
    from fakuicode.tui.app import FakuicodeApp

    repo = make_git_repository(tmp_path)
    app = FakuicodeApp(
        make_config(),
        provider=AgentTextProvider(),
        store=ConversationStore(tmp_path / "history.sqlite3"),
        workspace=repo,
        team_home=tmp_path / "teams",
        team_config=TeamFeatureConfig(coordinator_enabled=True),
        team_environment={"FAKUICODE_COORDINATOR": "1"},
    )
    registry = app.session.runner.tools

    result = registry._tools["team_create"].execute({"name": "alpha"})

    assert result.success is True
    names = {definition.name for definition in registry.definitions()}
    assert "team_member_start" in names
    assert "write_file" not in names
    assert "run_command" not in names
    assert "agent" not in names
    assert "Team Coordinator 模式" in app.session.runner.custom_instructions


def test_app_lifecycle_hook_payload_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        from fakuicode.hooks.models import HookEvent
        from fakuicode.hooks.runtime import HookEngine
        from fakuicode.tui.app import FakuicodeApp

        calls: list[tuple[HookEvent, object]] = []
        original_dispatch = HookEngine.dispatch

        def capture_dispatch(self, event, payload, *, plan_mode=False):
            calls.append((event, payload))
            return original_dispatch(self, event, payload, plan_mode=plan_mode)

        monkeypatch.setattr(HookEngine, "dispatch", capture_dispatch)
        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            workspace=tmp_path,
        )

        async with app.run_test():
            pass

        app_calls = [
            call
            for call in calls
            if call[0] in {HookEvent.APP_START, HookEvent.APP_STOP}
        ]
        assert app_calls == [
            (
                HookEvent.APP_START,
                {
                    "app": {
                        "workspace": str(tmp_path.resolve()),
                        "outcome": "started",
                    }
                },
            ),
            (
                HookEvent.APP_STOP,
                {
                    "app": {
                        "workspace": str(tmp_path.resolve()),
                        "outcome": "completed",
                    }
                },
            ),
        ]

    asyncio.run(run())


def test_memory_service_lifecycle_shows_first_notice_once_and_closes_nonblocking() -> None:
    async def run() -> None:
        from fakuicode.session import AgentSessionController
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import SystemNotice

        memory = FakeMemoryService()
        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), memory_service=memory)
        assert isinstance(app.session, AgentSessionController)
        assert app.session.memory_service is memory

        async with app.run_test():
            notices = "\n".join(item.render().plain for item in app.query(SystemNotice))
            assert "current model" in notices
            assert "saved locally" in notices
            assert "/memory off" in notices
            assert memory.confirmed == 1
            app._new_conversation()
            assert app.session.memory_service is memory
            assert memory.closed == []

        assert memory.closed == [False]

        second = FakuicodeApp(make_config(), provider=AgentTextProvider(), memory_service=memory)
        async with second.run_test():
            notices = "\n".join(item.render().plain for item in second.query(SystemNotice))
            assert "/memory off" not in notices

    asyncio.run(run())


def test_existing_memory_is_reinjected_after_off_new_on_new_flow(tmp_path: Path) -> None:
    async def run() -> None:
        from uuid import uuid4

        from fakuicode.memory.content_policy import serialize_entry
        from fakuicode.memory.identity import MemoryPaths, MemoryRegistry, ProjectIdentityResolver
        from fakuicode.memory.models import MemoryEntry, MemoryScopeRef, MemorySourceRef
        from fakuicode.memory.repository import MemoryRepository
        from fakuicode.memory.service import MemoryService
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        paths = MemoryPaths.from_home(tmp_path / "home")
        registry = MemoryRegistry(paths)
        repository = MemoryRepository(paths, registry)
        entry = MemoryEntry(
            str(uuid4()),
            "user",
            "user_preference",
            "所有项目默认使用简体中文",
            "用户希望在所有项目中默认使用简体中文交流。",
            "2026-07-22T00:00:00Z",
            "2026-07-22T00:00:00Z",
            (MemorySourceRef("11111111-1111-4111-8111-111111111111", 1, "user_turn"),),
        )
        notes = repository.scope_path(MemoryScopeRef("user")) / "notes"
        notes.mkdir(parents=True)
        (notes / f"{entry.id}.md").write_bytes(serialize_entry(entry))
        memory = MemoryService(
            workspace,
            registry,
            ProjectIdentityResolver(registry),
            repository,
        )
        provider = RequestCaptureAgentProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            workspace=workspace,
            memory_service=memory,
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            for text in (
                "/memory off",
                "/new",
                "我的跨项目交流偏好是什么？",
                "/memory on",
                "/new",
                "我的跨项目交流偏好是什么？",
            ):
                editor.text = text
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if app._active_turn is None and not editor.disabled:
                        break

        assert len(provider.requests) == 2
        disabled_request, enabled_request = provider.requests
        assert "所有项目默认使用简体中文" not in disabled_request.system_supplement
        assert "所有项目默认使用简体中文" in enabled_request.system_supplement
        assert "read_memory_entry" not in {tool.name for tool in disabled_request.tools}
        assert "read_memory_entry" in {tool.name for tool in enabled_request.tools}

    asyncio.run(run())


def test_memory_command_status_toggle_and_precise_forget() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        memory = FakeMemoryService()
        memory.notice_needed = False
        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), memory_service=memory)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            for command in (
                "/memory",
                "/memory off",
                "/memory on",
                "/memory forget 3f67a8d1-3853-4e09-989a-934cbf641629",
                "/memory forget 3f67a8d1-3853-4e09-989a-934cbf641628",
            ):
                editor.text = command
                await pilot.press("enter")
                await pilot.pause()

            rendered = "\n".join(item.render().plain for item in app.query(SystemNotice))
            assert "Memory: on" in rendered
            assert "user 2" in rendered and "project 3" in rendered
            assert "other projects 4" in rendered
            assert "concise replies" in rendered and "committed" in rendered
            assert "Automatic memory disabled" in rendered
            assert "Automatic memory enabled" in rendered
            assert "Memory entry forgotten" in rendered
            assert "Memory entry was not found" in rendered
            assert memory.forgotten == [
                "3f67a8d1-3853-4e09-989a-934cbf641629",
                "3f67a8d1-3853-4e09-989a-934cbf641628",
            ]

    asyncio.run(run())


def test_memory_forget_without_id_uses_a_local_picker_and_confirmation() -> None:
    async def run() -> None:
        from textual.widgets import OptionList

        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import ConfirmationScreen, MemoryPicker
        from fakuicode.tui.widgets import PromptEditor

        memory = FakeMemoryService()
        memory.notice_needed = False
        provider = AgentTextProvider()
        app = FakuicodeApp(make_config(), provider=provider, memory_service=memory)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/memory forget"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, MemoryPicker)
            options = app.screen.query_one(OptionList)
            assert "concise replies" in str(options.get_option_at_index(0).prompt)
            assert "3f67a8d1" not in str(options.get_option_at_index(0).prompt)
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, ConfirmationScreen)
            assert memory.forgotten == []
            await pilot.press("enter")
            await pilot.pause()
            assert memory.forgotten == []

            editor.text = "/memory forget"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, MemoryPicker)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmationScreen)
            await pilot.press("down", "enter")
            await pilot.pause()

            assert memory.forgotten == ["3f67a8d1-3853-4e09-989a-934cbf641629"]
            assert provider.calls == []

    asyncio.run(run())


def test_memory_diagnostic_is_one_safe_notice_and_normal_reply_still_finishes() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        memory = FakeMemoryService()
        memory.notice_needed = False
        memory.diagnostic_codes = ("invalid_entry", "storage_failure", "invalid_entry")
        provider = AgentTextProvider()
        app = FakuicodeApp(make_config(), provider=provider, memory_service=memory)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "continue normally"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if provider.calls and app._active_turn is None:
                    break

            notices = [item.render().plain for item in app.query(SystemNotice)]
            memory_notices = [item for item in notices if item.startswith("Automatic memory warning:")]
            assert len(memory_notices) == 1
            assert "invalid_entry" in memory_notices[0]
            assert "storage_failure" in memory_notices[0]
            assert "Traceback" not in memory_notices[0]
            assert provider.calls

    asyncio.run(run())


def test_memory_failure_diagnostic_consumer_does_not_spam_the_timer() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import SystemNotice

        class BrokenDiagnostics(FakeMemoryService):
            def consume_diagnostic_codes(self) -> tuple[str, ...]:
                raise RuntimeError("private failure detail")

        memory = BrokenDiagnostics()
        memory.notice_needed = False
        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), memory_service=memory)
        async with app.run_test() as pilot:
            for _ in range(10):
                await pilot.pause()
            notices = [item.render().plain for item in app.query(SystemNotice)]
            warning = [item for item in notices if item.startswith("Automatic memory warning:")]
            assert warning == ["Automatic memory warning: unavailable."]
            assert "private failure detail" not in "\n".join(notices)

    asyncio.run(run())


def test_resume_gap_shows_and_injects_one_nonpersistent_time_span_reminder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        from dataclasses import replace

        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        now = 1_800_000_000_000_000_000
        store = ConversationStore(tmp_path / "history.sqlite3")
        previous = store.create_conversation("Old work", tmp_path, "default")
        store.append_event(previous.id, "user", "old question")
        store.append_event(previous.id, "assistant", "old answer")
        original_list = store.list_conversations

        def old_conversations(*, workspace: Path | None = None) -> list[object]:
            return [
                replace(item, updated_at=now - 2 * 24 * 60 * 60 * 1_000_000_000)
                if item.id == previous.id
                else item
                for item in original_list(workspace=workspace)
            ]

        monkeypatch.setattr(store, "list_conversations", old_conversations)
        provider = RequestCaptureAgentProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
            clock_ns=lambda: now,
        )

        async with app.run_test() as pilot:
            app._resume_conversation(previous.id)
            for _ in range(20):
                await pilot.pause()
                notices = [item.render().plain for item in app.query(SystemNotice)]
                if any("inactive for about 2 days" in item for item in notices):
                    break
            notices = [item.render().plain for item in app.query(SystemNotice)]
            assert sum("inactive for about 2 days" in item for item in notices) == 1

            editor = app.query_one(PromptEditor)
            for prompt in ("verify current state", "next turn"):
                editor.text = prompt
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if len(provider.requests) >= (1 if prompt == "verify current state" else 2) and app._active_turn is None:
                        break

            first_request, second_request = provider.requests[-2:]
            assert "会话已中断约 2 days" in first_request.system_supplement
            assert "必须重新验证关键事实" in first_request.system_supplement
            assert "会话已中断" not in second_request.system_supplement
            persisted = store.load_events(previous.id)
            assert all("会话已中断" not in event.content for event in persisted)
            assert all("inactive for about" not in event.content for event in persisted)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("updated_at", "now"),
    [
        (1_000, 1_000 + 24 * 60 * 60 * 1_000_000_000 - 1),
        (2_000, 1_000),
        (None, 1_000),
        (-1, 1_000),
        (1_000, 1 << 63),
    ],
)
def test_resume_gap_skip_invalid_or_unrelated_times(updated_at: object, now: object) -> None:
    from fakuicode.tui.app import _build_resume_gap_reminder

    assert _build_resume_gap_reminder(updated_at, now) is None


def test_pending_stream_follow_is_safe_after_app_unmounts() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp

        app = FakuicodeApp(make_config(), provider=AgentTextProvider())
        async with app.run_test():
            app._follow_stream = True
            app._stream_follow_scheduled = True

        app._follow_conversation_end()

        assert app._stream_follow_scheduled is False

    asyncio.run(run())


def test_app_renders_chrome_streaming_thinking_and_final_markdown() -> None:
    async def run() -> None:
        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import BrandPanel, PromptEditor, PromptPanel

        provider = FakeProvider(
            [
                [
                    StreamEvent("thinking_start"),
                    StreamEvent("thinking_delta", "reason"),
                    StreamEvent("thinking_end"),
                    StreamEvent("text_delta", "- item\n\n```python\nprint('ok')\n```"),
                    StreamEvent("completed"),
                ]
            ]
        )
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test() as pilot:
            assert list(app.query(Header)) == []
            assert list(app.query(Footer)) == []
            brand = app.query_one(BrandPanel)
            prompt_panel = app.query_one(PromptPanel)
            brand_text = render_plain(brand.render())
            status = app.query_one("#status", Static)
            footer_model = app.query_one("#footer-model", Static).render().plain
            assert "Fakuicode v0.1.0" in brand_text
            assert "claude-test" in brand_text
            assert "ANTHROPIC" not in brand_text
            assert str(Path.cwd()) in brand_text
            assert status.parent is prompt_panel.query_one("#prompt-info")
            assert status.render().plain == "[DEFAULT] Ready"
            assert footer_model == "claude-test"
            assert app.query_one("#footer-model", Static).parent is status.parent
            assert list(app.query("#session-footer")) == []
            assert "MCP" not in brand_text + footer_model
            assert "tool" not in (brand_text + footer_model).lower()
            assert "test-key" not in brand_text + footer_model
            assert list(app.query("#shortcut-hint")) == []
            assert list(app.query("#insert-newline")) == []

            editor = app.query_one(PromptEditor)
            editor.text = "two\nlines"
            await pilot.press("enter")
            await pilot.pause()

            assert [(message.role, message.content) for message in provider.calls[0]] == [("user", "two lines")]
            assert app.query_one(Collapsible).display is True
            assert app.query_one(Markdown).source == "- item\n\n```python\nprint('ok')\n```"
            assert editor.disabled is False
            assert status.render().plain == "[DEFAULT] Ready"
            assert [(message.role, message.content) for message in app.session.history] == [
                ("user", "two lines"),
                ("assistant", "- item\n\n```python\nprint('ok')\n```"),
            ]

    asyncio.run(run())


def test_permission_prompt_denies_without_leaking_content_and_agent_continues(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent, ToolCall
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionPrompt
        from fakuicode.tui.widgets import PromptEditor

        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def stream_agent(self, messages, tools, *, cancel_event=None):
                del messages, tools, cancel_event
                self.calls += 1
                if self.calls == 1:
                    yield AgentStreamEvent(
                        "tool_call",
                        tool_call=ToolCall(
                            "permission-1",
                            "write_file",
                            {"path": "notes.txt", "content": "DO-NOT-DISPLAY"},
                        ),
                    )
                else:
                    yield AgentStreamEvent("text_delta", "I will not write that file.")
                yield AgentStreamEvent("completed")

        provider = Provider()
        app = FakuicodeApp(make_config(), provider=provider, workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "write a note"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if list(app.query(PermissionPrompt)):
                    break

            prompt = app.query_one(PermissionPrompt)
            rendered = " ".join(widget.render().plain for widget in prompt.query("Static"))
            assert "write_file" in rendered
            assert "notes.txt" in rendered
            assert "write_file(notes.txt)" in rendered
            assert "DO-NOT-DISPLAY" not in rendered
            assert prompt.parent is app.query_one("#conversation")
            assert prompt.query_one(OptionList).highlighted == 0

            await pilot.press("4")
            for _ in range(40):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert provider.calls == 2
            assert not (tmp_path / "notes.txt").exists()

    asyncio.run(run())


def test_permission_prompt_escape_cancels_the_entire_agent_turn(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent, ToolCall
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionPrompt
        from fakuicode.tui.widgets import PromptEditor

        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def stream_agent(self, messages, tools, *, cancel_event=None):
                del messages, tools, cancel_event
                self.calls += 1
                yield AgentStreamEvent(
                    "tool_call",
                    tool_call=ToolCall(
                        f"permission-{self.calls}",
                        "write_file",
                        {"path": f"notes-{self.calls}.txt", "content": "blocked"},
                    ),
                )
                yield AgentStreamEvent("completed")

        provider = Provider()
        app = FakuicodeApp(make_config(), provider=provider, workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "keep trying to write"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if list(app.query(PermissionPrompt)):
                    break

            prompt = app.query_one(PermissionPrompt)
            help_text = " ".join(widget.render().plain for widget in prompt.query("Static"))
            assert "Esc 停止任务" in help_text

            await pilot.press("escape")
            for _ in range(60):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert provider.calls == 1
            assert not list(app.query(PermissionPrompt))
            assert not list(tmp_path.glob("notes-*.txt"))

    asyncio.run(run())


def test_permission_prompt_session_choice_allows_the_same_exact_target_without_reprompt(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent, ToolCall
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionPrompt
        from fakuicode.tui.widgets import PromptEditor

        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def stream_agent(self, messages, tools, *, cancel_event=None):
                del tools, cancel_event
                self.calls += 1
                if self.calls <= 2:
                    if self.calls == 2:
                        assert messages[-1].tool_results[0].success is True
                    yield AgentStreamEvent(
                        "tool_call",
                        tool_call=ToolCall(
                            f"permission-{self.calls}",
                            "write_file",
                            {"path": "session.txt", "content": f"value-{self.calls}"},
                        ),
                    )
                else:
                    yield AgentStreamEvent("text_delta", "done")
                yield AgentStreamEvent("completed")

        provider = Provider()
        app = FakuicodeApp(make_config(), provider=provider, workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "write twice"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if list(app.query(PermissionPrompt)):
                    break

            assert app.query_one(PermissionPrompt)
            await pilot.press("2")
            for _ in range(60):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert provider.calls == 3
            assert not list(app.query(PermissionPrompt))
            assert (tmp_path / "session.txt").read_text(encoding="utf-8") == "value-2"

    asyncio.run(run())


def test_completed_plan_offers_inline_execution_and_reuses_the_do_path(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PlanExecutionPrompt
        from fakuicode.tui.widgets import PromptEditor

        class Provider:
            def __init__(self) -> None:
                self.requests = []
                self.tool_names = []

            def stream_agent(self, messages, tools, *, cancel_event=None, system_instruction=""):
                del cancel_event, system_instruction
                self.requests.append(messages)
                self.tool_names.append([tool.name for tool in tools])
                if len(self.requests) == 1:
                    yield AgentStreamEvent("text_delta", "1. 将数字 1 写入 hello.txt")
                else:
                    yield AgentStreamEvent("text_delta", "计划执行完成。")
                yield AgentStreamEvent("completed")

        provider = Provider()
        app = FakuicodeApp(make_config(), provider=provider, workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/plan"
            await pilot.press("enter")
            await pilot.pause()
            assert app.session.mode == "plan"
            assert editor.disabled is False
            assert app._active_turn is None
            app._begin_turn("把数字 1 写入 hello.txt", editor)
            for _ in range(40):
                await pilot.pause()
                prompts = list(app.query(PlanExecutionPrompt))
                if prompts and list(prompts[0].query("Static")):
                    break

            assert len(provider.requests) == 1
            assert app.session.mode == "plan"
            assert app.session.saved_plan == "1. 将数字 1 写入 hello.txt"
            prompt = app.query_one(PlanExecutionPrompt)
            rendered = " ".join(widget.render().plain for widget in prompt.query("Static"))
            assert "退出 Plan 模式并执行" in rendered
            assert prompt.parent is app.query_one("#conversation")
            assert editor.disabled is True
            assert "write_file" not in provider.tool_names[0]

            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if len(provider.requests) == 2 and not editor.disabled:
                    break

            assert len(provider.requests) == 2
            assert app.session.mode == "execute"
            assert "write_file" in provider.tool_names[1]
            assert provider.requests[1][-1].content.startswith("Execute the saved plan.")
            assert not list(app.query(PlanExecutionPrompt))

    asyncio.run(run())


def test_do_without_a_saved_plan_leaves_plan_mode_without_calling_provider(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        provider = AgentTextProvider()
        app = FakuicodeApp(make_config(), provider=provider, workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/plan"
            await pilot.press("enter")
            await pilot.pause()

            assert app.session.mode == "plan"
            assert app.query_one("#status", Static).render().plain.startswith("[PLAN]")

            editor.text = "/do"
            await pilot.press("enter")
            await pilot.pause()

            assert app.session.mode == "execute"
            assert provider.calls == []
            assert app.query_one("#status", Static).render().plain.startswith("[DEFAULT]")
            assert "No saved plan was executed" in app.query(SystemNotice).last().render().plain

    asyncio.run(run())


def test_review_skill_keeps_the_slash_invocation_visible_and_pins_its_sop(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, UserMessage

        provider = AgentTextProvider()
        app = FakuicodeApp(make_config(), provider=provider, workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/review"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if provider.calls and not editor.disabled:
                    break

            assert len(provider.calls) == 1
            assert provider.calls[0][-1].content == "/review"
            assert app.query(UserMessage).last().render().plain == "/review"
            assert app.skill_manager is not None
            assert "请只读审查当前工作树相对 HEAD 的已跟踪改动" in app.skill_manager.active_prompt

    asyncio.run(run())


def test_skill_hot_refresh_updates_dynamic_help_before_dispatch(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), workspace=tmp_path)
        async with app.run_test() as pilot:
            package = tmp_path / ".fakuicode" / "skills" / "hot-check"
            package.mkdir(parents=True)
            (package / "SKILL.md").write_text(
                "---\nname: hot-check\ndescription: Hot workflow\nfakuicode: {}\n---\nCheck it.\n",
                encoding="utf-8",
            )
            editor = app.query_one(PromptEditor)
            editor.text = "/help"
            await pilot.press("enter")
            await pilot.pause()

            help_text = app.query(SystemNotice).last().render().plain
            assert "/hot-check [arguments]" in help_text

            (package / "SKILL.md").write_text("broken", encoding="utf-8")
            editor.text = "/help"
            await pilot.press("enter")
            await pilot.pause()

            assert "/hot-check" not in app.query(SystemNotice).last().render().plain

    asyncio.run(run())


def test_skills_install_slash_flow_previews_installs_and_refreshes_completion(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.skill_install_screen import SkillInstallScreen
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            workspace=tmp_path,
            skill_fetcher=FakeSkillFetcher(),
        )
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = (
                "/skills install "
                "https://www.skills.sh/anthropics/skills/frontend-design"
            )
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if isinstance(app.screen, SkillInstallScreen):
                    break

            assert isinstance(app.screen, SkillInstallScreen)
            details = app.screen.query_one("#skill-install-details", Static).render().plain
            assert "github.com/anthropics/skills" in details
            assert "a" * 40 in details
            assert "LICENSE.txt" in details
            assert app.screen.query_one(OptionList).highlighted == 0

            await pilot.press("down", "down", "down", "enter")
            for _ in range(60):
                await pilot.pause()
                if not editor.disabled:
                    break

            target = tmp_path / ".fakuicode" / "skills" / "frontend-design"
            assert (target / "SKILL.md").is_file()
            assert (target / "LICENSE.txt").read_bytes() == b"upstream license\n"
            assert (target / ".fakuicode" / "install.yaml").is_file()
            assert app._command_registry.find("frontend-design") is not None
            assert any(
                item.completion == "/frontend-design "
                for item in app._command_registry.suggest("/front")
            )
            assert provider.calls == []
            notices = "\n".join(item.render().plain for item in app.query(SystemNotice))
            assert "frontend-design" in notices

    asyncio.run(run())


def test_natural_language_install_tool_confirms_once_and_provider_continues(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.skill_install_screen import SkillInstallScreen
        from fakuicode.tui.widgets import PromptEditor

        provider = InstallSkillAgentProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            workspace=tmp_path,
            skill_fetcher=FakeSkillFetcher(),
        )
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = (
                "把这个 skill 装下："
                "https://www.skills.sh/anthropics/skills/frontend-design"
            )
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if isinstance(app.screen, SkillInstallScreen):
                    break

            assert isinstance(app.screen, SkillInstallScreen)
            await pilot.press("down", "down", "down", "enter")
            for _ in range(90):
                await pilot.pause()
                if (
                    len(provider.calls) == 2
                    and not editor.disabled
                    and any("已安装" in item.source for item in app.query(Markdown))
                ):
                    break

            assert len(provider.calls) == 2
            assert any(tool.name == "install_skill" for tool in provider.tool_sets[0])
            assert (tmp_path / ".fakuicode" / "skills" / "frontend-design" / "SKILL.md").is_file()
            assert any("已安装" in item.source for item in app.query(Markdown))
            assert app._command_registry.find("frontend-design") is not None

    asyncio.run(run())


def test_plan_execution_prompt_can_keep_the_plan_for_later(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PlanExecutionPrompt
        from fakuicode.tui.widgets import PromptEditor

        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            app.session.enable_plan_mode()
            app.session.saved_plan = "保留这份计划"
            editor.disabled = True
            app._show_inline_prompt(PlanExecutionPrompt())
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert not list(app.query(PlanExecutionPrompt))
            assert app.session.mode == "plan"
            assert app.session.saved_plan == "保留这份计划"
            assert editor.disabled is False

    asyncio.run(run())


def test_inline_choice_prompt_keeps_keyboard_navigation_after_page_takes_focus(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PlanExecutionPrompt
        from fakuicode.tui.widgets import ConversationView

        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), workspace=tmp_path)
        async with app.run_test() as pilot:
            app._show_inline_prompt(PlanExecutionPrompt())
            await pilot.pause()

            options = app.query_one(OptionList)
            assert app.focused is options
            assert options.highlighted == 0

            app.query_one(ConversationView).focus()
            await pilot.pause()
            await pilot.press("down")

            assert app.focused is options
            assert options.highlighted == 1

    asyncio.run(run())


def test_permissions_screen_switches_only_the_current_session_mode(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.permissions.models import PermissionMode
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionSettingsScreen
        from fakuicode.tui.widgets import PromptEditor

        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), workspace=tmp_path)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/permission"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, PermissionSettingsScreen)
            await pilot.press("down", "enter")
            await pilot.pause()

            assert app.session.runner.tools.permission_manager.mode is PermissionMode.TRUSTED
            app._new_conversation()
            assert app.session.runner.tools.permission_manager.mode is PermissionMode.DEFAULT

    asyncio.run(run())


def test_permissions_screen_persists_explicit_project_trust(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.permissions.config import PermissionConfigRepository, PermissionPaths
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionSettingsScreen
        from fakuicode.tui.widgets import PromptEditor

        repository = PermissionConfigRepository(
            PermissionPaths.for_workspace(tmp_path, home=tmp_path / "home"), tmp_path
        )
        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            workspace=tmp_path,
            permission_snapshot=repository.load(),
            permission_repository=repository,
        )
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/permissions"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, PermissionSettingsScreen)
            await pilot.press("end", "enter")
            await pilot.pause()

            assert repository.load().project_trusted is True
            assert app.session.runner.tools.permission_manager.snapshot.project_trusted is True

    asyncio.run(run())


def test_locked_permission_config_is_visible_and_cannot_switch_mode(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.permissions.config import PermissionConfigSnapshot
        from fakuicode.permissions.models import PermissionMode
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionSettingsScreen
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        snapshot = PermissionConfigSnapshot(
            mode=PermissionMode.STRICT,
            locked=True,
            diagnostics=("project shared permission config is invalid: invalid YAML",),
        )
        app = FakuicodeApp(
            make_config(), provider=AgentTextProvider(), workspace=tmp_path, permission_snapshot=snapshot
        )
        async with app.run_test() as pilot:
            assert any("project shared" in notice.render().plain for notice in app.query(SystemNotice))
            editor = app.query_one(PromptEditor)
            editor.text = "/permissions"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, PermissionSettingsScreen)
            rendered = " ".join(widget.render().plain for widget in app.screen.query("Static"))
            assert "invalid YAML" in rendered
            await pilot.press("enter")
            await pilot.pause()
            assert app.session.runner.tools.permission_manager.mode is PermissionMode.STRICT

    asyncio.run(run())


def test_app_uses_the_agent_session_for_a_native_tool_provider() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = AgentTextProvider()
        app = FakuicodeApp(make_config(), provider=provider)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "agent turn"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert len(provider.calls) == 1
            assert app.session.history[-1].content == "agent answer"

    asyncio.run(run())


def test_app_shows_a_final_fallback_after_a_tool_continuation_without_text() -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent, ToolCall
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.permission_prompt import PermissionPrompt
        from fakuicode.tui.widgets import AssistantTurn, PromptEditor, ToolActivity

        class ToolThenSilentProvider:
            def __init__(self) -> None:
                self.calls: list[Sequence[object]] = []

            def stream_agent(
                self,
                messages: Sequence[object],
                tools: Sequence[object],
                *,
                cancel_event: Event | None = None,
            ) -> Iterator[object]:
                self.calls.append(messages)
                if len(self.calls) == 1:
                    yield AgentStreamEvent(
                        "tool_call",
                        tool_call=ToolCall(
                            "call-1",
                            "edit_file",
                            {"path": "missing.file", "old_text": "before", "new_text": "after"},
                        ),
                    )
                yield AgentStreamEvent("completed")

        provider = ToolThenSilentProvider()
        app = FakuicodeApp(make_config(), provider=provider)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "inspect the readme"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if list(app.query(PermissionPrompt)):
                    await pilot.press("4")
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert len(provider.calls) == 2
            assert list(app.query(ToolActivity))
            assert list(app.query(Markdown))[-1].source == (
                "Tool execution completed, but the model did not provide a final response. "
                "Please use the results above or ask a more specific follow-up."
            )
            turn = app.query_one(AssistantTurn)
            assert turn.query_one(ToolActivity).region.y < turn.query_one(Markdown).region.y
            assert app.session.history[-1].content == list(app.query(Markdown))[-1].source

    asyncio.run(run())


def test_app_cycles_read_only_activity_in_one_line_and_keeps_the_final_answer() -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent, ToolCall
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, ToolActivity

        class Provider:
            def __init__(self) -> None:
                self.tool_sets: list[Sequence[object]] = []

            def stream_agent(
                self,
                messages: Sequence[object],
                tools: Sequence[object],
                *,
                cancel_event: Event | None = None,
            ) -> Iterator[object]:
                self.tool_sets.append(tools)
                if len(self.tool_sets) == 1:
                    yield AgentStreamEvent(
                        "tool_call", tool_call=ToolCall("call-1", "read_file", {"path": "README.md"})
                    )
                    yield AgentStreamEvent(
                        "tool_call", tool_call=ToolCall("call-2", "find_files", {"pattern": "README.md"})
                    )
                    yield AgentStreamEvent(
                        "tool_call",
                        tool_call=ToolCall("call-3", "search_code", {"query": "Fakuicode", "path": "README.md"}),
                    )
                else:
                    assert [definition.name for definition in tools] == [
                        "read_file",
                        "write_file",
                        "edit_file",
                        "run_command",
                        "find_files",
                        "search_code",
                        "agent",
                        "task_list",
                        "task_get",
                        "task_stop",
                        "send_message",
                    ]
                    yield AgentStreamEvent("text_delta", "README.md is available at the workspace root.")
                yield AgentStreamEvent("completed")

        provider = Provider()
        app = FakuicodeApp(make_config(), provider=provider)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "find the readme"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            await pilot.pause(0.6)
            activities = list(app.query(ToolActivity))
            assert len(activities) == 1
            assert "search_code" in activities[0].render().plain
            assert "Done" in activities[0].render().plain
            assert list(app.query(Markdown))[-1].source == "README.md is available at the workspace root."

    asyncio.run(run())


def test_app_persists_an_agent_turn_without_cross_thread_sqlite_errors(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        store = ConversationStore(tmp_path / "history.sqlite3")
        app = FakuicodeApp(make_config(), provider=AgentTextProvider(), store=store)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "agent turn"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            status = app.query_one("#status", Static).render().plain
            assert status.startswith("[DEFAULT] Ready · tokens unavailable")
            assert "Permissions: default · Project: untrusted" in status
            assert app.conversation is not None
            assert [(event.kind, event.content) for event in store.load_events(app.conversation.id)] == [
                ("user", "agent turn"),
                ("assistant", "agent answer"),
            ]

    asyncio.run(run())


def test_app_starts_a_new_conversation_and_resume_restores_the_selected_history(tmp_path: Path) -> None:
    from fakuicode.storage import ConversationStore

    store = ConversationStore(tmp_path / "history.sqlite3")
    record = store.create_conversation("Previous", tmp_path, "default")
    store.append_event(record.id, "user", "earlier question")
    store.append_event(record.id, "assistant", "earlier answer")

    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, UserMessage

        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )
        assert app.conversation is not None and app.conversation.id != record.id
        assert app.session.history == []
        assert {item.id for item in store.list_conversations()} == {record.id, app.conversation.id}
        async with app.run_test() as pilot:
            for _ in range(10):
                await pilot.pause()
                if app.is_mounted:
                    break
            assert list(app.query(UserMessage)) == []

            editor = app.query_one(PromptEditor)
            editor.text = "/resume"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down", "enter")
            for _ in range(20):
                await pilot.pause()
                if app.conversation is not None and app.conversation.id == record.id and list(app.query(UserMessage)):
                    break

            assert app.conversation is not None and app.conversation.id == record.id
            assert [(message.role, message.content) for message in app.session.history] == [
                ("user", "earlier question"),
                ("assistant", "earlier answer"),
            ]
            assert app.query_one(UserMessage).render().plain == "earlier question"
            assert app.query_one(Markdown).source == "earlier answer"

    asyncio.run(run())


def test_resume_without_an_id_opens_a_keyboard_session_picker_with_message_count(tmp_path: Path) -> None:
    async def run() -> None:
        from datetime import datetime

        from textual.widgets import OptionList

        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import SessionPicker
        from fakuicode.tui.widgets import PromptEditor, UserMessage

        store = ConversationStore(tmp_path / "history.sqlite3")
        previous = store.create_conversation("Previous work", tmp_path, "default")
        foreign_workspace = tmp_path / "another-workspace"
        foreign_workspace.mkdir()
        store.create_conversation("Foreign work", foreign_workspace, "default")
        store.append_event(previous.id, "user", "earlier question")
        store.append_event(previous.id, "assistant", "earlier answer")
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/resume"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, SessionPicker)
            options = app.screen.query_one(OptionList)
            assert options.highlighted == 0
            assert options.has_focus
            previous_option = str(options.get_option_at_index(1).prompt)
            assert "0 messages" in str(options.get_option_at_index(0).prompt)
            assert "Previous work" in previous_option
            assert "default" in previous_option
            assert datetime.fromtimestamp(previous.updated_at / 1_000_000_000).strftime("%Y-%m-%d %H:%M") in previous_option
            assert previous.id[:8] not in previous_option
            assert "2 messages" in previous_option
            assert all("Foreign work" not in str(options.get_option_at_index(index).prompt) for index in range(2))

            await pilot.press("down", "enter")
            for _ in range(20):
                await pilot.pause()
                if app.conversation is not None and app.conversation.id == previous.id and list(app.query(UserMessage)):
                    break

            assert app.conversation is not None and app.conversation.id == previous.id
            assert app.query_one(UserMessage).render().plain == "earlier question"

    asyncio.run(run())


def test_first_prompt_titles_new_session_and_delete_picker_backfills_old_sessions(tmp_path: Path) -> None:
    async def run() -> None:
        from textual.widgets import OptionList

        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import SessionPicker
        from fakuicode.tui.widgets import PromptEditor

        store = ConversationStore(tmp_path / "history.sqlite3")
        legacy = store.create_conversation("New conversation", tmp_path, "default")
        store.append_event(legacy.id, "user", "Legacy database migration question")
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )
        assert app.conversation is not None
        current_id = app.conversation.id

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "Explain the input rendering issue"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if app._active_turn is None and not editor.disabled:
                    break

            assert store.get_conversation(current_id).title == "Explain the input rendering issue"

            editor.text = "/delete"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, SessionPicker)
            options = app.screen.query_one(OptionList)
            prompts = [str(options.get_option_at_index(index).prompt) for index in range(options.option_count)]
            assert any("Explain the input rendering issue" in prompt for prompt in prompts)
            assert any("Legacy database migration question" in prompt for prompt in prompts)
            assert all("New conversation" not in prompt for prompt in prompts)
            assert store.get_conversation(legacy.id).title == "Legacy database migration question"

    asyncio.run(run())


def test_delete_without_an_id_uses_current_workspace_picker_and_confirmation(tmp_path: Path) -> None:
    async def run() -> None:
        from textual.widgets import OptionList, Static

        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import ConfirmationScreen, SessionPicker
        from fakuicode.tui.widgets import PromptEditor

        store = ConversationStore(tmp_path / "history.sqlite3")
        previous = store.create_conversation("Delete me", tmp_path, "default")
        foreign_workspace = tmp_path / "another-workspace"
        foreign_workspace.mkdir()
        foreign = store.create_conversation("Keep foreign", foreign_workspace, "default")
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/delete"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, SessionPicker)
            assert "Delete conversation" in app.screen.query_one("#session-picker-title", Static).render().plain
            options = app.screen.query_one(OptionList)
            assert options.option_count == 2
            assert all("Keep foreign" not in str(options.get_option_at_index(index).prompt) for index in range(2))
            await pilot.press("down", "enter")
            await pilot.pause()

            assert isinstance(app.screen, ConfirmationScreen)
            assert "Delete me" in app.screen.query_one("#confirmation-title", Static).render().plain
            assert {item.id for item in store.list_conversations()} >= {previous.id, foreign.id}
            await pilot.press("down", "enter")
            await pilot.pause()

            remaining_ids = {item.id for item in store.list_conversations()}
            assert previous.id not in remaining_ids
            assert foreign.id in remaining_ids
            assert provider.calls == []

    asyncio.run(run())


def test_session_picker_keeps_working_when_one_message_count_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        from textual.widgets import OptionList

        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import SessionPicker
        from fakuicode.tui.widgets import PromptEditor

        store = ConversationStore(tmp_path / "history.sqlite3")
        previous = store.create_conversation("Unreadable count", tmp_path, "default")
        original_count = store.visible_message_count

        def count_or_fail(conversation_id: str) -> int:
            if conversation_id == previous.id:
                raise RuntimeError("private database detail")
            return original_count(conversation_id)

        monkeypatch.setattr(store, "visible_message_count", count_or_fail)
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/resume"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, SessionPicker)
            options = app.screen.query_one(OptionList)
            unreadable = str(options.get_option_at_index(1).prompt)
            assert "? messages" in unreadable
            assert "private database detail" not in unreadable
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.conversation is not None
            assert app.conversation.id == previous.id

    asyncio.run(run())


def test_instruction_loader_warnings_are_repeated_once_when_a_new_snapshot_is_loaded(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.instructions import (
            InstructionDiagnostic,
            InstructionDiagnosticCode,
            InstructionScope,
            InstructionSnapshot,
        )
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import SystemNotice

        class Loader:
            def load(self) -> InstructionSnapshot:
                return InstructionSnapshot(
                    text="",
                    loaded_layers=(),
                    processed_target_count=0,
                    diagnostics=(
                        InstructionDiagnostic(
                            InstructionDiagnosticCode.FILE_NOT_FOUND,
                            InstructionScope.PROJECT,
                            "unsafe\x1b[31m\x9b\u202eAGENTS.md",
                        ),
                    ),
                )

        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            instruction_loader=Loader(),
            workspace=tmp_path,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app._new_conversation()
            await pilot.pause()

            warnings = [notice.render().plain for notice in app.query(SystemNotice)]
            assert warnings == [
                "Project instructions: 1 warning(s)\ncode=file_not_found scope=project source=unsafe�[31m��AGENTS.md",
                "Project instructions: 1 warning(s)\ncode=file_not_found scope=project source=unsafe�[31m��AGENTS.md",
            ]

    asyncio.run(run())


def test_instruction_snapshot_reloads_only_at_approved_session_boundaries(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.instructions import InstructionSnapshot
        from fakuicode.models import ProfileSet, ProviderConfig
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        class Loader:
            def __init__(self) -> None:
                self.calls = 0

            def load(self) -> InstructionSnapshot:
                self.calls += 1
                return InstructionSnapshot(f"snapshot-{self.calls}", (), 3, ())

        default = make_config()
        careful = ProviderConfig("openai", "careful-model", "https://api.example.test/v1", "test-key")
        profiles = ProfileSet({"default": default, "careful": careful}, "default")
        store = ConversationStore(tmp_path / "history.sqlite3")
        previous = store.create_conversation("Previous", tmp_path, "default")
        loader = Loader()
        provider = AgentTextProvider()
        app = FakuicodeApp(
            default,
            provider=provider,
            provider_factory=lambda _config: provider,
            profiles=profiles,
            store=store,
            workspace=tmp_path,
            instruction_loader=loader,
        )

        assert loader.calls == 1
        async with app.run_test() as pilot:
            app._last_prompt = "retry this"
            app._handle_command("/retry")
            for _ in range(20):
                await pilot.pause()
                if app._active_turn is None and not app.query_one(PromptEditor).disabled:
                    break
            app._handle_command("/clear")
            app._handle_command("/compact")
            for _ in range(20):
                await pilot.pause()
                if not app._compact_active:
                    break
            assert loader.calls == 1

            app._handle_command("/new")
            assert loader.calls == 2

            app._resume_conversation(previous.id)
            for _ in range(20):
                await pilot.pause()
                if app.conversation is not None and app.conversation.id == previous.id:
                    break
            assert loader.calls == 3

            app._switch_profile("careful")
            assert loader.calls == 4
            assert app.conversation is not None
            app._delete_conversation(app.conversation.id)
            assert loader.calls == 4

    asyncio.run(run())


def test_profile_switch_and_app_shutdown_close_each_owned_provider_once(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        from fakuicode.models import ProfileSet, ProviderConfig
        from fakuicode.tui.app import FakuicodeApp

        class Provider(AgentTextProvider):
            def __init__(self) -> None:
                super().__init__()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        default = make_config()
        alternate = ProviderConfig(
            "openai",
            "alternate",
            "https://api.example.test/v1",
            "test-key",
        )
        profiles = ProfileSet(
            {"default": default, "alternate": alternate},
            "default",
        )
        first = Provider()
        created: list[Provider] = []

        def factory(_config: ProviderConfig) -> Provider:
            provider = Provider()
            created.append(provider)
            return provider

        app = FakuicodeApp(
            default,
            provider=first,
            provider_factory=factory,
            profiles=profiles,
            workspace=tmp_path,
        )

        async with app.run_test():
            app._switch_profile("alternate")
            assert first.close_calls == 1
            assert len(created) == 1

        assert created[0].close_calls == 1

    asyncio.run(run())


def test_status_reports_instruction_snapshot_metadata_without_content(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.instructions import (
            InstructionDiagnostic,
            InstructionDiagnosticCode,
            InstructionScope,
            InstructionSnapshot,
        )
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import SystemNotice

        class Loader:
            def load(self) -> InstructionSnapshot:
                return InstructionSnapshot(
                    text="项目秘密",
                    loaded_layers=(InstructionScope.USER, InstructionScope.PROJECT),
                    processed_target_count=7,
                    diagnostics=(
                        InstructionDiagnostic(
                            InstructionDiagnosticCode.FILE_NOT_FOUND,
                            InstructionScope.PROJECT,
                            "AGENTS.md",
                        ),
                    ),
                )

        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            instruction_loader=Loader(),
            workspace=tmp_path,
        )
        async with app.run_test() as pilot:
            app._handle_command("/status")
            await pilot.pause()

            status = app.query(SystemNotice).last().render().plain
            assert "Instructions: 2 layers · 7 targets · 12 bytes · 1 warning(s)" in status
            assert "项目秘密" not in status

    asyncio.run(run())


def test_app_restores_compact_tool_status_above_the_answer(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import AssistantTurn, PromptEditor, ToolActivity

        store = ConversationStore(tmp_path / "history.sqlite3")
        record = store.create_conversation("Tools", tmp_path, "default")
        store.append_event(record.id, "user", "inspect the readme")
        store.append_event(
            record.id,
            "assistant",
            "I will read it. ",
            metadata={"tool_calls": [{"id": "call-1", "name": "write_file", "arguments": {"path": "test/output.txt"}}]},
        )
        store.append_event(
            record.id,
            "tool_call",
            "write_file",
            call_id="call-1",
            metadata={"arguments": {"path": "test/output.txt"}},
        )
        store.append_event(
            record.id,
            "tool_result",
            "contents",
            call_id="call-1",
            metadata={"tool_name": "write_file", "success": True, "summary": "wrote test/output.txt"},
        )
        store.append_event(record.id, "assistant", "I read it.")
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/resume"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down", "enter")
            for _ in range(20):
                await pilot.pause()
                if list(app.query(ToolActivity)):
                    break
            activity = app.query_one(ToolActivity)
            assert isinstance(activity, Static)
            assert not isinstance(activity, Collapsible)
            assert activity.styles.height.value == 1
            assert "Done" in activity.render().plain
            assert "test/output.txt" in activity.render().plain
            turn = app.query_one(AssistantTurn)
            assert activity.parent is turn.query_one(".tool-activities")
            assert activity.region.y < turn.query_one(Markdown).region.y

    asyncio.run(run())


def test_app_restores_only_latest_compaction_status_without_summary_content(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice, UserMessage

        store = ConversationStore(tmp_path / "history.sqlite3")
        record = store.create_conversation("Compacted", tmp_path, "default")
        user = store.append_event(record.id, "user", "visible original question")
        store.append_event(record.id, "assistant", "visible original answer")
        store.append_context_summary(
            record.id,
            "OLD-SUMMARY-SECRET",
            through_sequence=user.sequence,
            preserved_user_sequences=(user.sequence,),
            trigger="automatic",
            estimated_before=120_000,
            estimated_after=15_000,
            format_version=1,
        )
        store.append_context_diagnostic(
            record.id,
            {
                "trigger": "automatic",
                "result": "compacted",
                "estimated_before": 120_000,
                "estimated_after": 15_000,
                "artifact_count": 0,
                "artifact_bytes": 0,
                "duration_ms": 10,
                "consecutive_failures": 0,
                "error_category": "none",
            },
        )
        store.append_context_summary(
            record.id,
            "LATEST-SUMMARY-SECRET",
            through_sequence=user.sequence,
            preserved_user_sequences=(user.sequence,),
            trigger="manual",
            estimated_before=115_000,
            estimated_after=12_000,
            format_version=1,
        )
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            provider_factory=lambda _config: provider,
            store=store,
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/resume"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down", "enter")
            for _ in range(30):
                await pilot.pause()
                if list(app.query(UserMessage)):
                    break

            notices = [notice.render().plain for notice in app.query(SystemNotice)]
            rendered = "\n".join(notices)
            assert "visible original question" in app.query_one(UserMessage).render().plain
            assert "OLD-SUMMARY-SECRET" not in rendered
            assert "LATEST-SUMMARY-SECRET" not in rendered
            assert sum("context compaction" in notice.lower() for notice in notices) == 1
            assert "~115,000 → ~12,000 tokens" in rendered

    asyncio.run(run())


def test_tui_routes_local_session_commands_without_calling_the_provider(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        provider = AgentTextProvider()
        app = FakuicodeApp(make_config(), provider=provider, store=ConversationStore(tmp_path / "history.sqlite3"))
        async with app.run_test() as pilot:
            initial_id = app.conversation.id
            editor = app.query_one(PromptEditor)
            editor.text = "/new"
            await pilot.press("enter")
            await pilot.pause()
            assert app.conversation.id != initial_id
            assert provider.calls == []

            editor.text = "/sessions"
            await pilot.press("enter")
            await pilot.pause()
            assert initial_id[:8] in app.query(SystemNotice).last().render().plain

            editor.text = "/session"
            await pilot.press("enter")
            await pilot.pause()
            assert initial_id[:8] in app.query(SystemNotice).last().render().plain
            assert provider.calls == []

    asyncio.run(run())


def test_tui_delete_command_coordinates_database_and_context_artifacts(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.context_artifacts import ContextArtifactStore
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        store = ConversationStore(tmp_path / "history.sqlite3")
        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            store=store,
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            deleted_id = app.conversation.id
            artifacts = ContextArtifactStore(tmp_path, deleted_id)
            reference = artifacts.write_tool_result(
                source_sequence=1,
                output="tool result",
                success=True,
            )
            editor = app.query_one(PromptEditor)
            editor.text = f"/delete {deleted_id[:8]}"
            await pilot.press("enter")
            await pilot.pause()

            assert app.conversation.id != deleted_id
            assert deleted_id not in {record.id for record in store.list_conversations()}
            assert not (tmp_path / reference.read_path).exists()
            assert provider.calls == []

    asyncio.run(run())


def test_tui_stops_the_current_session_before_deleting_its_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        import fakuicode.tui.app as app_module
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp

        store = ConversationStore(tmp_path / "history.sqlite3")
        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            store=store,
            workspace=tmp_path,
        )
        operations: list[str] = []
        original_close = app._close_agent_session
        original_delete = app_module.delete_conversation_with_artifacts

        def track_close() -> None:
            operations.append("close")
            original_close()

        def track_delete(target_store, conversation_id):
            operations.append("delete")
            return original_delete(target_store, conversation_id)

        monkeypatch.setattr(app, "_close_agent_session", track_close)
        monkeypatch.setattr(
            app_module,
            "delete_conversation_with_artifacts",
            track_delete,
        )

        async with app.run_test():
            assert app.conversation is not None
            app._delete_conversation(app.conversation.id)
            assert operations[:2] == ["close", "delete"]

    asyncio.run(run())


def test_tui_session_commands_do_not_expose_or_delete_another_workspace(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import SystemNotice

        workspace = tmp_path / "current"
        other_workspace = tmp_path / "other"
        workspace.mkdir()
        other_workspace.mkdir()
        store = ConversationStore(tmp_path / "history.sqlite3")
        local = store.create_conversation("Local session", workspace, "default")
        foreign = store.create_conversation(
            "Foreign secret session",
            other_workspace,
            "default",
        )
        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            store=store,
            workspace=workspace,
        )

        async with app.run_test():
            app.show_sessions()
            rendered = app.query(SystemNotice).last().render().plain
            assert local.id[:8] in rendered
            assert foreign.id[:8] not in rendered
            assert foreign.title not in rendered

            app._delete_conversation(foreign.id)
            assert store.get_conversation(foreign.id) == foreign
            assert "not found or is ambiguous" in app.query(SystemNotice).last().render().plain

            app._delete_conversation(foreign.id[:8])
            assert store.get_conversation(foreign.id) == foreign
            assert "not found or is ambiguous" in app.query(SystemNotice).last().render().plain

    asyncio.run(run())


def test_context_status_line_contains_only_non_content_metadata() -> None:
    from fakuicode.models import ContextStatus
    from fakuicode.tui.app import _format_context_status

    rendered = _format_context_status(
        ContextStatus(
            trigger="automatic",
            result="compacted",
            estimated_before=118_400,
            estimated_after=12_300,
        )
    )

    assert rendered == "Automatic context compaction complete · ~118,400 → ~12,300 tokens"
    assert "summary" not in rendered.lower()


def test_automatic_and_manual_compaction_each_add_one_compact_notice(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent, ContextStatus
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor, SystemNotice

        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            store=ConversationStore(tmp_path / "history.sqlite3"),
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            app._handle_stream_event(
                AgentStreamEvent(
                    "context_status",
                    context_status=ContextStatus(
                        "automatic",
                        "compacted",
                        estimated_before=115_000,
                        estimated_after=12_000,
                    ),
                )
            )
            await pilot.pause()
            automatic_notices = [
                notice.render().plain
                for notice in app.query(SystemNotice)
                if "context compaction" in notice.render().plain.lower()
            ]
            assert automatic_notices == [
                "Automatic context compaction complete · ~115,000 → ~12,000 tokens"
            ]

            app.session.compact = lambda **_kwargs: ContextStatus(  # type: ignore[method-assign, union-attr]
                "manual",
                "compacted",
                estimated_before=80_000,
                estimated_after=10_000,
            )
            editor = app.query_one(PromptEditor)
            editor.text = "/compact"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if not editor.disabled:
                    break

            all_notices = [
                notice.render().plain
                for notice in app.query(SystemNotice)
                if "context compaction" in notice.render().plain.lower()
            ]
            assert all_notices == [
                "Automatic context compaction complete · ~115,000 → ~12,000 tokens",
                "Manual context compaction complete · ~80,000 → ~10,000 tokens",
            ]
            assert provider.calls == []

    asyncio.run(run())


def test_compact_command_runs_in_background_and_escape_cancels_it(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.errors import RequestCancelled
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = AgentTextProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            store=ConversationStore(tmp_path / "history.sqlite3"),
            workspace=tmp_path,
        )
        started = Event()

        def compact(*, cancel_event: Event | None = None):
            assert cancel_event is not None
            started.set()
            while not cancel_event.wait(0.01):
                pass
            raise RequestCancelled()

        app.session.compact = compact  # type: ignore[method-assign, union-attr]

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/compact"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if started.is_set():
                    break

            assert started.is_set()
            assert editor.disabled is True
            assert app._active_turn is None
            assert provider.calls == []

            await pilot.press("escape")
            for _ in range(30):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert app._compact_active is False
            assert app.session.context_manager.consecutive_summary_failures == 0  # type: ignore[union-attr]
            assert app.query_one("#status", Static).render().plain == "[DEFAULT] Request cancelled."

    asyncio.run(run())


def test_retry_command_resends_the_last_prompt() -> None:
    async def run() -> None:
        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = FakeProvider(
            [
                [StreamEvent("text_delta", "first answer"), StreamEvent("completed")],
                [StreamEvent("text_delta", "second answer"), StreamEvent("completed")],
            ]
        )
        app = FakuicodeApp(make_config(), provider=provider)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "try this"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break
            editor.text = "/retry"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break
            assert len(provider.calls) == 2
            assert provider.calls[1][-1].content == "try this"

    asyncio.run(run())


def test_model_command_opens_a_picker_and_switches_only_after_confirmation(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import ProfileSet, ProviderConfig
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import ModelPicker
        from fakuicode.tui.widgets import PromptEditor

        fast = ProviderConfig("anthropic", "fast-model", "https://api.example.test/v1", "test-key")
        careful = ProviderConfig("openai", "careful-model", "https://api.example.test/v1", "test-key")
        profiles = ProfileSet({"fast": fast, "careful": careful}, "fast")
        created = []

        def factory(config):
            created.append(config.model)
            return AgentTextProvider()

        app = FakuicodeApp(
            fast,
            provider_factory=factory,
            profiles=profiles,
            profile_name="fast",
            store=ConversationStore(tmp_path / "history.sqlite3"),
        )
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/model"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ModelPicker)
            assert created == ["fast-model"]

            picker = app.screen
            picker.query_one("#model-filter").value = "missing"
            await pilot.pause()
            assert picker.query_one("#model-picker-empty").display
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ModelPicker)

            picker.query_one("#model-filter").value = ""
            await pilot.pause()
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.config.model == "careful-model"
            assert app.profile_name == "careful"
            assert created == ["fast-model", "careful-model"]
            assert app.conversation.profile_name == "careful"
            assert app.query_one("#footer-model", Static).render().plain == "careful-model"

            editor = app.query_one(PromptEditor)
            editor.text = "/model fast"
            await pilot.press("enter")
            await pilot.pause()
            assert app.profile_name == "careful"
            assert created == ["fast-model", "careful-model"]
            assert "Please use /model to open the picker." in app.query_one("#status", Static).render().plain

    asyncio.run(run())


def test_model_picker_cancel_keeps_the_current_profile_and_session(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import ProfileSet, ProviderConfig
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.model_picker import ModelPicker
        from fakuicode.tui.widgets import PromptEditor

        fast = ProviderConfig("anthropic", "fast-model", "https://api.example.test/v1", "test-key")
        careful = ProviderConfig("openai", "careful-model", "https://api.example.test/v1", "test-key")
        profiles = ProfileSet({"fast": fast, "careful": careful}, "fast")
        created = []

        def factory(config):
            created.append(config.model)
            return AgentTextProvider()

        app = FakuicodeApp(
            fast,
            provider_factory=factory,
            profiles=profiles,
            profile_name="fast",
            store=ConversationStore(tmp_path / "history.sqlite3"),
        )
        async with app.run_test() as pilot:
            original_conversation = app.conversation.id
            editor = app.query_one(PromptEditor)
            editor.text = "/model"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ModelPicker)

            await pilot.press("escape")
            await pilot.pause()
            assert app.profile_name == "fast"
            assert app.config.model == "fast-model"
            assert app.conversation.id == original_conversation
            assert created == ["fast-model"]

    asyncio.run(run())


def test_provider_error_recovers_input_without_committing_failed_turn() -> None:
    async def run() -> None:
        from fakuicode.errors import ProviderError
        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = FakeProvider(
            [
                ProviderError("safe failure"),
                [StreamEvent("text_delta", "[bold]literal[/bold]"), StreamEvent("completed")],
            ]
        )
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "first"
            await pilot.press("enter")
            await pilot.pause()
            assert editor.disabled is False
            assert app.session.history == []
            assert "safe failure" in app.query_one("#status", Static).render().plain
            assert app.query_one("#footer-model", Static).render().plain == "claude-test"
            assert app.query_one(Collapsible).display is False

            editor.text = "second"
            await pilot.press("enter")
            await pilot.pause()
            assert list(app.query(Markdown))[-1].source == "[bold]literal[/bold]"
            assert [(message.role, message.content) for message in app.session.history] == [
                ("user", "second"),
                ("assistant", "[bold]literal[/bold]"),
            ]

    asyncio.run(run())


def test_empty_provider_completion_is_shown_as_an_error_instead_of_a_blank_turn() -> None:
    async def run() -> None:
        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = FakeProvider(
            [
                [
                    StreamEvent("thinking_start"),
                    StreamEvent("thinking_delta", "reasoning without an answer"),
                    StreamEvent("thinking_end"),
                    StreamEvent("completed"),
                ]
            ]
        )
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "answer this"
            await pilot.press("enter")
            await pilot.pause()

            assert editor.disabled is False
            assert app.session.history == []
            assert "without text content" in app.query_one("#status", Static).render().plain
            assert "without text content" in app.query_one(".assistant-stream", Static).render().plain

    asyncio.run(run())


def test_app_blocks_duplicate_submission_while_streaming() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = BlockingProvider()
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "first"
            await pilot.press("enter")
            for _ in range(10):
                if provider.started.is_set():
                    break
                await pilot.pause()

            assert provider.started.is_set()
            assert editor.disabled is True
            assert "生成中" in app.query_one("#status", Static).render().plain

            editor.text = "must not send"
            await pilot.press("enter")
            assert len(provider.calls) == 1

            provider.release.set()
            for _ in range(10):
                if not editor.disabled:
                    break
                await pilot.pause()
            assert editor.disabled is False
            assert len(provider.calls) == 1

    asyncio.run(run())


def test_escape_cancels_an_active_turn_without_committing_it() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = BlockingProvider()
        app = FakuicodeApp(make_config(), provider=provider)
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "cancel me"
            await pilot.press("enter")
            for _ in range(10):
                await pilot.pause()
                if provider.started.is_set():
                    break
            assert provider.started.is_set()

            await pilot.press("escape")
            assert "Cancelling" in app.query_one("#status", Static).render().plain
            provider.release.set()
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert app.session.history == []
            assert "Request cancelled" in app.query_one("#status", Static).render().plain

    asyncio.run(run())


def test_exit_shortcut_stops_the_tui_cleanly() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp

        app = FakuicodeApp(make_config(), provider=FakeProvider([]))
        async with app.run_test() as pilot:
            await pilot.press("ctrl+q")

        assert app.is_running is False

    asyncio.run(run())


def test_background_completion_renders_task_id_and_plain_result_report(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        from fakuicode.subagents.runtime import ChildRunResult
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import SubagentResultNotice

        class CompletedSession:
            id = "session-report"
            name = "delivery-plan"
            role = "plan"
            profile_name = "default"
            conversation_id = "conversation-report"

            def run_to_completion(self, prompt: str, *, event_sink=None):
                del prompt, event_sink
                return ChildRunResult(
                    "## 配送方案\n\n**不要解析为 Rich markup**",
                    "completed",
                    tool_count=2,
                    last_activity="read_file",
                )

            def cancel(self) -> None:
                pass

            def close(self, *, status: str = "completed") -> None:
                del status

        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            workspace=tmp_path,
        )
        async with app.run_test() as pilot:
            assert app._task_manager is not None
            task_id = app._task_manager.launch(
                CompletedSession(),
                "plan delivery",
                "delivery plan",
                notify_on_done=True,
            )
            assert app._task_manager.wait(task_id, timeout=1) is not None

            for _ in range(20):
                await pilot.pause()
                if len(app.query(SubagentResultNotice)):
                    break

            report = app.query_one(SubagentResultNotice)
            assert report.task_id == task_id
            assert report.agent_name == "delivery-plan"
            assert report.task_status == "completed"
            rendered = report.result_body.render().plain
            assert "## 配送方案" in rendered
            assert "**不要解析为 Rich markup**" in rendered
            assert task_id in report.title

    asyncio.run(run())


def test_narrow_terminal_keeps_the_prompt_available_after_a_long_reply() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        long_reply = "\n".join(f"line {index}" for index in range(60))
        app = FakuicodeApp(
            make_config(),
            provider=FakeProvider([[StreamEvent("text_delta", long_reply), StreamEvent("completed")]]),
        )

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "show a long reply"
            await pilot.press("enter")
            await pilot.pause()

            conversation = app.query_one("#conversation", VerticalScroll)
            assert conversation.max_scroll_y > 0
            assert editor.disabled is False
            assert app.focused is editor

    asyncio.run(run())


def test_chrome_hides_the_protocol_and_keeps_model_in_input_information_row() -> None:
    async def run() -> None:
        from textual.widgets import Static

        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import BrandPanel, PromptEditor, PromptPanel

        app = FakuicodeApp(make_config(), provider=FakeProvider([]))
        async with app.run_test() as pilot:
            await pilot.pause()
            brand = app.query_one(BrandPanel)
            prompt_panel = app.query_one(PromptPanel)
            brand_text = render_plain(brand.render())
            assert "claude-test" in brand_text
            assert "ANTHROPIC" not in brand_text
            assert prompt_panel.query_one("#footer-model", Static).render().plain == "claude-test"
            assert prompt_panel.query_one("#status", Static).region.y > app.query_one(PromptEditor).region.y

    asyncio.run(run())


def test_expanded_thinking_refreshes_the_conversation_scroll_range() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        app = FakuicodeApp(
            make_config(),
            provider=FakeProvider(
                [
                    [
                        StreamEvent("thinking_start"),
                        StreamEvent("thinking_delta", "reason " * 100),
                        StreamEvent("thinking_end"),
                        StreamEvent("text_delta", "answer"),
                        StreamEvent("completed"),
                    ]
                ]
            ),
        )

        async with app.run_test(size=(48, 24)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "show your reasoning"
            await pilot.press("enter")
            await pilot.pause()

            thinking = app.query_one(Collapsible)
            thinking.collapsed = False
            await pilot.pause()

            conversation = app.query_one("#conversation", VerticalScroll)
            assert conversation.max_scroll_y > 0
            conversation.scroll_to(
                y=conversation.max_scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
            assert conversation.scroll_y == conversation.max_scroll_y

            expanded_scroll_range = conversation.max_scroll_y
            thinking.collapsed = True
            await pilot.pause()
            assert conversation.max_scroll_y < expanded_scroll_range

    asyncio.run(run())


def test_twelve_streamed_turns_keep_all_messages_scrollable() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import AssistantTurn, PromptEditor, UserMessage

        responses = [
            [StreamEvent("text_delta", f"answer {index} " * 10), StreamEvent("completed")]
            for index in range(12)
        ]
        app = FakuicodeApp(make_config(), provider=FakeProvider(responses))

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            for index in range(12):
                editor.text = f"question {index}"
                await pilot.press("enter")
                for _ in range(10):
                    await pilot.pause()
                    if not editor.disabled:
                        break

            conversation = app.query_one("#conversation", VerticalScroll)
            assert len(list(conversation.query(UserMessage))) == 12
            assert len(list(conversation.query(AssistantTurn))) == 12
            assert conversation.max_scroll_y > 0
            conversation.scroll_to(y=0, animate=False, force=True, immediate=True)
            assert conversation.scroll_y == 0
            conversation.scroll_to(
                y=conversation.max_scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())


def test_provider_error_preserves_completed_turns_before_and_after_recovery() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.errors import ProviderError
        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import AssistantTurn, PromptEditor, UserMessage

        app = FakuicodeApp(
            make_config(),
            provider=FakeProvider(
                [
                    [StreamEvent("text_delta", "first answer"), StreamEvent("completed")],
                    ProviderError("safe failure"),
                    [StreamEvent("text_delta", "second answer"), StreamEvent("completed")],
                ]
            ),
        )

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            for prompt in ("first", "fails", "second"):
                editor.text = prompt
                await pilot.press("enter")
                for _ in range(10):
                    await pilot.pause()
                    if not editor.disabled:
                        break
                if prompt == "fails":
                    assert "safe failure" in app.query_one("#status", Static).render().plain

            conversation = app.query_one("#conversation", VerticalScroll)
            assert len(list(conversation.query(UserMessage))) == 3
            assert len(list(conversation.query(AssistantTurn))) == 3
            assert [(message.role, message.content) for message in app.session.history] == [
                ("user", "first"),
                ("assistant", "first answer"),
                ("user", "second"),
                ("assistant", "second answer"),
            ]

    asyncio.run(run())


def test_conversation_contains_startup_information_that_scrolls_with_messages() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.models import StreamEvent
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import BrandPanel, PromptEditor

        long_reply = "\n".join(f"line {index}" for index in range(60))
        app = FakuicodeApp(
            make_config(),
            provider=FakeProvider([[StreamEvent("text_delta", long_reply), StreamEvent("completed")]]),
        )

        async with app.run_test(size=(48, 16)) as pilot:
            conversation = app.query_one("#conversation", VerticalScroll)
            brand = app.query_one(BrandPanel)
            assert brand.parent is conversation
            assert brand.region.height <= 12

            editor = app.query_one(PromptEditor)
            editor.text = "show a long reply"
            await pilot.press("enter")
            await pilot.pause()

            assert conversation.max_scroll_y > 0
            conversation.scroll_to(
                y=conversation.max_scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
            assert brand.region.bottom <= conversation.scroll_y

    asyncio.run(run())


def test_narrow_information_row_keeps_the_full_model_name_visible_at_the_right() -> None:
    async def run() -> None:
        from fakuicode.models import ProviderConfig
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptPanel

        model = "deepseek-v4-flash[1m]"
        config = ProviderConfig("anthropic", model, "https://api.example.test/v1", "test-key")
        app = FakuicodeApp(config, provider=FakeProvider([]))

        async with app.run_test(size=(48, 16)) as pilot:
            await pilot.pause()
            prompt_panel = app.query_one(PromptPanel)
            footer_model = prompt_panel.query_one("#footer-model", Static)
            assert footer_model.render().plain == model
            assert footer_model.region.x > prompt_panel.query_one("#status", Static).region.x
            assert footer_model.region.bottom <= app.size.height

    asyncio.run(run())


def test_prompt_editor_caps_multiline_text_at_two_visible_rows() -> None:
    async def run() -> None:
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        app = FakuicodeApp(make_config(), provider=FakeProvider([]))
        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "\n".join("line" for _ in range(7))
            await pilot.pause()
            assert editor.region.height == 4
            assert editor.size.height == 2
            assert editor.max_scroll_y > 0

    asyncio.run(run())


def test_expanding_thinking_while_streaming_keeps_the_current_viewport_position() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = PausedThinkingProvider()
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "show your reasoning"
            await pilot.press("enter")
            for _ in range(10):
                if provider.started.is_set():
                    break
                await pilot.pause()

            assert provider.started.is_set()
            conversation = app.query_one("#conversation", VerticalScroll)
            thinking = app.query_one(Collapsible)
            thinking.collapsed = False
            await pilot.pause()
            scroll_before_remaining_thinking = conversation.scroll_y

            provider.release.set()
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False
            assert conversation.max_scroll_y > scroll_before_remaining_thinking
            assert conversation.scroll_y < conversation.max_scroll_y
            assert abs(conversation.scroll_y - scroll_before_remaining_thinking) < 2

    asyncio.run(run())


def test_streaming_text_keeps_following_the_bottom_before_completion() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = PausedTextProvider()
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "stream a long reply"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if provider.started.is_set():
                    break

            assert provider.started.is_set()
            for _ in range(20):
                await pilot.pause()

            conversation = app.query_one("#conversation", VerticalScroll)
            assert editor.disabled is True
            assert conversation.max_scroll_y > 0
            assert conversation.scroll_y == conversation.max_scroll_y

            provider.release.set()
            for _ in range(20):
                await pilot.pause()
                if not editor.disabled:
                    break

    asyncio.run(run())


def test_gradually_streamed_text_reflows_and_follows_before_completion() -> None:
    async def run() -> None:
        from textual.containers import VerticalScroll

        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        provider = SlowTextProvider()
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "stream a long reply gradually"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if provider.started.is_set():
                    break

            assert provider.started.is_set()
            await asyncio.sleep(0.15)
            conversation = app.query_one("#conversation", VerticalScroll)
            assert provider.finished.is_set() is False
            assert editor.disabled is True
            assert conversation.max_scroll_y > 0
            assert conversation.scroll_y == conversation.max_scroll_y

            for _ in range(40):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert editor.disabled is False

    asyncio.run(run())


def test_mouse_wheel_pauses_stream_following_until_the_user_returns_to_the_bottom() -> None:
    async def run() -> None:
        from textual import events

        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import ConversationView, PromptEditor

        provider = ControlledTextProvider()
        app = FakuicodeApp(make_config(), provider=provider)

        async with app.run_test(size=(48, 16)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "let me inspect earlier output"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if provider.initial_ready.is_set():
                    break

            assert provider.initial_ready.is_set()
            conversation = app.query_one("#conversation", ConversationView)
            assert conversation.scroll_y == conversation.max_scroll_y

            conversation._on_mouse_scroll_up(
                events.MouseScrollUp(conversation, 1, 1, 0, 1, 0, False, False, False)
            )
            await pilot.pause()
            manual_scroll_y = conversation.scroll_y
            assert manual_scroll_y < conversation.max_scroll_y
            assert app._follow_stream is False

            provider.release_second_batch.set()
            for _ in range(20):
                await pilot.pause()
                if provider.second_batch_ready.is_set():
                    break

            assert provider.second_batch_ready.is_set()
            for _ in range(10):
                await pilot.pause()
            assert conversation.scroll_y == manual_scroll_y
            assert conversation.scroll_y < conversation.max_scroll_y
            assert app._follow_stream is False

            conversation.scroll_to(
                y=conversation.max_scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
            conversation._on_mouse_scroll_down(
                events.MouseScrollDown(conversation, 1, 1, 0, 1, 0, False, False, False)
            )
            await pilot.pause()
            assert app._follow_stream is True

            provider.release_final_batch.set()
            for _ in range(40):
                await pilot.pause()
                if not editor.disabled and conversation.scroll_y == conversation.max_scroll_y:
                    break

            assert editor.disabled is False
            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())


def test_mcp_startup_gates_first_prompt_registers_tools_and_reuses_manager(tmp_path: Path) -> None:
    async def run() -> None:
        from types import MappingProxyType

        from fakuicode.mcp.models import (
            McpConfigSnapshot,
            McpRemoteTool,
            McpServerState,
            McpServerStatus,
            McpStartupSnapshot,
            StdioServerConfig,
            McpConfigSource,
        )
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        class Manager:
            def __init__(self) -> None:
                self.started = 0
                self.closed = 0

            def start(self, configs):
                self.started += 1
                self.current = McpStartupSnapshot(
                    states=(McpServerState("docs", configs[0].transport, McpServerStatus.CONNECTED, 1),)
                )
                return self.current

            def snapshot(self):
                return self.current

            def discovered_tools(self):
                return MappingProxyType(
                    {"docs": (McpRemoteTool("lookup", "Lookup docs", {"type": "object"}),)}
                )

            def close(self) -> None:
                self.closed += 1

        manager = Manager()
        snapshot = McpConfigSnapshot(
            servers=(StdioServerConfig("docs", McpConfigSource.USER, "python"),),
            has_configuration=True,
        )
        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            workspace=tmp_path,
            mcp_snapshot=snapshot,
            mcp_manager_factory=lambda: manager,  # type: ignore[arg-type]
        )
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            for _ in range(30):
                await pilot.pause()
                if not editor.disabled:
                    break
            assert editor.disabled is False
            assert manager.started == 1
            assert app.session.runner.tools.is_known("mcp__docs__lookup")  # type: ignore[union-attr]
            app._new_conversation()
            assert app.session.runner.tools.is_known("mcp__docs__lookup")  # type: ignore[union-attr]
            assert manager.started == 1
            assert "connected" in app._format_mcp_status()
        assert manager.closed == 1

    asyncio.run(run())


def test_project_mcp_trust_prompt_defaults_to_reject_without_connecting(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.mcp.models import McpConfigSnapshot, McpConfigSource, McpServerStatus, StdioServerConfig
        from fakuicode.mcp.trust import McpTrustRepository
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.mcp_trust_prompt import McpTrustPrompt
        from fakuicode.tui.widgets import ConversationView, PromptEditor

        snapshot = McpConfigSnapshot(
            servers=(
                StdioServerConfig(
                    "project",
                    McpConfigSource.PROJECT,
                    "python executable",
                    ("server.py", "value with spaces"),
                ),
            ),
            has_configuration=True,
        )
        app = FakuicodeApp(
            make_config(),
            provider=AgentTextProvider(),
            workspace=tmp_path,
            mcp_snapshot=snapshot,
            mcp_trust_repository=McpTrustRepository(tmp_path / "trust.yaml"),
            mcp_manager_factory=lambda: (_ for _ in ()).throw(AssertionError("must not connect")),
        )
        async with app.run_test() as pilot:
            prompt = app.query_one(McpTrustPrompt)
            options = prompt.query_one(OptionList)
            rendered = "\n".join(item.render().plain for item in prompt.query(Static))
            assert 'command: "python executable"' in rendered
            assert 'argv[0] = "server.py"' in rendered
            assert 'argv[1] = "value with spaces"' in rendered
            assert f"工作目录：{tmp_path.resolve()}" in rendered
            assert options.highlighted == 1
            app.query_one(ConversationView).focus()
            await pilot.pause()
            assert app.focused is options
            await pilot.press("up", "down")
            assert options.highlighted == 1
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if not app.query_one(PromptEditor).disabled:
                    break
            assert app._mcp_states["project"].status is McpServerStatus.TRUST_DENIED
            assert not (tmp_path / "trust.yaml").exists()

    asyncio.run(run())


def test_project_skill_script_prompts_for_fingerprint_trust_before_registration(tmp_path: Path) -> None:
    async def run() -> None:
        import json

        from fakuicode.skills.trust import SkillTrustRepository
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.skill_trust_prompt import SkillTrustPrompt
        from fakuicode.tui.widgets import PromptEditor

        package = tmp_path / ".fakuicode" / "skills" / "scripted"
        (package / "tools").mkdir(parents=True)
        (package / "scripts").mkdir()
        (package / "SKILL.md").write_text(
            "---\nname: scripted\ndescription: scripted workflow\nfakuicode:\n"
            "  invocation: manual\n  execution: shared\n---\nUse the tool.\n",
            encoding="utf-8",
        )
        (package / "scripts" / "echo.py").write_text(
            "import json,sys\nprint(json.dumps({'output':'ok','summary':'ok'}))\n",
            encoding="utf-8",
        )
        (package / "tools" / "echo.json").write_text(
            json.dumps(
                {
                    "name": "echo",
                    "description": "Echo",
                    "input_schema": {"type": "object", "additionalProperties": False},
                    "entrypoint": "scripts/echo.py",
                }
            ),
            encoding="utf-8",
        )
        trust_path = tmp_path / "skill-trust.yaml"
        provider = RequestCaptureAgentProvider()
        app = FakuicodeApp(
            make_config(),
            provider=provider,
            workspace=tmp_path,
            skill_trust_repository=SkillTrustRepository(trust_path),
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/scripted"
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if list(app.query(SkillTrustPrompt)):
                    break
            prompt = app.query_one(SkillTrustPrompt)
            assert prompt.query_one(OptionList).highlighted == 1
            await pilot.press("1")
            for _ in range(60):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert trust_path.exists()
            assert app.skill_manager is not None
            assert "skill__scripted__echo" in app.session.runner.tools.all_names()

    asyncio.run(run())


def test_builtin_test_skill_runs_in_hidden_child_and_returns_summary(tmp_path: Path) -> None:
    async def run() -> None:
        from fakuicode.models import AgentStreamEvent
        from fakuicode.storage import ConversationStore
        from fakuicode.tui.app import FakuicodeApp
        from fakuicode.tui.widgets import PromptEditor

        class ChildProvider:
            def __init__(self, config) -> None:
                self.config = config

            def stream_agent(self, messages, tools, *, request=None, cancel_event=None):
                del messages, tools, request, cancel_event
                yield AgentStreamEvent("text_delta", "3 passed")
                yield AgentStreamEvent("completed")

            def cancel(self) -> None:
                pass

        main_provider = RequestCaptureAgentProvider()
        store = ConversationStore(tmp_path / "conversations.sqlite3")
        app = FakuicodeApp(
            make_config(),
            provider=main_provider,
            provider_factory=ChildProvider,
            store=store,
            workspace=tmp_path,
        )

        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/test changed files"
            await pilot.press("enter")
            for _ in range(80):
                await pilot.pause()
                if not editor.disabled:
                    break

            assert main_provider.requests == []
            assert app.conversation is not None
            children = store.child_conversation_ids(app.conversation.id)
            assert len(children) == 1
            assert store.get_conversation(children[0]).status == "completed"
            assert "3 passed" in app.session.history[-1].content
            assert children[0] in app.session.history[-1].content

    asyncio.run(run())


def test_worktree_sweeper_shutdown_is_bounded_when_sweep_is_unresponsive(
    monkeypatch,
) -> None:
    from threading import Event, Thread
    from time import monotonic

    import fakuicode.tui.app as app_module
    from fakuicode.tui.app import FakuicodeApp

    release = Event()
    worker = Thread(target=release.wait, daemon=True)
    worker.start()
    app = object.__new__(FakuicodeApp)
    object.__setattr__(app, "_worktree_sweep_stop", Event())
    object.__setattr__(app, "_worktree_sweep_thread", worker)
    monkeypatch.setattr(app_module, "DEFAULT_COOPERATIVE_SHUTDOWN_GRACE_SECONDS", 0.02)

    started = monotonic()
    try:
        app._stop_worktree_sweeper()
        assert monotonic() - started < 0.5
        assert app._worktree_sweep_stop.is_set()
        assert app._worktree_sweep_thread is None
    finally:
        release.set()
