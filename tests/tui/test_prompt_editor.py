from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult


class PromptCaptureApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        from fakuicode.tui.widgets import PromptPanel

        yield PromptPanel("test-model")

    def on_prompt_editor_submitted(self, message: object) -> None:
        self.submitted.append(message.text)


def test_enter_submits_single_line_text_but_not_empty_input() -> None:
    async def run() -> None:
        from fakuicode.tui.widgets import PromptEditor

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            await pilot.press("enter")
            assert app.submitted == []

            editor.text = "first line\nsecond line\n\nthird line"
            await pilot.press("enter")
            assert app.submitted == ["first line second line third line"]
            assert editor.text == ""

    asyncio.run(run())


def test_ctrl_enter_does_not_insert_a_line_break_or_submit() -> None:
    async def run() -> None:
        from fakuicode.tui.widgets import PromptEditor

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "hello"
            editor.cursor_location = editor.document.end
            await pilot.press("ctrl+enter")
            assert editor.text == "hello"
            assert app.submitted == []

    asyncio.run(run())


def test_prompt_panel_has_no_line_break_controls_or_hint() -> None:
    async def run() -> None:
        from textual.widgets import Static

        from fakuicode.tui.widgets import PromptPanel

        app = PromptCaptureApp()
        async with app.run_test():
            panel = app.query_one(PromptPanel)
            assert list(panel.query("#insert-newline")) == []
            assert list(panel.query("#shortcut-hint")) == []
            assert panel.query_one("#status", Static).render().plain == "[DEFAULT] Ready"
            assert panel.query_one("#footer-model", Static).render().plain == "test-model"

    asyncio.run(run())


def test_command_completion_navigates_with_arrow_keys_and_tab_never_submits() -> None:
    async def run() -> None:
        from fakuicode.tui.widgets import CommandCompletionList, PromptEditor

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/s"
            editor.cursor_location = editor.document.end
            await pilot.pause()

            candidates = app.query_one(CommandCompletionList)
            assert candidates.display
            assert "/sessions" in candidates.render().plain
            assert "/status" in candidates.render().plain

            await pilot.press("down", "tab")
            await pilot.pause()

            assert editor.text == "/status"
            assert app.submitted == []
            assert not candidates.display

    asyncio.run(run())


def test_dynamic_skill_command_uses_the_same_real_completion_control() -> None:
    async def run() -> None:
        from fakuicode.commands import compose_command_registry
        from fakuicode.tui.widgets import CommandCompletionList, PromptEditor, PromptPanel

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            panel = app.query_one(PromptPanel)
            panel.set_command_registry(compose_command_registry((("hot-check", "Hot workflow"),)))
            editor = app.query_one(PromptEditor)
            editor.text = "/hot"
            editor.cursor_location = editor.document.end
            await pilot.pause()

            assert "/hot-check" in app.query_one(CommandCompletionList).render().plain
            await pilot.press("tab")
            await pilot.pause()

            assert editor.text == "/hot-check "
            assert app.submitted == []

    asyncio.run(run())


def test_command_completion_reports_a_missing_command_without_breaking_the_editor() -> None:
    async def run() -> None:
        from fakuicode.tui.widgets import CommandCompletionList, PromptEditor

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/not-a-command"
            editor.cursor_location = editor.document.end
            await pilot.pause()

            candidates = app.query_one(CommandCompletionList)
            assert candidates.display
            assert "No matching command" in candidates.render().plain
            assert not editor.disabled

    asyncio.run(run())


def test_alias_completion_deduplicates_to_the_canonical_command() -> None:
    async def run() -> None:
        from fakuicode.tui.widgets import CommandCompletionList, PromptEditor

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/sessio"
            editor.cursor_location = editor.document.end
            await pilot.pause()

            candidates = app.query_one(CommandCompletionList)
            assert candidates.render().plain.count("/sessions") == 1

            await pilot.press("tab")
            await pilot.pause()

            assert editor.text == "/sessions"
            assert app.submitted == []

    asyncio.run(run())


def test_memory_completion_filters_options_after_whitespace_and_tab_never_submits() -> None:
    async def run() -> None:
        from fakuicode.tui.widgets import CommandCompletionList, PromptEditor

        app = PromptCaptureApp()
        async with app.run_test() as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "/memory "
            editor.cursor_location = editor.document.end
            await pilot.pause()

            candidates = app.query_one(CommandCompletionList)
            assert candidates.display
            assert "/memory on" in candidates.render().plain
            assert "/memory off" in candidates.render().plain
            assert "/memory forget" in candidates.render().plain

            editor.text = "/memory o"
            editor.cursor_location = editor.document.end
            await pilot.pause()
            await pilot.press("down", "tab")
            await pilot.pause()

            assert editor.text == "/memory off"
            assert app.submitted == []
            assert not candidates.display

    asyncio.run(run())


def test_long_prompt_wraps_to_two_visible_rows_without_a_horizontal_scrollbar() -> None:
    async def run() -> None:
        from rich.color import Color

        from fakuicode.tui.widgets import PromptEditor

        app = PromptCaptureApp()
        async with app.run_test(size=(42, 18)) as pilot:
            editor = app.query_one(PromptEditor)
            editor.text = "short prompt"
            await pilot.pause()
            assert editor.region.height == 3

            long_prompt = "l" * 200
            editor.text = long_prompt
            editor.cursor_location = editor.document.end
            editor.select_all()
            await pilot.pause()

            selection_style = editor._theme.selection_style
            assert editor.text == long_prompt
            assert editor.soft_wrap is True
            assert editor.wrapped_document.height > 1
            assert editor.region.height == 4
            assert editor.max_scroll_x == 0
            assert editor.show_horizontal_scrollbar is False
            assert selection_style is not None
            assert selection_style.color == Color.parse("#f8fafc")
            assert selection_style.bgcolor == Color.parse("#075985")

    asyncio.run(run())
