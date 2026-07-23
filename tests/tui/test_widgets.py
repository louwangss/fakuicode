from __future__ import annotations

import asyncio


def test_tool_activity_displays_a_compact_success_without_output_body() -> None:
    async def run() -> None:
        from textual.app import App, ComposeResult

        from fakuicode.models import ToolCall, ToolResult
        from fakuicode.tui.widgets import ToolActivity

        class WidgetApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ToolActivity(ToolCall("call-1", "read_file", {"path": "README.md"}))

        app = WidgetApp()
        async with app.run_test() as pilot:
            activity = app.query_one(ToolActivity)
            activity.complete(ToolResult("call-1", "read_file", True, "x" * 4_001, "read README.md"))
            await pilot.pause()
            line = activity.render().plain
            assert activity.styles.height.value == 1
            assert "Done" in line
            assert "README.md" in line
            assert "x" * 20 not in line

    asyncio.run(run())


def test_tool_activity_displays_search_target_and_scope() -> None:
    async def run() -> None:
        from textual.app import App, ComposeResult

        from fakuicode.models import ToolCall
        from fakuicode.tui.widgets import ToolActivity

        class WidgetApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ToolActivity(ToolCall("call-1", "search_code", {"query": "needle", "path": "src"}))

        app = WidgetApp()
        async with app.run_test() as pilot:
            activity = app.query_one(ToolActivity)
            await pilot.pause()
            line = activity.render().plain
            assert "Running" in line
            assert "needle" in line
            assert "src" in line

    asyncio.run(run())


def test_tool_activity_keeps_a_failed_summary_on_one_bounded_line() -> None:
    async def run() -> None:
        from textual.app import App, ComposeResult

        from fakuicode.models import ToolCall, ToolResult
        from fakuicode.tui.widgets import ToolActivity

        class WidgetApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ToolActivity(ToolCall("call-1", "write_file", {"path": "test/output.txt"}))

        app = WidgetApp()
        async with app.run_test() as pilot:
            activity = app.query_one(ToolActivity)
            activity.complete(ToolResult("call-1", "write_file", False, "details", "failed " * 100))
            await pilot.pause()
            line = activity.render().plain
            assert activity.styles.height.value == 1
            assert "Failed" in line
            assert "test/output.txt" in line
            assert "\n" not in line
            assert len(line) < 200

    asyncio.run(run())


def test_tool_activity_cycles_completions_in_place() -> None:
    async def run() -> None:
        from textual.app import App, ComposeResult

        from fakuicode.models import ToolCall, ToolResult
        from fakuicode.tui.widgets import ToolActivity

        first = ToolCall("call-1", "read_file", {"path": "first.py"})
        second = ToolCall("call-2", "read_file", {"path": "second.py"})

        class WidgetApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ToolActivity(first)

        app = WidgetApp()
        async with app.run_test() as pilot:
            activity = app.query_one(ToolActivity)
            activity.complete(ToolResult("call-1", "read_file", True, "first", "read first.py"))
            activity.start(second)
            activity.complete(ToolResult("call-2", "read_file", True, "second", "read second.py"))
            assert len(list(app.query(ToolActivity))) == 1
            assert "first.py" in activity.render().plain

            await pilot.pause(0.2)
            assert "second.py" in activity.render().plain
            assert "Done" in activity.render().plain

    asyncio.run(run())
