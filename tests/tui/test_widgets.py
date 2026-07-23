from __future__ import annotations

import asyncio

from rich.segment import Segment


def _brand_render_lines(width: int) -> list[list[Segment]]:
    from rich.console import Console

    from fakuicode.models import ProviderConfig
    from fakuicode.tui.widgets import BrandPanel

    config = ProviderConfig(
        "anthropic",
        "claude-test[bold]",
        "https://api.example.test/v1",
        "never-show-this-key",
    )
    panel = BrandPanel(config, r"C:\Users\example\Desktop\fakuicode\[literal]")
    console = Console(width=width, color_system="truecolor", force_terminal=True)
    return console.render_lines(panel.render(), console.options.update(width=width), pad=False)


def _rendered_line_text(line: list[Segment]) -> str:
    return "".join(segment.text for segment in line)


def _colored_columns(line: list[Segment]) -> list[int]:
    columns: list[int] = []
    offset = 0
    for segment in line:
        text = segment.text
        style = segment.style
        if style is not None and (style.color is not None or style.bgcolor is not None):
            columns.extend(range(offset, offset + len(text)))
        offset += len(text)
    return columns


def test_brand_logo_grid_has_the_confirmed_dimensions_and_palette() -> None:
    from fakuicode.tui.widgets import _BRAND_LOGO_GRID

    assert len(_BRAND_LOGO_GRID) == 16
    assert {len(row) for row in _BRAND_LOGO_GRID} == {24}
    assert set("".join(_BRAND_LOGO_GRID)) == {".", "R", "D", "G"}


def test_brand_logo_half_cells_reconstruct_every_logical_pixel() -> None:
    from rich.console import Console

    from fakuicode.tui.widgets import _BRAND_LOGO_GRID, _render_brand_logo

    logo = _render_brand_logo()
    console = Console(color_system="truecolor")
    markers_by_color = {
        "#ef4444": "R",
        "#7f1d1d": "D",
        "#d1d5db": "G",
    }
    rendered_lines = logo.plain.splitlines()

    assert len(rendered_lines) == 8
    assert {len(line) for line in rendered_lines} == {24}
    assert set(logo.plain.replace("\n", "")) <= {" ", "▀", "▄"}

    reconstructed: list[str] = []
    for logical_row in range(16):
        row: list[str] = []
        for column in range(24):
            offset = (logical_row // 2) * 25 + column
            glyph = logo.plain[offset]
            style = logo.get_style_at_offset(console, offset)
            if glyph == " ":
                color = style.bgcolor
            elif glyph == "▀":
                color = style.color if logical_row % 2 == 0 else style.bgcolor
            else:
                assert glyph == "▄"
                color = style.bgcolor if logical_row % 2 == 0 else style.color
            color_hex = color.get_truecolor().hex if color is not None else None
            row.append(markers_by_color.get(color_hex, "."))
        reconstructed.append("".join(row))

    assert tuple(reconstructed) == _BRAND_LOGO_GRID


def test_brand_panel_keeps_literal_metadata_without_sensitive_configuration() -> None:
    lines = _brand_render_lines(100)
    visible = "\n".join(_rendered_line_text(line) for line in lines)

    assert "Fakuicode v0.1.0" in visible
    assert "claude-test[bold]" in visible
    assert r"C:\Users\example\Desktop\fakuicode\[literal]" in visible
    assert "anthropic" not in visible.lower()
    assert "never-show-this-key" not in visible
    assert "MCP" not in visible
    assert "tool" not in visible.lower()


def test_brand_panel_places_information_close_to_and_centered_on_logo_when_wide() -> None:
    lines = _brand_render_lines(100)
    visible_lines = [_rendered_line_text(line) for line in lines]
    title_row = next(index for index, line in enumerate(visible_lines) if "Fakuicode" in line)
    model_row = next(index for index, line in enumerate(visible_lines) if "claude-test[bold]" in line)
    directory_row = next(
        index
        for index, line in enumerate(visible_lines)
        if r"C:\Users\example\Desktop\fakuicode\[literal]" in line
    )
    colored_rows = [index for index, line in enumerate(lines) if _colored_columns(line)]

    assert colored_rows == list(range(8))
    assert (title_row, model_row, directory_row) == (2, 3, 4)
    assert visible_lines[title_row].index("Fakuicode") == 27
    assert visible_lines[model_row].index("claude-test[bold]") == 27
    assert visible_lines[directory_row].index(r"C:\Users\example") == 27


def test_brand_panel_stacks_information_above_complete_logo_when_narrow() -> None:
    lines = _brand_render_lines(44)
    visible_lines = [_rendered_line_text(line) for line in lines]
    title_row = next(index for index, line in enumerate(visible_lines) if "Fakuicode" in line)
    model_row = next(index for index, line in enumerate(visible_lines) if "claude-test[bold]" in line)
    directory_row = next(
        index
        for index, line in enumerate(visible_lines)
        if r"C:\Users\example\Desktop\fakuicode\[literal]" in line
    )
    colored_rows = [index for index, line in enumerate(lines) if _colored_columns(line)]

    assert title_row < colored_rows[0]
    assert model_row < colored_rows[0]
    assert directory_row < colored_rows[0]
    assert len(colored_rows) == 8


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


def test_subagent_result_notice_uses_error_when_no_result_text_exists() -> None:
    async def run() -> None:
        from textual.app import App, ComposeResult

        from fakuicode.tui.widgets import SubagentResultNotice

        class WidgetApp(App[None]):
            def compose(self) -> ComposeResult:
                yield SubagentResultNotice(
                    task_id="task-failed",
                    name="reviewer",
                    status="failed",
                    result="",
                    error="权限被拒绝",
                )

        app = WidgetApp()
        async with app.run_test():
            report = app.query_one(SubagentResultNotice)
            assert "失败" in report.title
            assert "task-failed" in report.title
            assert report.result_body.render().plain == "权限被拒绝"

    asyncio.run(run())
