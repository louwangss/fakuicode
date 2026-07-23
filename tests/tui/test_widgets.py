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
    panel = BrandPanel(config, r"C:\work\[literal]")
    console = Console(width=width, color_system="truecolor", force_terminal=True)
    return console.render_lines(panel.render(), console.options.update(width=width), pad=False)


def _rendered_line_text(line: list[Segment]) -> str:
    return "".join(segment.text for segment in line)


def _background_columns(line: list[Segment]) -> list[int]:
    columns: list[int] = []
    offset = 0
    for segment in line:
        text = segment.text
        style = segment.style
        if style is not None and style.bgcolor is not None:
            columns.extend(range(offset, offset + len(text)))
        offset += len(text)
    return columns


def test_brand_logo_grid_has_the_confirmed_dimensions_and_palette() -> None:
    from fakuicode.tui.widgets import _BRAND_LOGO_GRID

    assert len(_BRAND_LOGO_GRID) == 24
    assert {len(row) for row in _BRAND_LOGO_GRID} == {35}
    assert set("".join(_BRAND_LOGO_GRID)) == {".", "R", "D", "G"}


def test_brand_logo_renders_exact_background_colors_and_transparency() -> None:
    from rich.console import Console

    from fakuicode.tui.widgets import _BRAND_LOGO_GRID, _render_brand_logo

    logo = _render_brand_logo()
    console = Console(color_system="truecolor")
    expected_colors = {"R": "#ef4444", "D": "#7f1d1d", "G": "#d1d5db"}

    assert logo.plain == "\n".join(" " * 35 for _ in range(24))
    for marker, expected in expected_colors.items():
        row = next(index for index, value in enumerate(_BRAND_LOGO_GRID) if marker in value)
        column = _BRAND_LOGO_GRID[row].index(marker)
        style = logo.get_style_at_offset(console, row * 36 + column)
        assert style.bgcolor is not None
        assert style.bgcolor.get_truecolor().hex == expected

    transparent_style = logo.get_style_at_offset(console, 0)
    assert transparent_style.bgcolor is None


def test_brand_panel_keeps_literal_metadata_without_sensitive_configuration() -> None:
    lines = _brand_render_lines(100)
    visible = "\n".join(_rendered_line_text(line) for line in lines)

    assert "Fakuicode v0.1.0" in visible
    assert "claude-test[bold]" in visible
    assert r"C:\work\[literal]" in visible
    assert "anthropic" not in visible.lower()
    assert "never-show-this-key" not in visible
    assert "MCP" not in visible
    assert "tool" not in visible.lower()


def test_brand_panel_places_logo_left_of_information_when_wide() -> None:
    lines = _brand_render_lines(100)
    title_row = next(index for index, line in enumerate(lines) if "Fakuicode" in _rendered_line_text(line))
    title_line = _rendered_line_text(lines[title_row])
    colored_columns = _background_columns(lines[title_row])

    assert colored_columns
    assert min(colored_columns) < title_line.index("Fakuicode")


def test_brand_panel_stacks_information_above_complete_logo_when_narrow() -> None:
    lines = _brand_render_lines(44)
    visible_lines = [_rendered_line_text(line) for line in lines]
    title_row = next(index for index, line in enumerate(visible_lines) if "Fakuicode" in line)
    model_row = next(index for index, line in enumerate(visible_lines) if "claude-test[bold]" in line)
    directory_row = next(index for index, line in enumerate(visible_lines) if r"C:\work\[literal]" in line)
    colored_rows = [index for index, line in enumerate(lines) if _background_columns(line)]

    assert title_row < colored_rows[0]
    assert model_row < colored_rows[0]
    assert directory_row < colored_rows[0]
    assert len(colored_rows) == 24


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
