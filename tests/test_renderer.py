from __future__ import annotations

from io import StringIO

from rich.console import Console


def make_renderer() -> tuple[object, StringIO]:
    from fakuicode.renderer import Renderer

    output = StringIO()
    return Renderer(Console(file=output, force_terminal=False, color_system=None)), output


def test_renderer_writes_each_text_delta_immediately_and_keeps_markup_literal() -> None:
    from fakuicode.models import StreamEvent

    renderer, output = make_renderer()
    renderer.render(StreamEvent("text_delta", "[bold]literal[/bold]"))

    assert output.getvalue() == "[bold]literal[/bold]"


def test_renderer_marks_thinking_boundaries_and_ends_completed_output_with_newline() -> None:
    from fakuicode.models import StreamEvent

    renderer, output = make_renderer()
    renderer.render(StreamEvent("thinking_start"))
    renderer.render(StreamEvent("thinking_delta", "reason"))
    renderer.render(StreamEvent("thinking_end"))
    renderer.render(StreamEvent("text_delta", "answer"))
    renderer.render(StreamEvent("completed"))

    assert output.getvalue() == "Thinking: reason\nanswer\n"


def test_renderer_keeps_error_message_markup_literal() -> None:
    renderer, output = make_renderer()

    renderer.error("[bold]literal error[/bold]")

    assert "[bold]literal error[/bold]" in output.getvalue()
