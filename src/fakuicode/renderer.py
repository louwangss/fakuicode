"""Readable terminal rendering for provider stream events."""

from rich.console import Console
from rich.text import Text

from fakuicode.models import StreamEvent


class Renderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def render(self, event: StreamEvent) -> None:
        if event.kind == "thinking_start":
            self.console.print(Text("Thinking: ", style="dim"), end="")
        elif event.kind in {"thinking_delta", "text_delta"}:
            self.console.print(event.text, end="", markup=False)
        elif event.kind in {"thinking_end", "completed"}:
            self.console.print()

    def error(self, message: str) -> None:
        self.console.print(Text(f"Error: {message}", style="red"))
