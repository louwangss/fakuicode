"""Focused Textual widgets for composing and displaying chat turns."""

from __future__ import annotations

from collections import deque

from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Markdown, Static, TextArea
from textual.widgets.text_area import TextAreaTheme

from fakuicode.commands import CommandRegistry, CommandSuggestion, DEFAULT_COMMAND_REGISTRY
from fakuicode.models import ProviderConfig, ToolCall, ToolResult


APP_VERSION = "0.1.0"


class BrandPanel(Static):
    """Compact, local-only application identity and configuration summary."""

    def __init__(self, config: ProviderConfig, working_directory: str) -> None:
        content = "\n".join(
            (
                f" /\\_/\\    Fakuicode v{APP_VERSION}",
                f"( o.o )   {config.model}",
                f" > ^ <    {working_directory}",
            )
        )
        super().__init__(Text(content), id="brand-panel")


class ConversationView(VerticalScroll):
    """Conversation scroll view that reports intentional mouse-wheel navigation."""

    class UserScrolled(Message):
        """Posted after the scroll view has handled a vertical mouse-wheel event."""

        def __init__(self, conversation: ConversationView, direction: int) -> None:
            super().__init__()
            self.conversation = conversation
            self.direction = direction

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        super()._on_mouse_scroll_up(event)
        if not event.ctrl and not event.shift:
            self.post_message(self.UserScrolled(self, direction=-1))

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        super()._on_mouse_scroll_down(event)
        if not event.ctrl and not event.shift:
            self.post_message(self.UserScrolled(self, direction=1))


class PromptEditor(TextArea):
    """Single-line editor that submits non-blank prompts on Enter."""

    _THEME_NAME = "fakuicode-prompt"

    class Submitted(Message):
        """Posted when a non-blank prompt is ready to send."""

        def __init__(self, editor: PromptEditor, text: str) -> None:
            super().__init__()
            self.editor = editor
            self.text = text

    class CompletionNavigation(Message):
        """Posted when a visible command list should handle a navigation key."""

        def __init__(self, editor: PromptEditor, action: str) -> None:
            super().__init__()
            self.editor = editor
            self.action = action

    def on_mount(self) -> None:
        """Use an explicit high-contrast selection style for long horizontal input."""
        self.register_theme(
            TextAreaTheme(
                self._THEME_NAME,
                selection_style=Style(color="#f8fafc", bgcolor="#075985"),
            )
        )
        self.theme = self._THEME_NAME
        self.fit_to_wrapped_content()

    def on_resize(self, _event: events.Resize) -> None:
        """Recompute the compact height when terminal width changes wrapping."""
        self.call_after_refresh(self.fit_to_wrapped_content)

    def fit_to_wrapped_content(self) -> None:
        """Show one content row normally and at most two when text wraps."""
        self.styles.height = 3 if self.wrapped_document.height <= 1 else 4

    def set_completion_active(self, active: bool) -> None:
        """Let the owning prompt panel reserve navigation keys for candidates."""
        self._completion_active = active

    async def _on_key(self, event: events.Key) -> None:
        if getattr(self, "_completion_active", False) and event.key in {"up", "down", "tab"}:
            event.stop()
            event.prevent_default()
            self.post_message(self.CompletionNavigation(self, event.key))
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.submit()
            return
        if event.key == "ctrl+enter":
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    def apply_command_completion(self, completion: str) -> None:
        """Replace the unfinished command name without submitting it."""
        self.load_text(completion)
        self.cursor_location = self.document.end

    def submit(self) -> None:
        """Emit a non-blank prompt as a single normalized line."""
        text = " ".join(self.text.split())
        if not text or self.disabled:
            return
        self.clear()
        self.post_message(self.Submitted(self, text))


class CommandCompletionList(Static):
    """A compact, non-interactive view of command candidates above the editor."""

    _WINDOW_SIZE = 5

    def __init__(self) -> None:
        super().__init__(Text(""), id="command-completion")
        self.display = False

    def set_suggestions(
        self,
        suggestions: tuple[CommandSuggestion, ...],
        highlighted_index: int | None,
        *,
        show_empty: bool,
    ) -> None:
        """Render candidates around the active selection or a local empty state."""
        if not suggestions:
            self.display = show_empty
            self.update(Text("No matching command", style="dim"))
            return

        assert highlighted_index is not None
        start = max(0, min(highlighted_index - self._WINDOW_SIZE // 2, len(suggestions) - self._WINDOW_SIZE))
        stop = min(len(suggestions), start + self._WINDOW_SIZE)
        content = Text()
        for index in range(start, stop):
            suggestion = suggestions[index]
            selected = index == highlighted_index
            content.append("› " if selected else "  ", style="bold #5eead4" if selected else "#6b7280")
            content.append(suggestion.completion.rstrip(), style="bold #e5e7eb" if selected else "#c4c4c4")
            content.append(f"  {suggestion.description}", style="#9ca3af")
            if index < stop - 1:
                content.append("\n")
        self.update(content)
        self.display = True


class PromptPanel(Vertical):
    """A single-line editor with persistent status and model information."""

    def __init__(
        self,
        model: str,
        *,
        command_registry: CommandRegistry = DEFAULT_COMMAND_REGISTRY,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._command_registry = command_registry
        self._suggestions: tuple[CommandSuggestion, ...] = ()
        self._highlighted_index: int | None = None

    def compose(self) -> ComposeResult:
        yield CommandCompletionList()
        yield PromptEditor(id="prompt-editor", soft_wrap=True)
        with Horizontal(id="prompt-info"):
            yield Static(Text("[DEFAULT] Ready"), id="status")
            yield Static(self._model, id="footer-model")

    @on(TextArea.Changed)
    def _refresh_command_completions(self, message: TextArea.Changed) -> None:
        editor = self.query_one(PromptEditor)
        if message.text_area is not editor:
            return
        self._suggestions = self._command_registry.suggest(editor.text)
        self._highlighted_index = 0 if self._suggestions else None
        self._update_completion_view(editor.text)
        editor.fit_to_wrapped_content()

    @on(PromptEditor.CompletionNavigation)
    def _handle_completion_navigation(self, message: PromptEditor.CompletionNavigation) -> None:
        editor = self.query_one(PromptEditor)
        if message.editor is not editor or not self._suggestions or self._highlighted_index is None:
            return
        if message.action == "up":
            self._highlighted_index = (self._highlighted_index - 1) % len(self._suggestions)
            self._update_completion_view(editor.text)
            return
        if message.action == "down":
            self._highlighted_index = (self._highlighted_index + 1) % len(self._suggestions)
            self._update_completion_view(editor.text)
            return
        if message.action == "tab":
            editor.apply_command_completion(self._suggestions[self._highlighted_index].completion)
            self._suggestions = self._command_registry.suggest(editor.text)
            self._highlighted_index = 0 if self._suggestions else None
            self._update_completion_view(editor.text)

    def _update_completion_view(self, text: str) -> None:
        self.query_one(CommandCompletionList).set_suggestions(
            self._suggestions,
            self._highlighted_index,
            show_empty=self._command_registry.should_show_empty_completion(text),
        )
        self.query_one(PromptEditor).set_completion_active(bool(self._suggestions))

    def set_model(self, model: str) -> None:
        self._model = model
        self.query_one("#footer-model", Static).update(Text(model))

    def set_command_registry(self, registry: CommandRegistry) -> None:
        self._command_registry = registry
        self._suggestions = ()
        self._highlighted_index = None
        if self.is_mounted:
            self._update_completion_view(self.query_one(PromptEditor).text)


class UserMessage(Static):
    """Literal user prompt rendering."""

    def __init__(self, content: str) -> None:
        super().__init__(Text(content), classes="user-message")


class SystemNotice(Static):
    """A local-only notification, never sent to the provider."""

    def __init__(self, content: str) -> None:
        super().__init__(Text(content), classes="system-notice")


class AssistantTurn(Vertical):
    """One assistant response, from literal stream text to final Markdown."""

    def __init__(self) -> None:
        super().__init__(classes="assistant-turn")
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._thinking_text = Text()
        self._stream_text = Text()
        self._activity: ToolActivity | None = None
        self._mcp_activities: dict[str, ToolActivity] = {}
        self._thinking_body = Static(self._thinking_text, classes="thinking-body")
        self._thinking = Collapsible(self._thinking_body, title="Thinking", collapsed=True, classes="thinking")
        self._activities = Vertical(classes="tool-activities")
        self._stream = Static(Text(""), classes="assistant-stream")
        self._final = Markdown("", classes="assistant-markdown")
        self._thinking.display = False
        self._final.display = False

    def compose(self) -> ComposeResult:
        yield self._thinking
        yield self._activities
        yield self._stream
        yield self._final

    def show_tool_call(self, call: ToolCall) -> ToolActivity:
        """Show all tool progress through one reusable status line."""
        if call.name.startswith("mcp__"):
            activity = self._mcp_activities.get(call.id)
            if activity is None:
                activity = ToolActivity(call)
                self._mcp_activities[call.id] = activity
                if self.is_mounted and self._activities.is_mounted:
                    self._activities.mount(activity)
            return activity
        if self._activity is None:
            self._activity = ToolActivity(call)
            if self.is_mounted and self._activities.is_mounted:
                self._activities.mount(self._activity)
        else:
            self._activity.start(call)
        return self._activity

    def show_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        activity = self.show_tool_call(call)
        activity.complete(result, call)

    def on_mount(self) -> None:
        if self._activity is not None and not self._activity.is_mounted:
            self._activities.mount(self._activity)
        for activity in self._mcp_activities.values():
            if not activity.is_mounted:
                self._activities.mount(activity)

    def append_text(self, text: str) -> None:
        self._text_parts.append(text)
        self._stream_text.append(text)
        self._stream.update(self._stream_text)

    def start_thinking(self) -> None:
        self._thinking.display = True

    def append_thinking(self, text: str) -> None:
        self._thinking_parts.append(text)
        self._thinking_text.append(text)
        self._thinking_body.update(self._thinking_text)

    @on(Collapsible.Toggled)
    def _sync_height_after_thinking_toggle(self, event: Collapsible.Toggled) -> None:
        if event.collapsible is self._thinking:
            self.styles.height = "auto"

    async def finalize(self) -> None:
        """Render the completed Markdown before the conversation is relaid out."""
        answer = "".join(self._text_parts)
        await self._final.update(answer)
        self._stream.display = False
        self._final.display = True
        self.styles.height = "auto"
        self.refresh(layout=True)
        if self.parent is not None:
            self.parent.refresh(layout=True)

    def show_error(self, message: str) -> None:
        self._final.display = False
        self._stream.update(Text(f"Error: {message}", style="red"))
        self._stream.display = True


class ToolActivity(Static):
    """A one-line tool status that updates in place."""

    _COMPLETION_DWELL_SECONDS = 0.15

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self._completion_queue: deque[tuple[ToolCall, ToolResult]] = deque()
        self._showing_completion = False
        self._pending_running_call: ToolCall | None = None
        super().__init__(_tool_status(call, "Running", "yellow"), classes="tool-activity")

    def start(self, call: ToolCall) -> None:
        if self._showing_completion:
            self._pending_running_call = call
            return
        self.call = call
        self.update(_tool_status(call, "Running", "yellow"))

    def complete(self, result: ToolResult, call: ToolCall | None = None) -> None:
        completion_call = call or self.call
        if self._pending_running_call is not None and self._pending_running_call.id == result.call_id:
            completion_call = self._pending_running_call
            self._pending_running_call = None
        if not self.is_mounted:
            self._render_completion(completion_call, result)
            return
        if self._showing_completion:
            self._completion_queue.append((completion_call, result))
            return
        self._render_completion(completion_call, result)
        self._showing_completion = True
        self.set_timer(self._COMPLETION_DWELL_SECONDS, self._advance_completion)

    def _advance_completion(self) -> None:
        if self._completion_queue:
            call, result = self._completion_queue.popleft()
            self._render_completion(call, result)
            self.set_timer(self._COMPLETION_DWELL_SECONDS, self._advance_completion)
            return
        self._showing_completion = False
        if self._pending_running_call is not None:
            call = self._pending_running_call
            self._pending_running_call = None
            self.start(call)

    def _render_completion(self, call: ToolCall, result: ToolResult) -> None:
        self.call = call
        state = "Done" if result.success else "Failed"
        detail = "" if result.success else _single_line(result.summary or result.output)
        self.update(
            _tool_status(
                call,
                state,
                "green" if result.success else "red",
                detail,
                duration_seconds=result.duration_seconds,
            )
        )


def _tool_status(
    call: ToolCall,
    state: str,
    style: str,
    detail: str = "",
    *,
    duration_seconds: float | None = None,
) -> Text:
    marker = {"Running": "●", "Done": "✓", "Failed": "✗"}[state]
    if call.name.startswith("mcp__"):
        content = f"{marker} {call.name} · {state}"
        if duration_seconds is not None:
            content += f" ({duration_seconds:.1f}s)"
    else:
        content = f"{marker} {call.name} · {_tool_target(call)} · {state}"
    if detail:
        content += f": {detail}"
    return Text(content, style=style, no_wrap=True, overflow="ellipsis")


def _single_line(content: str, *, limit: int = 120) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _tool_target(call: ToolCall) -> str:
    scope = call.arguments.get("path")
    scope_text = scope if isinstance(scope, str) else None
    for key in ("query", "pattern"):
        value = call.arguments.get(key)
        if isinstance(value, str):
            return f"{value} in {scope_text}" if scope_text else value
    if scope_text:
        return scope_text
    value = call.arguments.get("command")
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " ".join(value)
    return "workspace"
