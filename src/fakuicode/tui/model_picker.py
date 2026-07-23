"""Keyboard-first model profile selection for the Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

@dataclass(frozen=True)
class ProfileChoice:
    """The safe subset of a profile that may be rendered in the picker."""

    profile_name: str
    model_name: str


@dataclass(frozen=True)
class SessionChoice:
    """Conversation details that are safe and useful in the local resume picker."""

    conversation_id: str
    title: str
    profile_name: str
    updated_at: int
    message_count: int | None = None


@dataclass(frozen=True)
class MemoryChoice:
    """Memory fields that are safe and useful in the local picker."""

    entry_id: str
    scope: str
    category: str
    summary: str


class PickerFilterInput(Input):
    """Keeps filtering focus while forwarding picker navigation keys."""

    class Navigation(Message):
        """A key that the enclosing model picker owns."""

        def __init__(self, input_widget: PickerFilterInput, action: str) -> None:
            super().__init__()
            self.input_widget = input_widget
            self.action = action

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"up", "down", "enter", "escape"}:
            event.stop()
            event.prevent_default()
            self.post_message(self.Navigation(self, event.key))
            return
        await super()._on_key(event)


class ModelPicker(ModalScreen[str | None]):
    """Filter and choose one configured profile without touching provider data."""

    def __init__(self, choices: tuple[ProfileChoice, ...], current_profile: str) -> None:
        super().__init__()
        self._choices = choices
        self._current_profile = current_profile
        self._filtered_choices: tuple[ProfileChoice, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-dialog"):
            yield Static("Choose model", id="model-picker-title")
            yield PickerFilterInput(placeholder="Search profiles or models", id="model-filter")
            yield OptionList(id="model-options", markup=False)
            yield Static("No matching profiles", id="model-picker-empty")

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one(PickerFilterInput).focus()

    @on(Input.Changed, "#model-filter")
    def _filter_choices(self, message: Input.Changed) -> None:
        self._refresh_options(message.value)

    @on(PickerFilterInput.Navigation)
    def _handle_navigation(self, message: PickerFilterInput.Navigation) -> None:
        if message.input_widget is not self.query_one(PickerFilterInput):
            return
        options = self.query_one(OptionList)
        if message.action == "up":
            options.action_cursor_up()
        elif message.action == "down":
            options.action_cursor_down()
        elif message.action == "enter":
            self._confirm_highlighted_option()
        elif message.action == "escape":
            self.dismiss(None)

    @on(OptionList.OptionSelected, "#model-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        self._dismiss_choice_at(message.option_index)

    def _refresh_options(self, query: str) -> None:
        needle = query.casefold().strip()
        self._filtered_choices = tuple(
            choice
            for choice in self._choices
            if not needle or needle in choice.profile_name.casefold() or needle in choice.model_name.casefold()
        )
        options = self.query_one(OptionList)
        options.set_options(
            Option(f"{choice.profile_name}  ·  {choice.model_name}") for choice in self._filtered_choices
        )
        has_choices = bool(self._filtered_choices)
        options.display = has_choices
        self.query_one("#model-picker-empty", Static).display = not has_choices
        if not has_choices:
            return

        current_index = next(
            (index for index, choice in enumerate(self._filtered_choices) if choice.profile_name == self._current_profile),
            0,
        )
        options.highlighted = current_index

    def _confirm_highlighted_option(self) -> None:
        highlighted = self.query_one(OptionList).highlighted
        if highlighted is not None:
            self._dismiss_choice_at(highlighted)

    def _dismiss_choice_at(self, index: int) -> None:
        if 0 <= index < len(self._filtered_choices):
            self.dismiss(self._filtered_choices[index].profile_name)


class SessionPicker(ModalScreen[str | None]):
    """Choose a persisted conversation using only the keyboard."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        choices: tuple[SessionChoice, ...],
        current_conversation_id: str | None,
        *,
        purpose: Literal["resume", "delete"] = "resume",
    ) -> None:
        super().__init__()
        self._choices = choices
        self._current_conversation_id = current_conversation_id
        self._purpose = purpose

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker-dialog"):
            action = "Resume conversation" if self._purpose == "resume" else "Delete conversation"
            yield Static(f"{action} · ↑/↓ select · Enter choose · Esc cancel", id="session-picker-title")
            yield OptionList(id="session-options", markup=False)
            yield Static("No saved conversations", id="session-picker-empty")

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.set_options(
            Option(
                f"{choice.title}  ·  {_format_session_time(choice.updated_at)}  ·  {choice.profile_name}"
                f"  ·  {_format_message_count(choice.message_count)}"
                + ("  (current)" if choice.conversation_id == self._current_conversation_id else "")
            )
            for choice in self._choices
        )
        has_choices = bool(self._choices)
        options.display = has_choices
        self.query_one("#session-picker-empty", Static).display = not has_choices
        if not has_choices:
            return
        options.highlighted = next(
            (index for index, choice in enumerate(self._choices) if choice.conversation_id == self._current_conversation_id),
            0,
        )
        options.focus()

    @on(OptionList.OptionSelected, "#session-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        if 0 <= message.option_index < len(self._choices):
            self.dismiss(self._choices[message.option_index].conversation_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MemoryPicker(ModalScreen[str | None]):
    """Filter safe memory summaries and return only the selected entry id."""

    def __init__(self, choices: tuple[MemoryChoice, ...]) -> None:
        super().__init__()
        self._choices = choices
        self._filtered_choices: tuple[MemoryChoice, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-picker-dialog"):
            yield Static("Forget memory · ↑/↓ select · Enter choose · Esc cancel", id="memory-picker-title")
            yield PickerFilterInput(placeholder="Search memory summaries", id="memory-filter")
            yield OptionList(id="memory-options", markup=False)
            yield Static("No matching memory entries", id="memory-picker-empty")

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one(PickerFilterInput).focus()

    @on(Input.Changed, "#memory-filter")
    def _filter_choices(self, message: Input.Changed) -> None:
        self._refresh_options(message.value)

    @on(PickerFilterInput.Navigation)
    def _handle_navigation(self, message: PickerFilterInput.Navigation) -> None:
        if message.input_widget is not self.query_one(PickerFilterInput):
            return
        options = self.query_one(OptionList)
        if message.action == "up":
            options.action_cursor_up()
        elif message.action == "down":
            options.action_cursor_down()
        elif message.action == "enter":
            self._confirm_highlighted_option()
        elif message.action == "escape":
            self.dismiss(None)

    @on(OptionList.OptionSelected, "#memory-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        self._dismiss_choice_at(message.option_index)

    def _refresh_options(self, query: str) -> None:
        needle = query.casefold().strip()
        self._filtered_choices = tuple(
            choice
            for choice in self._choices
            if not needle
            or needle in choice.summary.casefold()
            or needle in choice.category.casefold()
            or needle in choice.scope.casefold()
        )
        options = self.query_one(OptionList)
        options.set_options(
            Option(f"{choice.summary}  ·  {choice.scope}  ·  {choice.category}")
            for choice in self._filtered_choices
        )
        has_choices = bool(self._filtered_choices)
        options.display = has_choices
        self.query_one("#memory-picker-empty", Static).display = not has_choices
        if has_choices:
            options.highlighted = 0

    def _confirm_highlighted_option(self) -> None:
        highlighted = self.query_one(OptionList).highlighted
        if highlighted is not None:
            self._dismiss_choice_at(highlighted)

    def _dismiss_choice_at(self, index: int) -> None:
        if 0 <= index < len(self._filtered_choices):
            self.dismiss(self._filtered_choices[index].entry_id)


class ConfirmationScreen(ModalScreen[bool]):
    """Confirm one destructive local action with cancellation selected by default."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str, confirm_label: str) -> None:
        super().__init__()
        self._prompt = prompt
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation-dialog"):
            yield Static(self._prompt, id="confirmation-title", markup=False)
            yield OptionList(
                Option("Cancel", id="cancel"),
                Option(self._confirm_label, id="confirm"),
                id="confirmation-options",
                markup=False,
            )

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 0
        options.focus()

    @on(OptionList.OptionSelected, "#confirmation-options")
    def _select_option(self, message: OptionList.OptionSelected) -> None:
        self.dismiss(message.option.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


def _format_session_time(timestamp_ns: int) -> str:
    """Render a local, compact last-activity time from durable nanoseconds."""
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M")


def _format_message_count(count: int | None) -> str:
    return f"{count} messages" if count is not None else "? messages"
