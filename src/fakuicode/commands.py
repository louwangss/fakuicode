"""Registration, parsing, completion, and dispatch for slash commands."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Protocol
from uuid import UUID

from fakuicode.models import AgentMode, TokenUsage
from fakuicode.skills.install import (
    SkillInstallError,
    SkillInstallPreset,
    SkillInstallRequest,
    SkillInstallScope,
    parse_install_source,
)


class CommandError(ValueError):
    """Raised when slash-command syntax is not supported."""


class CommandRegistrationError(RuntimeError):
    """Raised when the built-in command catalog is internally inconsistent."""


class CommandKind(str, Enum):
    """Describe the primary execution path of a command."""

    LOCAL = "local"
    UI = "ui"
    PROMPT = "prompt"


class CommandHost(Protocol):
    """Renderer-neutral operations available to built-in command handlers."""

    def show_message(self, content: str) -> None: ...

    def send_user_message(self, content: str) -> None: ...

    def set_agent_mode(self, mode: AgentMode) -> bool: ...

    def get_agent_mode(self) -> AgentMode: ...

    def has_saved_plan(self) -> bool: ...

    def execute_saved_plan(self) -> None: ...

    def get_token_usage(self) -> TokenUsage | None: ...

    def refresh_status(self) -> None: ...

    def start_new_conversation(self) -> None: ...

    def clear_context(self) -> None: ...

    def compact_context(self) -> None: ...

    def show_sessions(self) -> None: ...

    def open_resume_picker(self) -> None: ...

    def delete_conversation(self, argument: str | None) -> None: ...

    def retry_last_prompt(self) -> None: ...

    def show_runtime_status(self) -> None: ...

    def show_mcp_status(self) -> None: ...

    def open_model_picker(self, argument: str | None) -> None: ...

    def handle_memory(self, argument: str | None) -> None: ...

    def open_permissions(self) -> None: ...

    def invoke_skill(self, name: str, arguments: str | None, original: str) -> None: ...

    def handle_skills(self, request: SkillInstallRequest | None) -> None: ...


@dataclass(frozen=True)
class CommandOption:
    """A fixed first argument that can be completed locally."""

    value: str
    description: str


CommandValidator = Callable[[str | None], str | None]
CommandHandler = Callable[[CommandHost, "CommandInvocation", "CommandRegistry"], None]


@dataclass(frozen=True)
class CommandSpec:
    """Metadata and behavior for one slash command."""

    name: str
    takes_argument: bool
    description: str
    options: tuple[CommandOption, ...] = ()
    aliases: tuple[str, ...] = ()
    usage: str = ""
    kind: CommandKind = CommandKind.LOCAL
    argument_hint: str | None = None
    hidden: bool = False
    handler: CommandHandler | None = None
    validator: CommandValidator | None = None


@dataclass(frozen=True)
class CommandInvocation:
    """One resolved command invocation with normalized arguments."""

    command: CommandSpec
    invoked_name: str
    arguments: str | None


@dataclass(frozen=True)
class CommandSuggestion:
    """A safe, display-ready command completion result."""

    command: CommandSpec
    completion: str
    description: str


_COMMAND_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


class CommandRegistry:
    """Immutable validated catalog used for command parsing and dispatch."""

    def __init__(self, commands: tuple[CommandSpec, ...]) -> None:
        self._commands = tuple(commands)
        self._by_name: dict[str, CommandSpec] = {}
        self._validate_and_index()

    @property
    def commands(self) -> tuple[CommandSpec, ...]:
        return self._commands

    def _validate_and_index(self) -> None:
        for command in self._commands:
            if not _COMMAND_NAME.fullmatch(command.name):
                raise CommandRegistrationError(f"Invalid command name: {command.name!r}")
            if command.handler is None:
                raise CommandRegistrationError(f"Command '/{command.name}' has no handler.")
            for candidate in (command.name, *command.aliases):
                if not _COMMAND_NAME.fullmatch(candidate):
                    raise CommandRegistrationError(f"Invalid command name or alias: {candidate!r}")
                key = candidate.casefold()
                previous = self._by_name.get(key)
                if previous is not None:
                    raise CommandRegistrationError(
                        f"Command name conflict for {candidate!r}: '/{previous.name}' and '/{command.name}'."
                    )
                self._by_name[key] = command

    def find(self, name: str) -> CommandSpec | None:
        return self._by_name.get(name.casefold())

    def parse(self, text: str) -> CommandInvocation | None:
        stripped = text.strip()
        if not stripped or not stripped.startswith("/"):
            return None
        command_text, separator, argument_text = stripped[1:].partition(" ")
        command = self.find(command_text)
        if command is None:
            rendered = f"/{command_text}" if command_text else "/"
            raise CommandError(f"Unknown command '{rendered}'. Use /help to see available commands.")
        arguments = argument_text.strip() if separator and argument_text.strip() else None
        if command.validator is not None:
            arguments = command.validator(arguments)
        return CommandInvocation(command, command_text, arguments)

    def dispatch(self, text: str, host: CommandHost) -> bool:
        """Dispatch slash input, returning whether the input was a command."""
        try:
            invocation = self.parse(text)
        except CommandError as error:
            host.show_message(str(error))
            return text.strip().startswith("/")
        if invocation is None:
            return False
        assert invocation.command.handler is not None
        try:
            invocation.command.handler(host, invocation, self)
        except CommandError as error:
            host.show_message(str(error))
        return True

    def suggest(self, text: str) -> tuple[CommandSuggestion, ...]:
        """Return visible command or fixed-option completions for slash input."""
        if not text.startswith("/") or "\n" in text or "\r" in text:
            return ()
        command_text, separator, argument_text = text[1:].partition(" ")
        command_prefix = command_text.casefold()
        if not separator:
            suggestions: list[CommandSuggestion] = []
            for command in self._commands:
                if command.hidden:
                    continue
                names = (command.name, *command.aliases)
                if any(name.casefold() == command_prefix for name in names):
                    continue
                if any(name.casefold().startswith(command_prefix) for name in names):
                    suggestions.append(
                        CommandSuggestion(
                            command,
                            f"/{command.name}" + (" " if command.takes_argument else ""),
                            command.description,
                        )
                    )
            return tuple(suggestions)

        command = self.find(command_text)
        if command is None or command.hidden or not command.options or any(
            character.isspace() for character in argument_text
        ):
            return ()
        argument_prefix = argument_text.casefold()
        return tuple(
            CommandSuggestion(command, f"/{command.name} {option.value}", option.description)
            for option in command.options
            if option.value.casefold().startswith(argument_prefix)
            and option.value.casefold() != argument_prefix
        )

    def should_show_empty_completion(self, text: str) -> bool:
        if not text.startswith("/") or "\n" in text or "\r" in text or self.suggest(text):
            return False
        command_text, separator, argument_text = text[1:].partition(" ")
        command = self.find(command_text)
        if not separator:
            return bool(command_text) and command is None
        if command is None or command.hidden or not command.options or any(
            character.isspace() for character in argument_text
        ):
            return False
        return bool(argument_text) and not any(
            option.value.casefold() == argument_text.casefold() for option in command.options
        )

    def format_help(self) -> str:
        lines: list[str] = []
        for command in self._commands:
            if command.hidden:
                continue
            aliases = ""
            if command.aliases:
                aliases = " (alias: " + ", ".join(f"/{alias}" for alias in command.aliases) + ")"
            lines.append(f"{command.usage or f'/{command.name}'}{aliases} — {command.description}")
        return "\n".join(lines)


def _no_arguments(arguments: str | None) -> None:
    if arguments is not None:
        raise CommandError("This command does not accept arguments. Use /help for usage.")
    return None


def _optional_single_argument(arguments: str | None) -> str | None:
    if arguments is not None and any(character.isspace() for character in arguments):
        raise CommandError("This command accepts at most one argument. Use /help for usage.")
    return arguments


def _memory_arguments(arguments: str | None) -> str | None:
    if arguments is None:
        return None
    parts = arguments.split()
    if len(parts) == 1 and parts[0].casefold() in {"on", "off", "forget"}:
        return parts[0].casefold()
    if len(parts) == 2 and parts[0].casefold() == "forget":
        try:
            entry_id = UUID(parts[1])
        except ValueError as error:
            raise CommandError("Memory entry id must be a UUID.") from error
        if str(entry_id) == parts[1].casefold():
            return f"forget {entry_id}"
    raise CommandError("Use /memory [on|off|forget [id]].")


def _skills_arguments(arguments: str | None) -> str | None:
    parse_skill_install_arguments(arguments)
    return arguments


def parse_skill_install_arguments(arguments: str | None) -> SkillInstallRequest | None:
    if arguments is None or arguments.strip().casefold() == "list":
        return None
    try:
        parts = shlex.split(arguments, posix=True)
    except ValueError as error:
        raise CommandError("Use /skills install <url> [--skill <name>] [--global] [--preset <name>] [--replace].") from error
    if len(parts) < 2 or parts[0].casefold() != "install":
        raise CommandError("Use /skills install <url> [--skill <name>] [--global] [--preset <name>] [--replace].")
    source = parts[1]
    skill: str | None = None
    scope = SkillInstallScope.PROJECT
    preset: SkillInstallPreset | None = None
    replace = False
    seen: set[str] = set()
    index = 2
    while index < len(parts):
        option = parts[index]
        if option in seen:
            raise CommandError(f"Duplicate /skills option: {option}")
        seen.add(option)
        if option == "--global":
            scope = SkillInstallScope.USER
            index += 1
            continue
        if option == "--replace":
            replace = True
            index += 1
            continue
        if option not in {"--skill", "--preset"} or index + 1 >= len(parts):
            raise CommandError("Unsupported or incomplete /skills install option.")
        value = parts[index + 1]
        if option == "--skill":
            skill = value
        else:
            try:
                preset = SkillInstallPreset(value)
            except ValueError as error:
                raise CommandError("Skill preset must be instruction, read-only, or coding.") from error
        index += 2
    try:
        parse_install_source(source, skill=skill)
    except SkillInstallError as error:
        raise CommandError(str(error)) from error
    return SkillInstallRequest(source, skill, scope, preset, replace)


def _help(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation
    host.show_message(registry.format_help())


def _new(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.start_new_conversation()


def _clear(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.clear_context()


def _compact(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.compact_context()


def _plan(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    if host.set_agent_mode("plan"):
        host.show_message("Plan mode enabled. Send a normal task to inspect and draft a read-only plan.")


def _do(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    if host.has_saved_plan():
        host.execute_saved_plan()
        return
    if host.get_agent_mode() == "plan":
        if host.set_agent_mode("execute"):
            host.show_message("Default execution mode enabled. No saved plan was executed.")
        return
    host.show_message("Default execution mode is already active and no saved plan is available.")


def _sessions(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.show_sessions()


def _resume(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.open_resume_picker()


def _delete(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del registry
    host.delete_conversation(invocation.arguments)


def _retry(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.retry_last_prompt()


def _status(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.show_runtime_status()


def _mcp(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.show_mcp_status()


def _model(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del registry
    host.open_model_picker(invocation.arguments)


def _memory(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del registry
    host.handle_memory(invocation.arguments)


def _permissions(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del invocation, registry
    host.open_permissions()


def _skills(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del registry
    host.handle_skills(parse_skill_install_arguments(invocation.arguments))


CORE_COMMAND_SPECS = (
    CommandSpec("help", False, "Show available commands", usage="/help", handler=_help, validator=_no_arguments),
    CommandSpec("new", False, "Start a new conversation", usage="/new", kind=CommandKind.UI, handler=_new, validator=_no_arguments),
    CommandSpec("clear", False, "Clear model context", usage="/clear", handler=_clear, validator=_no_arguments),
    CommandSpec("compact", False, "Compact older model context", usage="/compact", handler=_compact, validator=_no_arguments),
    CommandSpec("plan", False, "Plan the next task with read-only tools", usage="/plan", kind=CommandKind.UI, handler=_plan, validator=_no_arguments),
    CommandSpec("do", False, "Execute the saved plan or leave Plan mode", usage="/do", kind=CommandKind.UI, handler=_do, validator=_no_arguments),
    CommandSpec("sessions", False, "List saved conversations", aliases=("session",), usage="/sessions", handler=_sessions, validator=_no_arguments),
    CommandSpec("resume", False, "Choose a saved conversation to resume", usage="/resume", kind=CommandKind.UI, handler=_resume, validator=_no_arguments),
    CommandSpec("delete", True, "Delete a conversation", usage="/delete [id]", kind=CommandKind.UI, argument_hint="id", handler=_delete, validator=_optional_single_argument),
    CommandSpec("retry", False, "Retry the previous prompt", usage="/retry", kind=CommandKind.PROMPT, handler=_retry, validator=_no_arguments),
    CommandSpec("status", False, "Show current status", usage="/status", handler=_status, validator=_no_arguments),
    CommandSpec("mcp", False, "Show MCP server and tool status", usage="/mcp", handler=_mcp, validator=_no_arguments),
    CommandSpec("model", True, "Choose a model profile", usage="/model", kind=CommandKind.UI, argument_hint="profile", handler=_model, validator=_optional_single_argument),
    CommandSpec(
        "memory",
        True,
        "Manage automatic memory",
        (
            CommandOption("on", "Enable automatic memory"),
            CommandOption("off", "Disable automatic memory"),
            CommandOption("forget", "Choose a memory entry to forget"),
        ),
        usage="/memory [on|off|forget [id]]",
        kind=CommandKind.UI,
        argument_hint="on|off|forget [id]",
        handler=_memory,
        validator=_memory_arguments,
    ),
    CommandSpec("permissions", False, "Manage permission mode and project trust", aliases=("permission",), usage="/permissions", kind=CommandKind.UI, handler=_permissions, validator=_no_arguments),
    CommandSpec(
        "skills",
        True,
        "List or install reusable Skills",
        (CommandOption("list", "List effective Skills"), CommandOption("install", "Install a public Skill")),
        usage="/skills install <url> [--skill <name>] [--global] [--preset <name>] [--replace]",
        kind=CommandKind.UI,
        argument_hint="list|install <url>",
        handler=_skills,
        validator=_skills_arguments,
    ),
)


COMMAND_SPECS = CORE_COMMAND_SPECS
DEFAULT_COMMAND_REGISTRY = CommandRegistry(CORE_COMMAND_SPECS)
RESERVED_COMMAND_NAMES = frozenset(
    candidate.casefold()
    for command in CORE_COMMAND_SPECS
    for candidate in (command.name, *command.aliases)
)


def compose_command_registry(skills: Iterable[tuple[str, str]]) -> CommandRegistry:
    """Build one App-scoped command catalog from core commands and current Skills."""

    skill_specs = tuple(
        CommandSpec(
            name,
            True,
            description,
            usage=f"/{name} [arguments]",
            kind=CommandKind.PROMPT,
            argument_hint="arguments",
            handler=_invoke_skill,
        )
        for name, description in sorted(skills)
    )
    return CommandRegistry((*CORE_COMMAND_SPECS, *skill_specs))


def _invoke_skill(host: CommandHost, invocation: CommandInvocation, registry: CommandRegistry) -> None:
    del registry
    original = f"/{invocation.invoked_name}"
    if invocation.arguments:
        original += f" {invocation.arguments}"
    host.invoke_skill(invocation.command.name, invocation.arguments, original)


def suggest_commands(text: str) -> tuple[CommandSuggestion, ...]:
    """Compatibility wrapper over the built-in command registry."""
    return DEFAULT_COMMAND_REGISTRY.suggest(text)


def should_show_empty_completion(text: str) -> bool:
    """Compatibility wrapper over the built-in command registry."""
    return DEFAULT_COMMAND_REGISTRY.should_show_empty_completion(text)


def format_command_help() -> str:
    """Compatibility wrapper over the built-in command registry."""
    return DEFAULT_COMMAND_REGISTRY.format_help()


def parse_command(text: str) -> tuple[str, str | None]:
    """Return a normalized built-in command name and optional argument."""
    invocation = DEFAULT_COMMAND_REGISTRY.parse(text)
    if invocation is None:
        raise CommandError("Commands must start with '/'.")
    return invocation.command.name, invocation.arguments
