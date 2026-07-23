from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("text", "name", "argument"),
    [
        ("/help", "help", None),
        ("/new", "new", None),
        ("/sessions", "sessions", None),
        ("/resume", "resume", None),
        ("/model", "model", None),
        ("/model fast", "model", "fast"),
        ("/plan", "plan", None),
        ("/do", "do", None),
        ("/compact", "compact", None),
        ("/permissions", "permissions", None),
        ("/memory", "memory", None),
        ("/memory on", "memory", "on"),
        ("/memory off", "memory", "off"),
        ("/memory forget", "memory", "forget"),
        ("/memory forget 3f67a8d1-3853-4e09-989a-934cbf641629", "memory", "forget 3f67a8d1-3853-4e09-989a-934cbf641629"),
        ("/skills", "skills", None),
        ("/skills list", "skills", "list"),
        (
            "/skills install https://www.skills.sh/anthropics/skills/frontend-design --preset coding",
            "skills",
            "install https://www.skills.sh/anthropics/skills/frontend-design --preset coding",
        ),
        ("/delete", "delete", None),
    ],
)
def test_parse_command_returns_a_safe_structured_intent(text: str, name: str, argument: str | None) -> None:
    from fakuicode.commands import parse_command

    assert parse_command(text) == (name, argument)


@pytest.mark.parametrize(
    "text",
    [
        "/unknown", "/resume 1234", "/sessions extra", "/permissions trusted", "/compact anything",
        "/memory maybe", "/memory forget ../note.md", "/memory forget not-a-uuid", "/memory off extra",
    ],
)
def test_parse_command_rejects_unknown_or_invalid_syntax(text: str) -> None:
    from fakuicode.commands import CommandError, parse_command

    with pytest.raises(CommandError):
        parse_command(text)


def test_command_suggestions_match_a_slash_prefix_and_preserve_catalog_order() -> None:
    from fakuicode.commands import suggest_commands

    assert [item.completion for item in suggest_commands("/m")] == ["/mcp", "/model ", "/memory "]
    assert [item.command.name for item in suggest_commands("/s")] == ["sessions", "status", "skills"]
    assert suggest_commands("message /s") == ()
    assert suggest_commands("/model careful") == ()


def test_command_suggestions_offer_all_local_commands_for_a_bare_slash() -> None:
    from fakuicode.commands import suggest_commands

    names = [item.command.name for item in suggest_commands("/")]

    assert names == [
        "help", "new", "clear", "compact", "plan", "do", "sessions", "resume", "delete", "retry", "status",
        "mcp", "model", "memory", "permissions", "skills",
    ]


def test_memory_command_suggestions_continue_after_the_command_name() -> None:
    from fakuicode.commands import suggest_commands

    assert [item.completion for item in suggest_commands("/memory ")] == [
        "/memory on",
        "/memory off",
        "/memory forget",
    ]
    assert [item.completion for item in suggest_commands("/memory o")] == [
        "/memory on",
        "/memory off",
    ]
    assert suggest_commands("/memory on") == ()
    assert suggest_commands("/memory forget ") == ()


def test_command_help_does_not_advertise_a_resume_id() -> None:
    from fakuicode.commands import format_command_help

    assert "/resume [id]" not in format_command_help()
    assert "/resume <id>" not in format_command_help()
    assert "/delete [id]" in format_command_help()
    assert "/compact" in format_command_help()
    assert "/memory [on|off|forget [id]]" in format_command_help()
    assert "/skills install <url>" in format_command_help()


def test_skill_install_command_arguments_are_strictly_parsed() -> None:
    from fakuicode.commands import parse_skill_install_arguments

    request = parse_skill_install_arguments(
        "install https://github.com/acme/skills --skill demo --global --preset read-only --replace"
    )

    assert request is not None
    assert request.source == "https://github.com/acme/skills"
    assert request.skill == "demo"
    assert request.scope.value == "user"
    assert request.preset is not None and request.preset.value == "read-only"
    assert request.replace is True


@pytest.mark.parametrize(
    "arguments",
    [
        "install",
        "install https://github.com/acme/skills --unknown",
        "install https://github.com/acme/skills --preset full",
        "install https://github.com/acme/skills --skill",
        "remove demo",
    ],
)
def test_skill_install_command_rejects_ambiguous_or_unsupported_syntax(arguments: str) -> None:
    from fakuicode.commands import CommandError, parse_skill_install_arguments

    with pytest.raises(CommandError):
        parse_skill_install_arguments(arguments)


def _registry_spec(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    hidden: bool = False,
):
    from fakuicode.commands import CommandKind, CommandSpec

    def handler(host, invocation, registry) -> None:
        del host, invocation, registry

    return CommandSpec(
        name,
        False,
        f"Describe {name}",
        aliases=aliases,
        usage=f"/{name}",
        kind=CommandKind.LOCAL,
        hidden=hidden,
        handler=handler,
    )


def test_registry_resolves_names_and_aliases_case_insensitively() -> None:
    from fakuicode.commands import CommandRegistry

    registry = CommandRegistry((_registry_spec("sessions", aliases=("session",)),))

    invocation = registry.parse("/SeSsIoN   keep  inner   spacing")

    assert invocation is not None
    assert invocation.command.name == "sessions"
    assert invocation.invoked_name == "SeSsIoN"
    assert invocation.arguments == "keep  inner   spacing"


@pytest.mark.parametrize(
    "specs",
    [
        (_registry_spec("status"), _registry_spec("STATUS")),
        (_registry_spec("status"), _registry_spec("state", aliases=("STATUS",))),
        (_registry_spec("first", aliases=("state",)), _registry_spec("second", aliases=("STATE",))),
    ],
)
def test_registry_rejects_case_insensitive_name_and_alias_conflicts(specs) -> None:
    from fakuicode.commands import CommandRegistrationError, CommandRegistry

    with pytest.raises(CommandRegistrationError, match="conflict"):
        CommandRegistry(specs)


@pytest.mark.parametrize("name", ["", "/status", "two words", "bad_alias!"])
def test_registry_rejects_invalid_command_names_and_aliases(name: str) -> None:
    from fakuicode.commands import CommandRegistrationError, CommandRegistry

    spec = _registry_spec("valid", aliases=(name,)) if name else _registry_spec(name)
    with pytest.raises(CommandRegistrationError, match="Invalid command"):
        CommandRegistry((spec,))


def test_hidden_commands_can_be_parsed_but_are_not_helped_or_completed() -> None:
    from fakuicode.commands import CommandRegistry

    registry = CommandRegistry(
        (
            _registry_spec("visible"),
            _registry_spec("hidden", aliases=("secret",), hidden=True),
        )
    )

    assert registry.parse("/hidden") is not None
    assert registry.suggest("/h") == ()
    assert "/hidden" not in registry.format_help()
    assert "/secret" not in registry.format_help()


def test_default_registry_keeps_plural_commands_and_dynamic_registry_adds_skills() -> None:
    from fakuicode.commands import DEFAULT_COMMAND_REGISTRY, compose_command_registry

    session = DEFAULT_COMMAND_REGISTRY.parse("/session")
    permission = DEFAULT_COMMAND_REGISTRY.parse("/PERMISSION")
    review = compose_command_registry((("review", "Review changes"),)).parse("/review")

    assert session is not None and session.command.name == "sessions"
    assert permission is not None and permission.command.name == "permissions"
    assert review is not None and review.command.name == "review"
    assert [item.completion for item in DEFAULT_COMMAND_REGISTRY.suggest("/sessio")] == ["/sessions"]


def test_registry_returns_early_for_non_commands_and_guides_unknown_commands_to_help() -> None:
    from fakuicode.commands import CommandError, DEFAULT_COMMAND_REGISTRY

    assert DEFAULT_COMMAND_REGISTRY.parse("") is None
    assert DEFAULT_COMMAND_REGISTRY.parse("ordinary prompt") is None
    with pytest.raises(CommandError, match="/help"):
        DEFAULT_COMMAND_REGISTRY.parse("/missing")
