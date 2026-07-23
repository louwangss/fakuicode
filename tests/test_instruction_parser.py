"""Tests for deterministic Markdown instruction parsing."""

from __future__ import annotations

from fakuicode.instructions import InstructionDiagnosticCode
from fakuicode.instructions.parser import (
    IncludeLine,
    ParseDiagnostic,
    TextLine,
    parse_instruction_document,
)


def test_parser_keeps_include_literal_inside_fences() -> None:
    document = parse_instruction_document(
        "@include before.md\n"
        "```python\n"
        "@include literal-backtick.md\n"
        "````\n"
        "~~~~ note\n"
        "@include literal-tilde.md\n"
        "~~~~\n"
        "@include after.md"
    )

    assert document.nodes == (
        IncludeLine("before.md", 1),
        TextLine("```python", 2),
        TextLine("@include literal-backtick.md", 3),
        TextLine("````", 4),
        TextLine("~~~~ note", 5),
        TextLine("@include literal-tilde.md", 6),
        TextLine("~~~~", 7),
        IncludeLine("after.md", 8),
    )


def test_parser_only_closes_a_fence_with_matching_marker_and_allowed_trailing_whitespace() -> None:
    document = parse_instruction_document(
        "   ``` info\n"
        "@include remains-literal.md\n"
        "  `` extra\n"
        "  ```` \t\n"
        "@include expanded.md"
    )

    assert document.nodes == (
        TextLine("   ``` info", 1),
        TextLine("@include remains-literal.md", 2),
        TextLine("  `` extra", 3),
        TextLine("  ```` \t", 4),
        IncludeLine("expanded.md", 5),
    )


def test_parser_preserves_unclosed_fences_and_unescapes_only_outside_them() -> None:
    document = parse_instruction_document(
        "\\@include literal.md\n"
        "~~~\n"
        "\\@include fenced.md\n"
        "@include also-fenced.md"
    )

    assert document.nodes == (
        TextLine("@include literal.md", 1),
        TextLine("~~~", 2),
        TextLine("\\@include fenced.md", 3),
        TextLine("@include also-fenced.md", 4),
    )


def test_parser_normalizes_bom_and_crlf_without_losing_source_line_numbers() -> None:
    document = parse_instruction_document("\ufefffirst\r\n@include second.md\r\nlast")

    assert document.nodes == (
        TextLine("first", 1),
        IncludeLine("second.md", 2),
        TextLine("last", 3),
    )


def test_parser_accepts_only_a_strict_standalone_relative_markdown_include() -> None:
    document = parse_instruction_document("@include rules/base.MD")

    assert document.nodes == (IncludeLine("rules/base.MD", 1),)
    assert document.diagnostics == ()


def test_parser_keeps_non_candidate_include_like_lines_as_literal_text() -> None:
    document = parse_instruction_document(" @include rules.md\n@include\t rules.md")

    assert document.nodes == (
        TextLine(" @include rules.md", 1),
        TextLine("@include\t rules.md", 2),
    )
    assert document.diagnostics == ()


def test_parser_rejects_invalid_include_syntax_without_injecting_its_source_line() -> None:
    document = parse_instruction_document(
        "@include  rules.md\n"
        "@include rules.md \n"
        "@include 'rules.md'\n"
        "@include rules/*.md"
    )

    assert document.nodes == ()
    assert document.diagnostics == (
        ParseDiagnostic(InstructionDiagnosticCode.INVALID_INCLUDE, 1),
        ParseDiagnostic(InstructionDiagnosticCode.INVALID_INCLUDE, 2),
        ParseDiagnostic(InstructionDiagnosticCode.INVALID_INCLUDE, 3),
        ParseDiagnostic(InstructionDiagnosticCode.INVALID_INCLUDE, 4),
    )


def test_parser_uses_not_markdown_for_otherwise_valid_paths_with_wrong_suffix() -> None:
    document = parse_instruction_document("@include rules/base.txt")

    assert document.nodes == ()
    assert document.diagnostics == (ParseDiagnostic(InstructionDiagnosticCode.NOT_MARKDOWN, 1),)


def test_parser_rejects_paths_that_could_escape_or_expand_before_loader_validation() -> None:
    document = parse_instruction_document(
        "@include /absolute.md\n"
        "@include C:/drive.md\n"
        "@include //server/share.md\n"
        "@include dir\\file.md\n"
        "@include ~/home.md\n"
        "@include $HOME/file.md\n"
        "@include %USERPROFILE%/file.md\n"
        "@include unsafe\u202efile.md"
    )

    assert document.nodes == ()
    assert document.diagnostics == tuple(
        ParseDiagnostic(InstructionDiagnosticCode.INVALID_INCLUDE, line) for line in range(1, 9)
    )
