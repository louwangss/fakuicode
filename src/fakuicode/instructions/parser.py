"""Pure Markdown parsing for project-instruction documents."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from fakuicode.instructions.models import InstructionDiagnosticCode


@dataclass(frozen=True)
class TextLine:
    """One literal source line retained for later rendering."""

    text: str
    line: int


@dataclass(frozen=True)
class IncludeLine:
    """One syntactically candidate include declaration."""

    declared_path: str
    line: int


InstructionDocumentNode = TextLine | IncludeLine


@dataclass(frozen=True)
class ParseDiagnostic:
    """A parser-local issue that the loader later associates with a source."""

    code: InstructionDiagnosticCode
    line: int


@dataclass(frozen=True)
class InstructionDocument:
    """Normalized instruction source with stable line numbers."""

    nodes: tuple[InstructionDocumentNode, ...]
    diagnostics: tuple[ParseDiagnostic, ...] = ()


@dataclass(frozen=True)
class _Fence:
    character: str
    length: int


def parse_instruction_document(content: str) -> InstructionDocument:
    """Parse fences, escaped includes, and basic include declarations without I/O."""

    normalized = content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    nodes: list[InstructionDocumentNode] = []
    diagnostics: list[ParseDiagnostic] = []
    fence: _Fence | None = None

    for line_number, line in enumerate(normalized.split("\n"), start=1):
        if fence is not None:
            nodes.append(TextLine(line, line_number))
            if _is_fence_close(line, fence):
                fence = None
            continue

        opened_fence = _fence_opening(line)
        if opened_fence is not None:
            nodes.append(TextLine(line, line_number))
            fence = opened_fence
        elif line.startswith("\\@include"):
            nodes.append(TextLine(line[1:], line_number))
        elif line.startswith("@include "):
            declared_path = line[len("@include ") :]
            diagnostic = _validate_include_path(declared_path)
            if diagnostic is None:
                nodes.append(IncludeLine(declared_path, line_number))
            else:
                diagnostics.append(ParseDiagnostic(diagnostic, line_number))
        else:
            nodes.append(TextLine(line, line_number))

    return InstructionDocument(tuple(nodes), tuple(diagnostics))


def _fence_opening(line: str) -> _Fence | None:
    prefix_length = len(line) - len(line.lstrip(" "))
    if prefix_length > 3 or prefix_length == len(line):
        return None

    marker = line[prefix_length]
    if marker not in {"`", "~"}:
        return None
    marker_length = _marker_length(line, prefix_length, marker)
    if marker_length < 3:
        return None

    info = line[prefix_length + marker_length :]
    if marker == "`" and "`" in info:
        return None
    return _Fence(marker, marker_length)


def _is_fence_close(line: str, fence: _Fence) -> bool:
    prefix_length = len(line) - len(line.lstrip(" "))
    if prefix_length > 3 or prefix_length == len(line) or line[prefix_length] != fence.character:
        return False

    marker_length = _marker_length(line, prefix_length, fence.character)
    return marker_length >= fence.length and line[prefix_length + marker_length :].strip(" \t") == ""


def _marker_length(line: str, start: int, marker: str) -> int:
    end = start
    while end < len(line) and line[end] == marker:
        end += 1
    return end - start


def _validate_include_path(path: str) -> InstructionDiagnosticCode | None:
    if not path or any(character.isspace() for character in path):
        return InstructionDiagnosticCode.INVALID_INCLUDE
    if (
        path.startswith(("/", "~"))
        or "\\" in path
        or any(character in path for character in ('"', "'", ":", "*", "?", "[", "]", "$", "%"))
        or any(_is_disallowed_control(character) for character in path)
    ):
        return InstructionDiagnosticCode.INVALID_INCLUDE
    if not path.casefold().endswith(".md"):
        return InstructionDiagnosticCode.NOT_MARKDOWN
    return None


def _is_disallowed_control(character: str) -> bool:
    return unicodedata.category(character) in {"Cc", "Cf"}
