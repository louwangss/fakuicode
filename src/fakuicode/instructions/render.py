"""Render parsed project instructions within a strict UTF-8 byte budget."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape

from fakuicode.instructions.models import (
    InstructionDiagnosticCode,
    InstructionScope,
    sanitize_instruction_metadata,
)
from fakuicode.instructions.parser import IncludeLine, TextLine


_TRUNCATION_MARKER = "[instruction truncated]"


@dataclass(frozen=True)
class RenderDiagnostic:
    code: InstructionDiagnosticCode
    scope: InstructionScope
    source: str
    line: int | None = None


@dataclass(frozen=True)
class RenderResult:
    text: str
    rendered_scopes: tuple[InstructionScope, ...]
    diagnostics: tuple[RenderDiagnostic, ...]
    dropped: bool = False


def render_instruction_layers(layers: Iterable[object], *, max_payload_bytes: int) -> RenderResult:
    """Select higher-priority root documents first while retaining display order."""

    layer_items = tuple(layers)
    selected: dict[InstructionScope, RenderResult] = {}
    diagnostics_by_scope: dict[InstructionScope, tuple[RenderDiagnostic, ...]] = {}
    remaining = max_payload_bytes
    for layer in reversed(layer_items):
        separator_bytes = 2 if selected else 0
        result = _render_document(layer, include=False, available_bytes=remaining - separator_bytes)
        if result.text:
            selected[layer.source.scope] = result
            diagnostics_by_scope[layer.source.scope] = result.diagnostics
            remaining -= _byte_count(result.text) + separator_bytes
        else:
            diagnostics_by_scope[layer.source.scope] = (
                *result.diagnostics,
                RenderDiagnostic(
                    InstructionDiagnosticCode.MAIN_TRUNCATED,
                    layer.source.scope,
                    layer.source.source,
                ),
            )

    ordered = [selected[layer.source.scope] for layer in layer_items if layer.source.scope in selected]
    return RenderResult(
        text="\n\n".join(result.text for result in ordered),
        rendered_scopes=tuple(result.rendered_scopes[0] for result in ordered),
        diagnostics=tuple(
            diagnostic
            for layer in layer_items
            for diagnostic in diagnostics_by_scope.get(layer.source.scope, ())
        ),
    )


def _render_document(expanded: object, *, include: bool, available_bytes: int) -> RenderResult:
    source = expanded.source
    prefix, suffix = _boundaries(source.scope, source.source, include=include)
    body_budget = available_bytes - _byte_count(prefix) - _byte_count(suffix)
    if body_budget < 0:
        return RenderResult("", (), (), dropped=True)

    parts: list[str] = []
    diagnostics: list[RenderDiagnostic] = []
    text_truncated = False
    for node in source.document.nodes:
        if isinstance(node, TextLine):
            if text_truncated:
                continue
            if _fits(parts, node.text, body_budget):
                parts.append(node.text)
                continue
            if include:
                return RenderResult("", (), (), dropped=True)
            text_truncated = True
            if _fits(parts, _TRUNCATION_MARKER, body_budget):
                parts.append(_TRUNCATION_MARKER)
            diagnostics.append(
                RenderDiagnostic(
                    InstructionDiagnosticCode.MAIN_TRUNCATED,
                    source.scope,
                    source.source,
                )
            )
            continue

        child = expanded.children.get(node.line)
        if child is None:
            continue
        remaining_body = body_budget - _byte_count("\n".join(parts)) - (1 if parts else 0)
        candidate = _render_document(child, include=True, available_bytes=remaining_body)
        if candidate.dropped:
            diagnostics.append(
                RenderDiagnostic(
                    InstructionDiagnosticCode.INCLUDE_BUDGET,
                    source.scope,
                    source.source,
                    node.line,
                )
            )
            continue
        parts.append(candidate.text)
        diagnostics.extend(candidate.diagnostics)

    body = "\n".join(parts)
    if include and not body:
        return RenderResult("", (), (), dropped=False)
    text = f"{prefix}{body}{suffix}"
    if _byte_count(text) > available_bytes:
        return RenderResult("", (), (), dropped=True)
    return RenderResult(text, (source.scope,), tuple(diagnostics))


def _boundaries(scope: InstructionScope, source: str, *, include: bool) -> tuple[str, str]:
    path = _safe_attribute(source)
    if include:
        return f'<included-instructions path="{path}">\n', "\n</included-instructions>"
    safe_scope = _safe_attribute(scope.value)
    return (
        f'<instruction-source scope="{safe_scope}" path="{path}">\n',
        "\n</instruction-source>",
    )


def _fits(parts: list[str], candidate: str, budget: int) -> bool:
    return _byte_count("\n".join((*parts, candidate))) <= budget


def _byte_count(text: str) -> int:
    return len(text.encode("utf-8"))


def _safe_attribute(value: str) -> str:
    cleaned = sanitize_instruction_metadata(value.replace("\\", "/"))
    return escape(cleaned, quote=True)
