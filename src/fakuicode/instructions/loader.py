"""Bounded discovery and reading of the three project-instruction sources."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

from fakuicode.errors import ToolPolicyError
from fakuicode.instructions.models import (
    DEFAULT_INSTRUCTION_LIMITS,
    InstructionDiagnostic,
    InstructionDiagnosticCode,
    InstructionLimits,
    InstructionScope,
    InstructionSnapshot,
)
from fakuicode.instructions.parser import IncludeLine, InstructionDocument, TextLine, parse_instruction_document
from fakuicode.instructions.render import render_instruction_layers
from fakuicode.tools.policy import WorkspacePolicy


class InstructionSnapshotLoader(Protocol):
    """Produces the immutable instruction snapshot for a session boundary."""

    def load(self) -> InstructionSnapshot: ...


@dataclass(frozen=True)
class _LayerSpec:
    scope: InstructionScope
    root: Path
    path: Path


@dataclass(frozen=True)
class _SourceDocument:
    scope: InstructionScope
    root: Path
    path: Path
    source: str
    document: InstructionDocument


@dataclass
class _ExpandedDocument:
    source: _SourceDocument
    children: dict[int, "_ExpandedDocument"]


class InstructionLoader:
    """Load fixed instruction main files without discovering other directories."""

    def __init__(
        self,
        workspace: Path,
        *,
        user_home: Path | None = None,
        limits: InstructionLimits = DEFAULT_INSTRUCTION_LIMITS,
    ) -> None:
        self.workspace = workspace.resolve()
        self.user_home = (user_home if user_home is not None else Path.home()).resolve()
        self.limits = limits

    def load(self) -> InstructionSnapshot:
        """Read the fixed sources while isolating each source failure."""

        try:
            return self._load()
        except Exception:
            return InstructionSnapshot.failed()

    def _load(self) -> InstructionSnapshot:
        """Load one snapshot, preserving expected file-level failures as diagnostics."""

        layers = self._layer_specs()
        self._diagnostics: list[InstructionDiagnostic] = []
        self._target_keys = {self._lexical_key(layer.path) for layer in layers}
        self._source_cache: dict[Path, InstructionDocument] = {}
        expanded_layers: dict[InstructionScope, _ExpandedDocument] = {}

        for layer in reversed(layers):
            source = self._read_main_file(layer)
            if source is not None:
                expanded_layers[layer.scope] = self._expand(source, depth=0, stack=(source.path,))

        render_result = render_instruction_layers(
            (expanded_layers[layer.scope] for layer in layers if layer.scope in expanded_layers),
            max_payload_bytes=self.limits.max_payload_bytes,
        )
        self._diagnostics.extend(
            InstructionDiagnostic(diagnostic.code, diagnostic.scope, diagnostic.source, diagnostic.line)
            for diagnostic in render_result.diagnostics
        )

        return InstructionSnapshot(
            text=render_result.text,
            loaded_layers=render_result.rendered_scopes,
            processed_target_count=len(self._target_keys),
            diagnostics=tuple(sorted(self._diagnostics, key=self._diagnostic_sort_key)),
        )

    def _layer_specs(self) -> tuple[_LayerSpec, ...]:
        user_root = self.user_home / ".fakuicode"
        return (
            _LayerSpec(InstructionScope.USER, user_root, user_root / "AGENTS.md"),
            _LayerSpec(InstructionScope.PROJECT, self.workspace, self.workspace / "AGENTS.md"),
            _LayerSpec(
                InstructionScope.PROJECT_LOCAL,
                self.workspace,
                self.workspace / ".fakuicode" / "AGENTS.md",
            ),
        )

    def _read_main_file(self, layer: _LayerSpec) -> _SourceDocument | None:
        source = layer.path.relative_to(layer.root).as_posix()
        return self._read_source(
            layer,
            layer.path,
            source=source,
            line=None,
            missing_is_silent=True,
        )

    def _read_source(
        self,
        layer: _LayerSpec,
        requested_path: Path,
        *,
        source: str,
        line: int | None,
        missing_is_silent: bool,
    ) -> _SourceDocument | None:
        try:
            candidate = WorkspacePolicy(layer.root).resolve_path(str(requested_path))
        except ToolPolicyError:
            self._record(InstructionDiagnosticCode.PATH_REJECTED, layer, source, line)
            return None
        if not candidate.exists():
            if not missing_is_silent:
                self._record(InstructionDiagnosticCode.FILE_NOT_FOUND, layer, source, line)
            return None
        if not candidate.is_file():
            self._record(InstructionDiagnosticCode.NOT_REGULAR_FILE, layer, source, line)
            return None
        parsed_document = self._source_cache.get(candidate)
        if parsed_document is None:
            try:
                with candidate.open("rb") as handle:
                    raw = handle.read(self.limits.max_read_bytes)
            except FileNotFoundError:
                if not missing_is_silent:
                    self._record(InstructionDiagnosticCode.FILE_NOT_FOUND, layer, source, line)
                return None
            except PermissionError:
                self._record(InstructionDiagnosticCode.FILE_PERMISSION_DENIED, layer, source, line)
                return None
            except OSError:
                self._record(InstructionDiagnosticCode.FILE_READ_FAILED, layer, source, line)
                return None

            if len(raw) > self.limits.max_source_bytes:
                self._record(InstructionDiagnosticCode.FILE_TOO_LARGE, layer, source, line)
                return None
            try:
                content = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            except UnicodeDecodeError:
                self._record(InstructionDiagnosticCode.INVALID_UTF8, layer, source, line)
                return None
            if not content:
                return None
            parsed_document = parse_instruction_document(content)
            self._source_cache[candidate] = parsed_document

        declared_source = Path(os.path.abspath(requested_path)).relative_to(layer.root).as_posix()
        return _SourceDocument(
            scope=layer.scope,
            root=layer.root,
            path=candidate,
            source=declared_source,
            document=parsed_document,
        )

    def _expand(
        self,
        source: _SourceDocument,
        *,
        depth: int,
        stack: tuple[Path, ...],
    ) -> _ExpandedDocument:
        for diagnostic in source.document.diagnostics:
            self._record(diagnostic.code, self._layer_for(source), source.source, diagnostic.line)
        children: dict[int, _ExpandedDocument] = {}
        layer = self._layer_for(source)
        for node in source.document.nodes:
            if not isinstance(node, IncludeLine):
                continue
            if depth >= self.limits.max_include_depth:
                self._record(InstructionDiagnosticCode.DEPTH_LIMIT, layer, source.source, node.line)
                continue
            requested_path = source.path.parent / node.declared_path
            key = self._lexical_key(requested_path)
            if key not in self._target_keys:
                if len(self._target_keys) >= self.limits.max_file_targets:
                    self._record(InstructionDiagnosticCode.FILE_TARGET_LIMIT, layer, source.source, node.line)
                    continue
                self._target_keys.add(key)
            child = self._read_source(
                layer,
                requested_path,
                source=source.source,
                line=node.line,
                missing_is_silent=False,
            )
            if child is None:
                continue
            if child.path in stack:
                self._record(InstructionDiagnosticCode.INCLUDE_CYCLE, layer, source.source, node.line)
                continue
            children[node.line] = self._expand(child, depth=depth + 1, stack=(*stack, child.path))
        return _ExpandedDocument(source, children)

    @staticmethod
    def _lexical_key(path: Path) -> str:
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def _layer_for(source: _SourceDocument) -> _LayerSpec:
        return _LayerSpec(source.scope, source.root, source.path)

    def _record(
        self,
        code: InstructionDiagnosticCode,
        layer: _LayerSpec,
        source: str,
        line: int | None,
    ) -> None:
        self._diagnostics.append(self._diagnostic(code, layer, source, line))

    @staticmethod
    def _diagnostic_sort_key(diagnostic: InstructionDiagnostic) -> tuple[int, str, int, str]:
        scope_order = {
            InstructionScope.USER: 0,
            InstructionScope.PROJECT: 1,
            InstructionScope.PROJECT_LOCAL: 2,
        }
        return (
            scope_order[diagnostic.scope],
            diagnostic.source,
            -1 if diagnostic.line is None else diagnostic.line,
            diagnostic.code.value,
        )

    @staticmethod
    def _diagnostic(
        code: InstructionDiagnosticCode,
        layer: _LayerSpec,
        source: str,
        line: int | None = None,
    ) -> InstructionDiagnostic:
        return InstructionDiagnostic(code, layer.scope, source, line)
