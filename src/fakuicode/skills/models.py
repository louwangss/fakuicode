"""Immutable contracts for reusable fakuiCode Skill packages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping


class SkillSource(StrEnum):
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


class SkillInvocation(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class SkillExecution(StrEnum):
    SHARED = "shared"
    ISOLATED = "isolated"


@dataclass(frozen=True)
class SkillInstallReceipt:
    schema_version: int
    requested_url: str
    source_url: str
    revision: str
    skill_path: str
    upstream_fingerprint: str


@dataclass(frozen=True)
class SkillToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, object]
    entrypoint: Path


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    source: SkillSource
    package_path: Path
    body: str
    invocation: SkillInvocation
    visible_tools: tuple[str, ...]
    execution: SkillExecution
    history_turns: int
    profile: str
    tools: tuple[SkillToolSpec, ...]
    fingerprint: str
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, str] | None = None
    allowed_tools: str | None = None
    install_receipt: SkillInstallReceipt | None = None
    author_warnings: tuple[str, ...] = ()

    @property
    def runtime_tool_names(self) -> tuple[str, ...]:
        return tuple(f"skill__{self.name}__{tool.name}" for tool in self.tools)

    def render(self, arguments: str | None) -> str:
        value = arguments or ""
        if "$ARGUMENTS" in self.body:
            return self.body.replace("$ARGUMENTS", value)
        if value:
            return f"{self.body.rstrip()}\n\nARGUMENTS:\n{value}"
        return self.body


@dataclass(frozen=True)
class SkillDiagnostic:
    code: str
    name: str
    source: SkillSource
    message: str


@dataclass(frozen=True)
class SkillSnapshot:
    skills: Mapping[str, SkillDefinition]
    diagnostics: tuple[SkillDiagnostic, ...]

    @property
    def command_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.skills))


@dataclass(frozen=True)
class SkillCatalog:
    text: str
    estimated_tokens: int
    omitted_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActiveSkill:
    name: str
    rendered_body: str
    source: SkillSource
    arguments: str
    fingerprint: str
    visible_tools: tuple[str, ...]
    runtime_tool_names: tuple[str, ...]
    stale: bool = False
    package_path: Path | None = None
