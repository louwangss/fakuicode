"""Strict, layered loading for Markdown subagent definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import re

import yaml

from fakuicode.agent import MAX_ITERATIONS
from fakuicode.subagents.models import (
    AgentDefinition,
    AgentSource,
    CatalogDiagnostic,
    PermissionBehavior,
)


_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?\Z")
_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_FIELDS = {
    "name",
    "description",
    "tools",
    "disallowedTools",
    "profile",
    "maxTurns",
    "permissionMode",
    "background",
}


class CatalogLoadError(ValueError):
    """An embedded definition is invalid and therefore indicates a build error."""


class AgentCatalog:
    def __init__(
        self,
        definitions: Mapping[str, AgentDefinition],
        diagnostics: Iterable[CatalogDiagnostic] = (),
    ) -> None:
        self.definitions = dict(definitions)
        self.diagnostics = tuple(diagnostics)

    def resolve(self, name: str) -> AgentDefinition:
        try:
            return self.definitions[name]
        except KeyError as error:
            raise KeyError(f"未知 subagent_type: {name}") from error

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        user_root: Path | None = None,
        builtin_root: Path | None = None,
        plugin_roots: Iterable[Path] = (),
    ) -> AgentCatalog:
        definitions: dict[str, AgentDefinition] = {}
        diagnostics: list[CatalogDiagnostic] = []
        sources = (
            *((AgentSource.PLUGIN, root, False) for root in plugin_roots),
            (AgentSource.BUILTIN, builtin_root, True),
            (AgentSource.USER, user_root, False),
            (AgentSource.PROJECT, project_root, False),
        )
        for source, root, required in sources:
            if root is None:
                continue
            loaded, invalid_names, source_diagnostics = _load_source(root, source, required=required)
            diagnostics.extend(source_diagnostics)
            for name in invalid_names:
                definitions.pop(name, None)
            definitions.update(loaded)
        return cls(definitions, diagnostics)


def _load_source(
    root: Path,
    source: AgentSource,
    *,
    required: bool,
) -> tuple[dict[str, AgentDefinition], set[str], list[CatalogDiagnostic]]:
    loaded: dict[str, AgentDefinition] = {}
    invalid_names: set[str] = set()
    diagnostics: list[CatalogDiagnostic] = []
    if not root.exists():
        if required:
            raise CatalogLoadError(f"内置 Agent 目录不存在：{root}")
        return loaded, invalid_names, diagnostics
    paths = sorted(path for path in root.glob("*.md") if path.is_file())
    seen: set[str] = set()
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        candidate_name = _peek_name(raw)
        try:
            definition = _parse_definition(raw, source=source, path=path)
            if definition.name in seen:
                raise ValueError(f"同一来源存在重复角色名：{definition.name}")
            seen.add(definition.name)
            loaded[definition.name] = definition
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
            if required:
                raise CatalogLoadError(f"内置 Agent 定义无效：{path.name}: {error}") from error
            if candidate_name is not None:
                invalid_names.add(candidate_name)
                loaded.pop(candidate_name, None)
            diagnostics.append(CatalogDiagnostic(source, path, str(error), candidate_name))
    return loaded, invalid_names, diagnostics


def _parse_definition(raw: str, *, source: AgentSource, path: Path) -> AgentDefinition:
    frontmatter, body = _split_frontmatter(raw)
    loaded = yaml.safe_load(frontmatter)
    if not isinstance(loaded, Mapping):
        raise ValueError("frontmatter 必须是 YAML mapping")
    unknown = set(loaded) - _FIELDS
    if unknown:
        raise ValueError(f"未知 frontmatter 字段：{', '.join(sorted(str(item) for item in unknown))}")
    name = _required_string(loaded, "name")
    if _NAME.fullmatch(name) is None:
        raise ValueError("name 必须为 1-32 位小写字母、数字或连字符")
    description = _required_string(loaded, "description")
    if "\n" in description:
        raise ValueError("description 必须是单行文本")
    prompt = body.strip()
    if not prompt:
        raise ValueError("Agent 正文不能为空")
    tools = _optional_string_list(loaded, "tools")
    disallowed = _optional_string_list(loaded, "disallowedTools") or ()
    profile = loaded.get("profile", "inherit")
    if not isinstance(profile, str) or (
        profile != "inherit" and _PROFILE.fullmatch(profile) is None
    ):
        raise ValueError("profile 必须是 inherit 或有效的 Profile 名称")
    raw_turns = loaded.get("maxTurns")
    if raw_turns is not None and (
        isinstance(raw_turns, bool)
        or not isinstance(raw_turns, int)
        or not 1 <= raw_turns <= MAX_ITERATIONS
    ):
        raise ValueError(f"maxTurns 必须在 1-{MAX_ITERATIONS} 之间")
    raw_mode = loaded.get("permissionMode", PermissionBehavior.INHERIT.value)
    try:
        permission_mode = PermissionBehavior(raw_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"未知 permissionMode：{raw_mode}") from error
    background = loaded.get("background", False)
    if not isinstance(background, bool):
        raise ValueError("background 必须是布尔值")
    return AgentDefinition(
        name=name,
        description=description,
        prompt=prompt,
        source=source,
        path=path,
        tools=tools,
        disallowed_tools=disallowed,
        profile=profile,
        max_turns=raw_turns,
        permission_mode=permission_mode,
        background=background,
    )


def _split_frontmatter(raw: str) -> tuple[str, str]:
    normalized = raw.lstrip("\ufeff")
    lines = normalized.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Agent 定义必须以 YAML frontmatter 开头")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    raise ValueError("Agent frontmatter 缺少结束分隔符")


def _peek_name(raw: str) -> str | None:
    try:
        frontmatter, _ = _split_frontmatter(raw)
        loaded = yaml.safe_load(frontmatter)
    except (ValueError, yaml.YAMLError):
        return None
    if not isinstance(loaded, Mapping):
        return None
    value = loaded.get("name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _required_string(loaded: Mapping[object, object], key: str) -> str:
    value = loaded.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串")
    return value.strip()


def _optional_string_list(
    loaded: Mapping[object, object],
    key: str,
) -> tuple[str, ...] | None:
    value = loaded.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{key} 必须是字符串数组")
    normalized = tuple(dict.fromkeys(item.strip() for item in value))
    return normalized

