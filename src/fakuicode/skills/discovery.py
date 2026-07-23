"""Three-layer Skill discovery with fail-closed shadowing."""

from __future__ import annotations

from pathlib import Path

from fakuicode.skills.models import SkillDiagnostic, SkillSnapshot, SkillSource
from fakuicode.skills.parser import SkillParseError, parse_skill_package


class BuiltinSkillError(RuntimeError):
    """A bundled Skill is invalid and startup cannot safely continue."""


class SkillDiscovery:
    def __init__(
        self,
        project_root: Path,
        user_root: Path,
        builtin_root: Path,
        *,
        reserved_commands: frozenset[str] = frozenset(),
    ) -> None:
        self.roots = (
            (SkillSource.PROJECT, project_root),
            (SkillSource.USER, user_root),
            (SkillSource.BUILTIN, builtin_root),
        )
        self.reserved_commands = {name.casefold() for name in reserved_commands}

    def refresh(self, available_tools: set[str]) -> SkillSnapshot:
        claimed: set[str] = set()
        parsed_skills = []
        diagnostics: list[SkillDiagnostic] = []
        for source, root in self.roots:
            candidates = self._candidates(root)
            for key, packages in candidates:
                if key in claimed:
                    continue
                claimed.add(key)
                package = packages[0]
                if len(packages) > 1:
                    diagnostics.append(
                        SkillDiagnostic("same_layer_conflict", package.name, source, "同层存在同名 Skill。")
                    )
                    if source is SkillSource.BUILTIN:
                        raise BuiltinSkillError(f"Bundled Skill name conflict: {package.name}")
                    continue
                try:
                    skill = parse_skill_package(package, source)
                    if skill.name.casefold() in self.reserved_commands:
                        raise SkillParseError("Skill name conflicts with a reserved command")
                except SkillParseError as error:
                    if source is SkillSource.BUILTIN:
                        raise BuiltinSkillError(f"Bundled Skill '{package.name}' is invalid: {error}") from error
                    diagnostics.append(SkillDiagnostic("invalid_skill", package.name, source, str(error)))
                    continue
                parsed_skills.append(skill)

        dormant_tools = {
            runtime_name
            for skill in parsed_skills
            for runtime_name in skill.runtime_tool_names
        }
        known_tools = set(available_tools) | dormant_tools
        skills = {}
        for skill in parsed_skills:
            conflicts = set(skill.runtime_tool_names).intersection(available_tools)
            unknown = set(skill.visible_tools).difference(known_tools)
            if conflicts or unknown:
                code = "tool_name_conflict" if conflicts else "unknown_tool"
                message = (
                    "专属工具运行时名称与已有工具冲突。"
                    if conflicts
                    else "visible-tools 包含当前不存在的工具。"
                )
                if skill.source is SkillSource.BUILTIN:
                    raise BuiltinSkillError(f"Bundled Skill '{skill.name}' has an invalid tool reference")
                diagnostics.append(SkillDiagnostic(code, skill.name, skill.source, message))
                continue
            skills[skill.name] = skill
            source = skill.source
            if skill.author_warnings:
                diagnostics.extend(
                    SkillDiagnostic(code, skill.name, source, "Skill 正文较长，建议拆分引用资料。")
                    for code in skill.author_warnings
                )
        return SkillSnapshot(dict(sorted(skills.items())), tuple(diagnostics))

    @staticmethod
    def _candidates(root: Path) -> tuple[tuple[str, tuple[Path, ...]], ...]:
        if not root.exists():
            return ()
        if not root.is_dir():
            return ()
        grouped: dict[str, list[Path]] = {}
        try:
            children = tuple(root.iterdir())
        except OSError:
            return ()
        for child in children:
            if child.is_dir():
                grouped.setdefault(child.name.casefold(), []).append(child)
        return tuple(
            (key, tuple(sorted(packages, key=lambda item: item.name)))
            for key, packages in sorted(grouped.items())
        )
