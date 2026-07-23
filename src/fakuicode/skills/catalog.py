"""Bounded model-visible Skill catalog rendering."""

from __future__ import annotations

from fakuicode.context import approximate_token_count
from fakuicode.skills.models import SkillCatalog, SkillInvocation, SkillSnapshot, SkillSource


_SOURCE_ORDER = {SkillSource.PROJECT: 0, SkillSource.USER: 1, SkillSource.BUILTIN: 2}


def render_skill_catalog(snapshot: SkillSnapshot, *, context_window: int) -> SkillCatalog:
    budget = max(0, int(context_window * 0.02))
    ordered = sorted(
        (skill for skill in snapshot.skills.values() if skill.invocation is SkillInvocation.AUTO),
        key=lambda skill: (_SOURCE_ORDER[skill.source], skill.name),
    )
    selected = []
    omitted = []
    for index, skill in enumerate(ordered):
        candidate = "\n".join([*(f"- {item.name}" for item in selected), f"- {skill.name}"])
        if approximate_token_count(candidate) <= budget:
            selected.append(skill)
        else:
            omitted.extend(item.name for item in ordered[index:])
            break
    if not selected:
        return SkillCatalog("", 0, tuple(omitted))

    base_lines = [f"- {skill.name}" for skill in selected]
    remaining = budget - approximate_token_count("\n".join(base_lines))
    descriptions = ["" for _ in selected]
    if remaining > 0:
        per_skill = max(0, remaining // len(selected))
        descriptions = [_truncate_tokens(skill.description, per_skill) for skill in selected]

    def render_lines() -> list[str]:
        return [
            f"{base}: {description}" if description else base
            for base, description in zip(base_lines, descriptions, strict=True)
        ]

    lines = render_lines()
    while approximate_token_count("\n".join(lines)) > budget and any(descriptions):
        longest = max(range(len(descriptions)), key=lambda index: len(descriptions[index]))
        descriptions[longest] = descriptions[longest][:-1].rstrip()
        lines = render_lines()
    text = "\n".join(lines)
    return SkillCatalog(text, approximate_token_count(text), tuple(omitted))


def _truncate_tokens(value: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    result = ""
    for character in value:
        candidate = result + character
        if approximate_token_count(candidate) > token_budget:
            break
        result = candidate
    return result
