from __future__ import annotations

from pathlib import Path


def test_repository_skill_examples_follow_the_current_package_contract() -> None:
    from fakuicode.skills.models import SkillExecution, SkillInvocation, SkillSource
    from fakuicode.skills.parser import parse_skill_package

    examples = Path(__file__).parents[2] / "examples" / "skills"

    parsed = {
        package.name: parse_skill_package(package, SkillSource.PROJECT)
        for package in sorted(examples.iterdir())
        if package.is_dir()
    }

    assert set(parsed) == {"backend-interview", "explain-change"}
    interview = parsed["backend-interview"]
    assert interview.invocation is SkillInvocation.AUTO
    assert interview.execution is SkillExecution.SHARED
    assert interview.visible_tools == ()
    assert "$ARGUMENTS" in interview.body
