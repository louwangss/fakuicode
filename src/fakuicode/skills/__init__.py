"""Public Skill subsystem contracts."""

from fakuicode.skills.catalog import render_skill_catalog
from fakuicode.skills.discovery import BuiltinSkillError, SkillDiscovery
from fakuicode.skills.models import (
    ActiveSkill,
    SkillCatalog,
    SkillDefinition,
    SkillDiagnostic,
    SkillExecution,
    SkillInvocation,
    SkillSnapshot,
    SkillSource,
    SkillToolSpec,
)
from fakuicode.skills.parser import SkillParseError, fingerprint_package, parse_skill_package
from fakuicode.skills.runtime import LoadSkillTool, SkillManager
from fakuicode.skills.install import (
    GitHubSkillFetcher,
    SkillInstallDecision,
    SkillInstallError,
    SkillInstaller,
    SkillInstallPreset,
    SkillInstallPreview,
    SkillPackageFetcher,
    SkillInstallRequest,
    SkillInstallResult,
    SkillInstallScope,
    parse_install_source,
)
from fakuicode.skills.trust import (
    SkillTrustIdentity,
    SkillTrustRepository,
    SkillTrustRequest,
    SkillTrustStorageError,
)


def __getattr__(name: str):
    """Lazily expose isolated execution without importing the session at package load."""

    if name in {"IsolatedSkillExecutor", "select_recent_user_turns"}:
        from fakuicode.skills.isolated import IsolatedSkillExecutor, select_recent_user_turns

        exports = {
            "IsolatedSkillExecutor": IsolatedSkillExecutor,
            "select_recent_user_turns": select_recent_user_turns,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ActiveSkill",
    "BuiltinSkillError",
    "SkillCatalog",
    "SkillDefinition",
    "SkillDiagnostic",
    "SkillDiscovery",
    "SkillExecution",
    "SkillInstallDecision",
    "SkillInstallError",
    "SkillInstaller",
    "SkillInstallPreset",
    "SkillInstallPreview",
    "SkillPackageFetcher",
    "SkillInstallRequest",
    "SkillInstallResult",
    "SkillInstallScope",
    "SkillInvocation",
    "SkillManager",
    "SkillParseError",
    "SkillSnapshot",
    "SkillSource",
    "SkillToolSpec",
    "SkillTrustIdentity",
    "SkillTrustRepository",
    "SkillTrustRequest",
    "SkillTrustStorageError",
    "LoadSkillTool",
    "GitHubSkillFetcher",
    "IsolatedSkillExecutor",
    "select_recent_user_turns",
    "fingerprint_package",
    "parse_skill_package",
    "parse_install_source",
    "render_skill_catalog",
]
