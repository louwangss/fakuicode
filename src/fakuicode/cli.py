"""Command-line entry point for Fakuicode."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from fakuicode.config import load_profiles
from fakuicode.errors import ConfigurationError
from fakuicode.hooks.config import HookConfigRepository, HookPaths
from fakuicode.hooks.trust import HookTrustRepository
from fakuicode.models import ProfileSet, ProviderConfig
from fakuicode.mcp.config import McpConfigRepository, McpPaths
from fakuicode.mcp.trust import McpTrustRepository
from fakuicode.memory.identity import MemoryPaths, MemoryRegistry, ProjectIdentityResolver
from fakuicode.memory.maintenance import MemoryMaintenanceRunner
from fakuicode.memory.repository import MemoryRepository
from fakuicode.memory.service import MemoryService
from fakuicode.permissions.config import PermissionConfigRepository, PermissionPaths
from fakuicode.providers.base import ChatProvider
from fakuicode.providers.factory import create_provider
from fakuicode.instructions import InstructionLoader
from fakuicode.renderer import Renderer
from fakuicode.storage import ConversationStore, default_store_path
from fakuicode.skills.trust import SkillTrustRepository
from fakuicode.tui.app import FakuicodeApp


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared by the script and module entry points."""
    return argparse.ArgumentParser(
        prog="fakuicode",
        description="Fakuicode: a streaming terminal chat assistant.",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    renderer: Renderer | None = None,
    config_loader: Callable[[Path], ProviderConfig | ProfileSet] = load_profiles,
    provider_factory: Callable[[ProviderConfig], ChatProvider] = create_provider,
    app_factory: Callable[[ProviderConfig, ChatProvider], FakuicodeApp] | None = None,
    store_factory: Callable[[Path], ConversationStore] = ConversationStore,
    store_path: Path | None = None,
    workspace: Path | None = None,
    permission_home: Path | None = None,
) -> int:
    """Run the command-line program."""
    parser = build_parser()
    parser.add_argument("--config", type=Path, default=Path("fakuicode.yaml"), help="Path to YAML provider configuration.")
    args = parser.parse_args(argv)
    active_renderer = renderer or Renderer()
    try:
        loaded_config = config_loader(args.config)
        profiles = loaded_config if isinstance(loaded_config, ProfileSet) else ProfileSet({"default": loaded_config}, "default")
        config = profiles.active
        provider = provider_factory(config)
    except ConfigurationError as error:
        active_renderer.error(str(error))
        return 2
    if app_factory is not None:
        app_factory(config, provider).run()
        return 0
    active_workspace = (workspace or Path.cwd()).resolve()
    permission_repository = PermissionConfigRepository(
        PermissionPaths.for_workspace(active_workspace, home=permission_home),
        active_workspace,
    )
    permission_snapshot = permission_repository.load()
    mcp_paths = McpPaths.for_workspace(active_workspace, home=permission_home)
    mcp_snapshot = McpConfigRepository(mcp_paths, active_workspace).load()
    mcp_trust_repository = McpTrustRepository(
        (permission_home or Path.home()) / ".fakuicode" / "mcp-trust.yaml"
    )
    hook_trust_repository = HookTrustRepository(
        (permission_home or Path.home()) / ".fakuicode" / "trusted-hooks.yaml"
    )
    hook_repository = HookConfigRepository(
        HookPaths.for_workspace(active_workspace, home=permission_home),
        active_workspace,
        trust_repository=hook_trust_repository,
    )
    hook_snapshot = hook_repository.load()
    store = store_factory(store_path or default_store_path())
    memory_home = permission_home or Path.home()
    memory_service = _build_memory_service(
        active_workspace,
        memory_home,
        provider_factory,
    )
    FakuicodeApp(
        config,
        provider=provider,
        provider_factory=provider_factory,
        store=store,
        profile_name=profiles.active_name,
        profiles=profiles,
        workspace=active_workspace,
        permission_snapshot=permission_snapshot,
        permission_repository=permission_repository,
        mcp_snapshot=mcp_snapshot,
        mcp_trust_repository=mcp_trust_repository,
        hook_snapshot=hook_snapshot,
        hook_repository=hook_repository,
        hook_trust_repository=hook_trust_repository,
        instruction_loader=InstructionLoader(
            active_workspace,
            user_home=memory_home,
        ),
        memory_service=memory_service,
        skill_user_root=memory_home / ".fakuicode" / "skills",
        skill_trust_repository=SkillTrustRepository(
            memory_home / ".fakuicode" / "skill-trust.yaml"
        ),
    ).run()
    return 0


def _build_memory_service(
    workspace: Path,
    home: Path,
    provider_factory: Callable[[ProviderConfig], ChatProvider],
) -> MemoryService | None:
    """Build private automatic memory without making CLI startup depend on it."""

    try:
        paths = MemoryPaths.from_home(home)
        registry = MemoryRegistry(paths)
        repository = MemoryRepository(paths, registry)
        maintenance = MemoryMaintenanceRunner(provider_factory, repository)
        return MemoryService(
            workspace,
            registry,
            ProjectIdentityResolver(registry),
            repository,
            maintenance_runner=maintenance,
        )
    except Exception:
        return None
