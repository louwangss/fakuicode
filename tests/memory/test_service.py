from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fakuicode.memory.content_policy import serialize_entry
from fakuicode.memory.identity import MemoryPaths, MemoryRegistry, ProjectIdentityResolver
from fakuicode.memory.models import CompletedTurn, MemoryEntry, MemoryScopeRef, MemorySourceRef
from fakuicode.memory.repository import MemoryRepository
from fakuicode.memory.service import MemoryService
from fakuicode.models import ProviderConfig


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


def _components(tmp_path: Path):
    paths = MemoryPaths.from_home(tmp_path / "home")
    registry = MemoryRegistry(paths)
    repository = MemoryRepository(paths, registry)
    return paths, registry, repository


def _write(repository: MemoryRepository, scope: MemoryScopeRef, entry: MemoryEntry) -> None:
    notes = repository.scope_path(scope) / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / f"{entry.id}.md").write_bytes(serialize_entry(entry))


def _entry(scope: str, category: str, summary: str) -> MemoryEntry:
    return MemoryEntry(
        str(uuid4()),
        scope,  # type: ignore[arg-type]
        category,  # type: ignore[arg-type]
        summary,
        f"Detail: {summary}",
        "2026-07-21T00:00:00Z",
        "2026-07-21T00:00:00Z",
        (MemorySourceRef(CONVERSATION_ID, 1, "user_turn"),),
    )


def test_settings_notice_and_status_are_persistent_and_safe(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths, registry, repository = _components(tmp_path)
    service = MemoryService(workspace, registry, ProjectIdentityResolver(registry), repository)

    assert service.first_notice_needed() is True
    service.confirm_first_notice()
    assert MemoryService(workspace, registry, ProjectIdentityResolver(registry), repository).first_notice_needed() is False
    assert service.set_enabled(False).enabled is False
    assert service.set_enabled(True).generation == 2

    status = service.status()
    assert status.enabled is True
    assert status.user_count == 0
    assert str(paths.root) not in repr(status)


def test_project_isolation_combines_user_and_current_project_only(tmp_path: Path) -> None:
    workspace_a = tmp_path / "project-a"
    workspace_b = tmp_path / "project-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    _, registry, repository = _components(tmp_path)
    resolver = ProjectIdentityResolver(registry)
    identity_a = resolver.resolve(workspace_a)
    identity_b = resolver.resolve(workspace_b)
    assert identity_a is not None and identity_b is not None
    user_entry = _entry("user", "user_preference", "User preference")
    project_entry = _entry("project", "project_knowledge", "Project A fact")
    other_entry = _entry("project", "reference", "Project B reference")
    _write(repository, MemoryScopeRef("user"), user_entry)
    _write(repository, MemoryScopeRef("project", identity_a.project_id), project_entry)
    _write(repository, MemoryScopeRef("project", identity_b.project_id), other_entry)

    context = MemoryService(workspace_a, registry, resolver, repository).capture_turn_context()

    assert context.memory_snapshot is not None
    assert context.memory_snapshot.active_ids == frozenset({user_entry.id, project_entry.id})
    assert other_entry.id not in context.memory_snapshot.rendered


def test_list_visible_entries_exposes_only_safe_fields_from_current_scopes(tmp_path: Path) -> None:
    workspace_a = tmp_path / "project-a"
    workspace_b = tmp_path / "project-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    _, registry, repository = _components(tmp_path)
    resolver = ProjectIdentityResolver(registry)
    identity_a = resolver.resolve(workspace_a)
    identity_b = resolver.resolve(workspace_b)
    assert identity_a is not None and identity_b is not None
    user_entry = _entry("user", "user_preference", "Concise replies")
    project_entry = _entry("project", "project_knowledge", "Uses SQLite")
    other_entry = _entry("project", "reference", "Other project secret")
    _write(repository, MemoryScopeRef("user"), user_entry)
    _write(repository, MemoryScopeRef("project", identity_a.project_id), project_entry)
    _write(repository, MemoryScopeRef("project", identity_b.project_id), other_entry)

    items = MemoryService(workspace_a, registry, resolver, repository).list_visible_entries()

    assert [(item.id, item.scope, item.category, item.summary) for item in items] == [
        (user_entry.id, "user", "user_preference", "Concise replies"),
        (project_entry.id, "project", "project_knowledge", "Uses SQLite"),
    ]
    assert "Detail:" not in repr(items)
    assert other_entry.id not in repr(items)


def test_project_identity_failure_degrades_to_user_memory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry, repository = _components(tmp_path)
    user_entry = _entry("user", "user_preference", "User preference")
    _write(repository, MemoryScopeRef("user"), user_entry)

    class FailedResolver:
        def resolve(self, workspace):
            return None

    context = MemoryService(workspace, registry, FailedResolver(), repository).capture_turn_context()

    assert context.memory_snapshot is not None
    assert context.memory_snapshot.active_ids == frozenset({user_entry.id})
    assert any(item.code == "identity_unavailable" for item in context.memory_snapshot.diagnostics)


def test_registry_failure_drops_memory_but_preserves_request_and_safe_diagnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry, repository = _components(tmp_path)
    service = MemoryService(workspace, registry, ProjectIdentityResolver(registry), repository)

    def fail_state():
        raise RuntimeError("private registry path and traceback")

    monkeypatch.setattr(registry, "user_state", fail_state)
    context = service.capture_turn_context(reminder="keep this reminder")

    assert context.memory_snapshot is None
    assert context.first_request_reminder == "keep this reminder"
    assert service.consume_diagnostic_codes() == ("storage_failure",)


def test_disabled_memory_does_not_load_or_schedule(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry, repository = _components(tmp_path)
    registry.set_enabled(False)

    class ExplodingRepository:
        def load_scope(self, scope):
            raise AssertionError("disabled memory must not load")

    class CapturingCoordinator:
        def __init__(self):
            self.jobs = []

        def submit(self, job):
            self.jobs.append(job)
            return True

        def close(self, *, wait=False):
            pass

    coordinator = CapturingCoordinator()
    service = MemoryService(
        workspace,
        registry,
        ProjectIdentityResolver(registry),
        ExplodingRepository(),  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )
    turn = CompletedTurn(
        CONVERSATION_ID, 1, 2, "hello", "answer", (),
        ProviderConfig("openai", "test", "https://example.test", "key"), None, 1,
    )

    assert service.capture_turn_context().memory_snapshot is None
    assert service.schedule_completed_turn(turn, None) is False
    assert coordinator.jobs == []


def test_cli_builds_memory_service_only_under_temporary_home(tmp_path: Path) -> None:
    from fakuicode.cli import _build_memory_service

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary_home = tmp_path / "temporary-home"

    def provider_factory(config: ProviderConfig):
        del config
        return object()

    service = _build_memory_service(workspace, temporary_home, provider_factory)

    assert service is not None
    assert service.repository.paths.root == temporary_home / ".fakuicode" / "memory"
    assert service.maintenance_runner is not None
    assert service.maintenance_runner.provider_factory is provider_factory
    assert not (workspace / ".fakuicode").exists()
    service.close(wait=False)
