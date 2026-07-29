"""Application-facing orchestration for automatic memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from fakuicode.memory.identity import MemoryRegistry, ProjectIdentity
from fakuicode.memory.maintenance import MaintenanceJob, MemoryMaintenanceCoordinator, MemoryMaintenanceRunner
from fakuicode.memory.models import (
    AgentTurnContext,
    CommitResult,
    CompletedTurn,
    MemoryDiagnostic,
    MemoryScopeRef,
    MemorySnapshot,
    VisibleScopes,
)
from fakuicode.memory.repository import MemoryRepository
from fakuicode.memory.tool import ReadMemoryEntryTool


class IdentityResolver(Protocol):
    def resolve(self, workspace: Path) -> ProjectIdentity | None: ...


@dataclass(frozen=True)
class MemoryStatus:
    enabled: bool
    generation: int
    user_count: int
    project_count: int
    other_project_count: int
    summaries: tuple[str, ...]
    last_update_code: str
    last_update_at: str | None
    diagnostic_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryListItem:
    """The minimal safe fields exposed to a local memory picker."""

    id: str
    scope: str
    category: str
    summary: str


class MemoryService:
    def __init__(
        self,
        workspace: Path,
        registry: MemoryRegistry,
        identity_resolver: IdentityResolver,
        repository: MemoryRepository,
        *,
        maintenance_runner: MemoryMaintenanceRunner | None = None,
        coordinator: MemoryMaintenanceCoordinator | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.registry = registry
        self.identity_resolver = identity_resolver
        self.repository = repository
        self.maintenance_runner = maintenance_runner
        self.coordinator = coordinator
        self._diagnostic_lock = Lock()
        self._pending_diagnostic_codes: list[str] = []
        if self.coordinator is None and self.maintenance_runner is not None:
            self.coordinator = MemoryMaintenanceCoordinator(self._process_job)

    def first_notice_needed(self) -> bool:
        state = self.registry.user_state()
        return state.enabled and not state.notice_shown

    def confirm_first_notice(self) -> None:
        self.registry.mark_notice_shown()

    def set_enabled(self, enabled: bool):
        return self.registry.set_enabled(enabled)

    @property
    def settings_generation(self) -> int:
        return self.registry.user_state().generation

    def capture_turn_context(self, *, reminder: str = "") -> AgentTurnContext:
        try:
            state = self.registry.user_state()
        except Exception:
            self._record_diagnostics(("storage_failure",))
            return AgentTurnContext(first_request_reminder=reminder)
        if not state.enabled:
            return AgentTurnContext(
                first_request_reminder=reminder,
                settings_generation=state.generation,
            )
        try:
            user = self.repository.load_scope(MemoryScopeRef("user"))
            identity = self.identity_resolver.resolve(self.workspace)
            project = None
            identity_diagnostic = None
            if identity is None:
                identity_diagnostic = MemoryDiagnostic("identity_unavailable", "project")
            else:
                project = self.repository.load_scope(MemoryScopeRef("project", identity.project_id))
            snapshot = self.repository.combined_snapshot(user, project)
            if identity_diagnostic is not None:
                snapshot = MemorySnapshot(
                    snapshot.rendered,
                    snapshot.active_ids,
                    snapshot.project_id,
                    snapshot.user_digest,
                    snapshot.project_digest,
                    (*snapshot.diagnostics, identity_diagnostic),
                )
            self._record_diagnostics(item.code for item in snapshot.diagnostics)
            return AgentTurnContext(snapshot, reminder, state.generation)
        except Exception:
            self._record_diagnostics(("storage_failure",))
            return AgentTurnContext(
                first_request_reminder=reminder,
                settings_generation=state.generation,
            )

    def schedule_completed_turn(
        self,
        turn: CompletedTurn,
        snapshot: MemorySnapshot | None,
    ) -> bool:
        if snapshot is None or self.coordinator is None:
            return False
        try:
            state = self.registry.user_state()
            if not state.enabled or state.generation != turn.settings_generation:
                return False
            return self.coordinator.submit(MaintenanceJob(turn, snapshot))
        except Exception:
            return False

    def forget(self, entry_id: str) -> CommitResult:
        identity = self.identity_resolver.resolve(self.workspace)
        scopes = VisibleScopes(
            MemoryScopeRef("user"),
            MemoryScopeRef("project", identity.project_id) if identity is not None else None,
        )
        return self.repository.forget(scopes, entry_id)

    def list_visible_entries(self) -> tuple[MemoryListItem, ...]:
        """List user and current-project notes without exposing note bodies or sources."""
        snapshots = [self.repository.load_scope(MemoryScopeRef("user"))]
        identity = self.identity_resolver.resolve(self.workspace)
        if identity is not None:
            snapshots.append(
                self.repository.load_scope(MemoryScopeRef("project", identity.project_id))
            )
        return tuple(
            MemoryListItem(entry.id, entry.scope, entry.category, entry.summary)
            for snapshot in snapshots
            for entry in snapshot.entries
        )

    def detail_tool(self, snapshot: MemorySnapshot | None):
        if snapshot is None or not snapshot.active_ids:
            return None
        return ReadMemoryEntryTool(self.repository, snapshot)

    def status(self) -> MemoryStatus:
        state = self.registry.user_state()
        diagnostics: list[str] = []
        user = self.repository.load_scope(MemoryScopeRef("user"))
        diagnostics.extend(item.code for item in user.diagnostics)
        identity = self.identity_resolver.resolve(self.workspace)
        project = None
        if identity is not None:
            project = self.repository.load_scope(MemoryScopeRef("project", identity.project_id))
            diagnostics.extend(item.code for item in project.diagnostics)
        summaries = tuple(
            line
            for snapshot in (user, project)
            if snapshot is not None
            for line in snapshot.index.splitlines()
        )[:20]
        return MemoryStatus(
            state.enabled,
            state.generation,
            len(user.entries),
            len(project.entries) if project is not None else 0,
            self.registry.project_count(excluding=identity.project_id if identity is not None else None),
            summaries,
            state.last_update_code,
            state.last_update_at,
            tuple(dict.fromkeys(diagnostics)),
        )

    def close(self, *, wait: bool = False) -> None:
        if self.coordinator is not None:
            self.coordinator.close(wait=wait)

    def consume_diagnostic_codes(self) -> tuple[str, ...]:
        """Return one deduplicated, content-free diagnostic batch for the UI."""

        with self._diagnostic_lock:
            result = tuple(dict.fromkeys(self._pending_diagnostic_codes))
            self._pending_diagnostic_codes.clear()
        return result

    def _record_diagnostics(self, codes) -> None:
        with self._diagnostic_lock:
            self._pending_diagnostic_codes.extend(str(code)[:64] for code in codes)

    def _process_job(self, job: MaintenanceJob):
        try:
            assert self.maintenance_runner is not None
            propose_result = getattr(self.maintenance_runner, "propose_result", None)
            if callable(propose_result):
                proposal = propose_result(job)
                batch = proposal.batch
                skipped_code = proposal.code
            else:
                batch = self.maintenance_runner.propose(job)
                skipped_code = "maintenance_skipped"
            if batch is None:
                self.registry.update_last_status(skipped_code)
                if skipped_code != "maintenance_skipped":
                    self._record_diagnostics((skipped_code,))
                return None
            result = self.repository.apply(
                batch,
                generation=job.turn.settings_generation,
                project_id=job.snapshot.project_id,
            )
            self.registry.update_last_status(result.code)
            if not result.success:
                self._record_diagnostics((result.code,))
            return result
        except Exception:
            self._record_diagnostics(("maintenance_failed",))
            try:
                self.registry.update_last_status("maintenance_failed")
            except Exception:
                pass
            return None
