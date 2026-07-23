"""Thread bridge for interactive Skill installation confirmation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Event, RLock

from fakuicode.skills.install import SkillInstallDecision, SkillInstallPreview


@dataclass
class _PendingInstall:
    preview: SkillInstallPreview
    completed: Event
    decision: SkillInstallDecision


class SkillInstallBroker:
    """Move a blocking installer confirmation onto the Textual UI thread."""

    def __init__(self, *, poll_seconds: float = 0.05) -> None:
        self._poll_seconds = poll_seconds
        self._queue: deque[_PendingInstall] = deque()
        self._active: _PendingInstall | None = None
        self._closed = False
        self._lock = RLock()

    def request(
        self,
        preview: SkillInstallPreview,
        *,
        cancel_event: Event | None = None,
    ) -> SkillInstallDecision:
        pending = _PendingInstall(
            preview,
            Event(),
            SkillInstallDecision(False, preview.preset),
        )
        with self._lock:
            if self._closed:
                return pending.decision
            self._queue.append(pending)
        while not pending.completed.wait(self._poll_seconds):
            if cancel_event is not None and cancel_event.is_set():
                self._complete(pending, SkillInstallDecision(False, preview.preset))
                break
        return pending.decision

    def next_request(self) -> SkillInstallPreview | None:
        with self._lock:
            if self._closed or self._active is not None:
                return None
            while self._queue:
                pending = self._queue.popleft()
                if pending.completed.is_set():
                    continue
                self._active = pending
                return pending.preview
        return None

    def resolve(self, preview: SkillInstallPreview, decision: SkillInstallDecision) -> bool:
        with self._lock:
            pending = self._active
            if pending is None or pending.preview is not preview:
                return False
            self._active = None
            pending.decision = decision
            pending.completed.set()
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pending = ([self._active] if self._active is not None else []) + list(self._queue)
            self._active = None
            self._queue.clear()
            for item in pending:
                item.decision = SkillInstallDecision(False, item.preview.preset)
                item.completed.set()

    def _complete(self, pending: _PendingInstall, decision: SkillInstallDecision) -> None:
        with self._lock:
            if pending.completed.is_set():
                return
            if self._active is pending:
                self._active = None
            else:
                try:
                    self._queue.remove(pending)
                except ValueError:
                    pass
            pending.decision = decision
            pending.completed.set()
