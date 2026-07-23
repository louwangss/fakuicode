"""Read-only exact-ID tool bound to one immutable memory snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Event

from fakuicode.errors import ToolExecutionError
from fakuicode.memory.models import MemoryScopeRef, MemorySnapshot, canonical_uuid
from fakuicode.memory.repository import MemoryRepository, MemoryRepositoryError
from fakuicode.models import ToolDefinition
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments


class ReadMemoryEntryTool:
    def __init__(self, repository: MemoryRepository, snapshot: MemorySnapshot) -> None:
        self.repository = repository
        self.snapshot = snapshot

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            "read_memory_entry",
            "按当前轮次记忆索引中的精确 UUID 读取一条详情；不支持路径、关键词、搜索或跨项目枚举。",
            {
                "type": "object",
                "required": ["id"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "format": "uuid"},
                },
            },
        )

    @property
    def read_only(self) -> bool:
        return True

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        if set(arguments) != {"id"} or not isinstance(arguments.get("id"), str):
            raise ToolExecutionError("Memory entry is unavailable.")
        entry_id = arguments["id"]
        try:
            canonical_uuid(entry_id)
        except ValueError as error:
            raise ToolExecutionError("Memory entry is unavailable.") from error
        if entry_id not in self.snapshot.active_ids:
            raise ToolExecutionError("Memory entry is unavailable.")
        return ToolPreparation(freeze_arguments({"id": entry_id}), f"memory:{entry_id}")

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        try:
            return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)
        except (ToolExecutionError, MemoryRepositoryError):
            return ToolExecution(False, "Memory entry is unavailable.", "memory entry unavailable")

    def execute_prepared(
        self,
        arguments: Mapping[str, object],
        *,
        cancel_event: Event | None = None,
    ) -> ToolExecution:
        del cancel_event
        entry_id = arguments.get("id")
        if not isinstance(entry_id, str) or entry_id not in self.snapshot.active_ids:
            raise ToolExecutionError("Memory entry is unavailable.")
        try:
            entry = self.repository.read_active(MemoryScopeRef("user"), entry_id)
        except MemoryRepositoryError:
            if self.snapshot.project_id is None:
                raise
            entry = self.repository.read_active(
                MemoryScopeRef("project", self.snapshot.project_id), entry_id
            )
        output = (
            f"ID: {entry.id}\n"
            f"Scope: {entry.scope}\n"
            f"Category: {entry.category}\n"
            f"Summary: {entry.summary}\n\n"
            f"{entry.body}"
        )
        return ToolExecution(True, output, f"read memory entry {entry.id}")
