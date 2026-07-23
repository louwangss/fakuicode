from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fakuicode.memory.content_policy import serialize_entry
from fakuicode.memory.identity import MemoryPaths, MemoryRegistry
from fakuicode.memory.models import MemoryEntry, MemoryScopeRef, MemorySourceRef
from fakuicode.memory.repository import MemoryRepository
from fakuicode.memory.tool import ReadMemoryEntryTool
from fakuicode.models import ToolCall
from fakuicode.tools.policy import WorkspacePolicy
from fakuicode.tools.registry import ToolRegistry


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


def _bound_tool(tmp_path: Path):
    paths = MemoryPaths.from_home(tmp_path / "home")
    registry = MemoryRegistry(paths)
    repository = MemoryRepository(paths, registry)
    entry = MemoryEntry(
        str(uuid4()), "user", "user_preference", "Concise", "Prefer concise answers.",
        "2026-07-21T00:00:00Z", "2026-07-21T00:00:00Z",
        (MemorySourceRef(CONVERSATION_ID, 1, "user_turn"),),
    )
    notes = repository.scope_path(MemoryScopeRef("user")) / "notes"
    notes.mkdir(parents=True)
    (notes / f"{entry.id}.md").write_bytes(serialize_entry(entry))
    snapshot = repository.combined_snapshot(repository.load_scope(MemoryScopeRef("user")), None)
    return ReadMemoryEntryTool(repository, snapshot), entry


def test_memory_tool_reads_only_a_bound_snapshot_uuid(tmp_path: Path) -> None:
    tool, entry = _bound_tool(tmp_path)

    result = tool.execute({"id": entry.id})
    invalid = tool.execute({"id": "../notes/file.md"})
    unknown = tool.execute({"id": str(uuid4())})

    assert tool.read_only is True
    assert set(tool.definition.input_schema["properties"]) == {"id"}
    assert result.success is True and entry.body in result.output
    assert invalid.success is False and invalid.output == "Memory entry is unavailable."
    assert unknown.success is False and unknown.output == "Memory entry is unavailable."


def test_registry_atomically_replaces_only_host_optional_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(WorkspacePolicy(workspace))
    tool, entry = _bound_tool(tmp_path)

    registry.replace_optional("read_memory_entry", tool)
    assert registry.is_known("read_memory_entry") is True
    assert "read_memory_entry" in [item.name for item in registry.definitions(read_only_only=True)]
    assert registry.execute(ToolCall("call-1", "read_memory_entry", {"id": entry.id}), read_only_only=True).success

    registry.replace_optional("read_memory_entry", None)
    assert registry.is_known("read_memory_entry") is False

    try:
        registry.replace_optional("read_file", tool)
    except ValueError:
        pass
    else:
        raise AssertionError("default tools must not be replaceable")
