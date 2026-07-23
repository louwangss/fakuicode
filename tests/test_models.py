from __future__ import annotations

import json

from fakuicode.errors import ProviderError, RequestCancelled, ToolPolicyError
from fakuicode.models import (
    AgentProgress,
    AgentStreamEvent,
    ContextStatus,
    ProfileSet,
    ProviderConfig,
    TimelineEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def test_profile_set_returns_the_selected_named_profile() -> None:
    primary = ProviderConfig("anthropic", "claude-test", "https://api.example.test/v1", "secret")
    fallback = ProviderConfig("openai", "gpt-test", "https://api.example.test/v1", "other-secret")

    profiles = ProfileSet({"primary": primary, "fallback": fallback}, "primary")

    assert profiles.active_name == "primary"
    assert profiles.active is primary
    assert profiles.get("fallback") is fallback


def test_timeline_tool_models_preserve_safe_structured_data() -> None:
    definition = ToolDefinition("read_file", "Read a workspace file", {"type": "object"})
    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    result = ToolResult("call-1", "read_file", True, "# Fakuicode", "read README.md")
    event = TimelineEvent(3, "tool_result", result.output, call_id=call.id, metadata={"summary": result.summary})

    assert definition.name == "read_file"
    assert call.arguments["path"] == "README.md"
    assert event.sequence == 3
    assert event.metadata["summary"] == "read README.md"


def test_tool_result_serializes_its_machine_readable_content() -> None:
    from fakuicode.tools.base import ToolExecution

    execution = ToolExecution(True, "1: hello", "read notes.txt")
    result = ToolResult("call-1", "read_file", execution.success, execution.output, execution.summary)

    assert execution.success is True
    assert json.loads(result.to_model_content()) == {
        "success": True,
        "summary": "read notes.txt",
        "output": "1: hello",
    }


def test_errors_expose_retryability_without_sensitive_details() -> None:
    retryable = ProviderError("service unavailable", retryable=True)

    assert retryable.retryable is True
    assert str(RequestCancelled()) == "Request cancelled."
    assert "outside" in str(ToolPolicyError("Path is outside the workspace."))


def test_agent_events_carry_typed_progress_and_token_usage() -> None:
    progress = AgentProgress(round_number=2, phase="tools")
    usage = TokenUsage(input_tokens=120, output_tokens=45)
    event = AgentStreamEvent("usage", usage=usage)

    assert progress.round_number == 2
    assert progress.phase == "tools"
    assert event.usage == usage
    assert event.text == ""


def test_token_usage_accepts_an_optional_normalized_context_input_count() -> None:
    legacy = TokenUsage(input_tokens=120, output_tokens=45)
    normalized = TokenUsage(
        input_tokens=20,
        output_tokens=5,
        cache_read_tokens=70,
        cache_write_tokens=30,
        context_input_tokens=120,
    )

    assert legacy.context_input_tokens is None
    assert normalized.context_input_tokens == 120
    assert normalized.input_tokens == 20


def test_context_status_contains_only_provider_neutral_non_content_state() -> None:
    status = ContextStatus(
        trigger="automatic",
        result="failed",
        estimated_before=116_000,
        estimated_after=None,
        artifact_count=2,
        artifact_bytes=48_000,
        duration_seconds=1.25,
        consecutive_failures=3,
        error_category="invalid_summary",
        recovery_hint="Use /compact or /clear.",
    )
    event = AgentStreamEvent("context_status", context_status=status)

    assert event.context_status is status
    assert status.trigger == "automatic"
    assert status.result == "failed"
    assert status.estimated_before == 116_000
    assert status.consecutive_failures == 3
    assert status.recovery_hint == "Use /compact or /clear."
