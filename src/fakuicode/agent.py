"""Provider-neutral bounded ReAct orchestration for native local tools."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from threading import Event
from typing import Protocol

from fakuicode.context_manager import (
    ContextLimitError,
    ContextManagementError,
    ContextManager,
    ContextRecoveryError,
)
from fakuicode.errors import (
    PROVIDER_ERROR_TYPE_VALUES,
    PROVIDER_FAILURE_PHASE_VALUES,
    ProviderError,
    RequestCancelled,
    normalize_provider_request_id,
)
from fakuicode.hooks.models import HookEvent
from fakuicode.hooks.runtime import HookEngine
from fakuicode.memory.models import AgentTurnContext
from fakuicode.models import (
    AgentMessage,
    AgentMode,
    AgentProgress,
    AgentStreamEvent,
    ProviderMessageState,
    ToolCall,
    ToolDefinition,
    ToolResult,
    TokenUsage,
)
from fakuicode.prompting import build_request_envelope
from fakuicode.providers.base import AgentProvider, AgentRequest
from fakuicode.providers.invocation import invoke_provider_stream


MAX_ITERATIONS = 30
PLAN_MODE_SYSTEM_INSTRUCTION = (
    "## 计划模式\n"
    "- 你现在处于只读计划模式，只能使用当前提供的只读工具检查工作区。\n"
    "- 先基于实际代码和配置确认现状，不要凭空设计。\n"
    "- 不得创建、编辑或删除文件，不得执行会改变工作区或外部状态的命令。\n"
    "- 输出一份按依赖顺序排列、可以直接执行和验证的计划，并指出关键文件与风险。\n"
    "- 计划完成后停止，等待用户通过 /do 显式执行。"
)
PLAN_MODE_CONCISE_REMINDER = (
    "## 计划模式提醒\n"
    "保持只读：继续检查必要上下文，禁止修改；最终只返回可执行计划并等待 /do。"
)
_MISSING_FINAL_RESPONSE = (
    "Tool execution completed, but the model did not provide a final response. "
    "Please use the results above or ask a more specific follow-up."
)
_CANCELLED_TOOL_OUTPUT = "Tool execution was cancelled before it completed."
_UNKNOWN_TOOL_OUTPUT = "Tool execution was skipped after repeated unknown tool calls."
_INVALID_TOOL_ARGUMENTS_OUTPUT = (
    "Tool arguments were incomplete or invalid JSON, usually because the model output was truncated. "
    "Retry with a smaller tool call; for a large file, write a smaller base and then use bounded edits."
)


class ToolExecutor(Protocol):
    def definitions(self, *, read_only_only: bool = False) -> list[ToolDefinition]: ...

    def is_known(self, name: str) -> bool: ...

    def is_read_only(self, name: str) -> bool: ...

    def execute(
        self,
        call: ToolCall,
        *,
        cancel_event: Event | None = None,
        read_only_only: bool = False,
    ) -> ToolResult: ...


@dataclass
class StreamCollector:
    """Keep a complete model response while forwarding its live events."""

    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    completed: bool = False
    latest_usage: TokenUsage | None = None
    provider_states: list[ProviderMessageState] = field(default_factory=list)

    def observe(self, event: AgentStreamEvent) -> None:
        if event.kind == "text_delta":
            self.text_parts.append(event.text)
        elif event.kind == "tool_call":
            if event.tool_call is None:
                raise ProviderError("Provider emitted an invalid tool call.")
            self.tool_calls.append(event.tool_call)
        elif event.kind == "completed":
            self.completed = True
        elif event.kind == "usage" and event.usage is not None:
            self.latest_usage = event.usage
        if event.provider_state is not None:
            self.provider_states.append(event.provider_state)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def provider_state(self) -> ProviderMessageState | None:
        if not self.provider_states:
            return None
        protocol = self.provider_states[0].protocol
        if any(state.protocol != protocol for state in self.provider_states):
            raise ProviderError("Provider emitted incompatible message state.")
        return ProviderMessageState(
            protocol,
            tuple(
                block
                for state in self.provider_states
                for block in state.thinking_blocks
            ),
        )


class AgentRunner:
    """Run a bounded ReAct loop and emit UI-neutral state events."""

    def __init__(
        self,
        provider: AgentProvider,
        tools: ToolExecutor,
        *,
        max_iterations: int = MAX_ITERATIONS,
        context_manager: ContextManager | None = None,
        custom_instructions: str = "",
        skill_manager: object | None = None,
        retry_provider_errors: bool = True,
        request_template: AgentRequest | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.context_manager = context_manager
        self.custom_instructions = custom_instructions
        self.skill_manager = skill_manager
        self.retry_provider_errors = retry_provider_errors
        self.request_template = request_template
        self._last_successful_request: AgentRequest | None = None
        candidate_hooks = getattr(tools, "hook_engine", None)
        self.hook_engine = candidate_hooks if isinstance(candidate_hooks, HookEngine) else None

    @property
    def last_successful_request(self) -> AgentRequest | None:
        """Return the immutable request snapshot eligible for a child fork."""

        return self._last_successful_request

    def run(
        self,
        messages: Sequence[AgentMessage],
        *,
        cancel_event: Event | None = None,
        mode: AgentMode = "execute",
        turn_context: AgentTurnContext | None = None,
    ) -> Iterator[AgentStreamEvent]:
        outcome = "failed"
        self._dispatch_hook(
            HookEvent.TURN_START,
            {"turn": {"message_count": len(messages), "outcome": "started"}},
            mode,
        )
        try:
            for event in self._run_loop(
                messages,
                cancel_event=cancel_event,
                mode=mode,
                turn_context=turn_context,
            ):
                if event.kind == "completed":
                    outcome = "completed"
                elif event.kind == "cancelled":
                    outcome = "cancelled"
                elif event.kind == "error":
                    outcome = "failed"
                yield event
        finally:
            self._dispatch_hook(
                HookEvent.TURN_END,
                {"turn": {"message_count": len(messages), "outcome": outcome}},
                mode,
            )

    def _run_loop(
        self,
        messages: Sequence[AgentMessage],
        *,
        cancel_event: Event | None = None,
        mode: AgentMode = "execute",
        turn_context: AgentTurnContext | None = None,
    ) -> Iterator[AgentStreamEvent]:
        set_mode = getattr(self.skill_manager, "set_mode", None)
        if callable(set_mode):
            set_mode(mode)
        begin_request = getattr(self.tools, "begin_request", None)
        if callable(begin_request):
            begin_request()
        history = list(messages)
        active_turn_context = turn_context or AgentTurnContext()
        had_tool_results = False
        consecutive_unknown = 0
        for round_number in range(1, self.max_iterations + 1):
            set_parent_messages = getattr(self.skill_manager, "set_parent_messages", None)
            if callable(set_parent_messages):
                set_parent_messages(tuple(history))
            if _is_cancelled(cancel_event):
                yield AgentStreamEvent("cancelled", "Agent cancelled.")
                return
            yield AgentStreamEvent("progress", progress=AgentProgress(round_number, "model"))
            collector = StreamCollector()
            self._dispatch_hook(
                HookEvent.PRE_MODEL_REQUEST,
                {
                    "message": {
                        "round": round_number,
                        "history_count": len(history),
                        "outcome": "pending",
                    }
                },
                mode,
            )
            request = self._request(
                tuple(history),
                tuple(self._definitions(read_only_only=mode == "plan")),
                cancel_event=cancel_event,
                mode=mode,
                round_number=round_number,
                turn_context=active_turn_context,
            )
            if self.context_manager is not None:
                try:
                    prepared = self.context_manager.prepare_request(request)
                except ContextLimitError as error:
                    self._post_model_hook(round_number, "failed", collector, mode)
                    yield AgentStreamEvent("context_status", context_status=error.status)
                    yield AgentStreamEvent("error", "Agent stopped because context cannot be sent safely.")
                    return
                except ContextManagementError:
                    self._post_model_hook(round_number, "failed", collector, mode)
                    yield AgentStreamEvent("error", "Agent stopped because context preparation failed.")
                    return
                request = prepared.request
                if prepared.status is not None:
                    yield AgentStreamEvent("context_status", context_status=prepared.status)
                if self.hook_engine is not None:
                    request = _append_hook_prompts(
                        request,
                        self.hook_engine.consume_pending_prompts(),
                    )
            try:
                for event in self._stream_with_retries(
                    request,
                    legacy_system_instruction=_plan_instruction(mode, round_number),
                ):
                    collector.observe(event)
                    if event.kind != "completed":
                        yield event
            except RequestCancelled:
                self._post_model_hook(round_number, "cancelled", collector, mode)
                yield AgentStreamEvent("cancelled", "Agent cancelled.")
                return
            except ProviderError as error:
                self._post_model_hook(round_number, "failed", collector, mode)
                yield AgentStreamEvent("error", _provider_error_message(error))
                return

            if not collector.completed:
                self._post_model_hook(round_number, "failed", collector, mode)
                yield AgentStreamEvent("error", "Agent stopped because the provider stream ended unexpectedly.")
                return
            self._post_model_hook(round_number, "completed", collector, mode)
            if (
                self.context_manager is not None
                and collector.latest_usage is not None
                and self._last_successful_request is not None
            ):
                self.context_manager.observe_usage(
                    self._last_successful_request,
                    collector.latest_usage,
                )
            if not collector.tool_calls:
                if had_tool_results and not collector.text.strip():
                    yield AgentStreamEvent("text_delta", _MISSING_FINAL_RESPONSE)
                yield AgentStreamEvent("completed")
                return

            history.append(
                AgentMessage(
                    "assistant",
                    collector.text,
                    tuple(collector.tool_calls),
                    provider_state=collector.provider_state,
                )
            )
            if callable(set_parent_messages):
                set_parent_messages(tuple(history))
            yield AgentStreamEvent("progress", progress=AgentProgress(round_number, "tools"))
            results: list[ToolResult] = []
            cancelled = False
            stop_for_unknown = False

            for batch in self._tool_batches(collector.tool_calls):
                batch_results, batch_cancelled = self._execute_batch(
                    batch,
                    cancel_event,
                    read_only_only=mode == "plan",
                )
                for call, result in zip(batch, batch_results, strict=True):
                    results.append(result)
                    yield AgentStreamEvent("tool_result", tool_result=result)
                    if self._is_known(call.name):
                        consecutive_unknown = 0
                    else:
                        consecutive_unknown += 1
                    if consecutive_unknown >= 2:
                        stop_for_unknown = True
                        break
                if batch_cancelled:
                    cancelled = True
                if cancelled or stop_for_unknown:
                    break

            resolved_ids = {result.call_id for result in results}
            if cancelled:
                for call in collector.tool_calls:
                    if call.id not in resolved_ids:
                        result = _cancelled_result(call)
                        results.append(result)
                        yield AgentStreamEvent("tool_result", tool_result=result)
                history.append(AgentMessage("user", tool_results=tuple(results)))
                yield AgentStreamEvent("cancelled", "Agent cancelled.")
                return
            if stop_for_unknown:
                for call in collector.tool_calls:
                    if call.id not in resolved_ids:
                        result = ToolResult(call.id, call.name, False, _UNKNOWN_TOOL_OUTPUT, "tool call skipped")
                        results.append(result)
                        yield AgentStreamEvent("tool_result", tool_result=result)
                history.append(AgentMessage("user", tool_results=tuple(results)))
                yield AgentStreamEvent("error", "Agent stopped after two consecutive unknown tool calls.")
                return

            history.append(AgentMessage("user", tool_results=tuple(results)))
            had_tool_results = True
            finish_turn = getattr(self.tools, "finish_turn_message", None)
            finish_message = (
                finish_turn(tuple(results))
                if callable(finish_turn)
                else None
            )
            if isinstance(finish_message, str) and finish_message.strip():
                # Start a synthetic final response so session persistence does
                # not duplicate the assistant preamble that carried tool calls.
                yield AgentStreamEvent(
                    "progress",
                    progress=AgentProgress(round_number, "model"),
                )
                yield AgentStreamEvent("text_delta", finish_message.strip())
                yield AgentStreamEvent("completed")
                return
            if round_number == self.max_iterations:
                yield AgentStreamEvent("error", f"Agent stopped after reaching the {self.max_iterations}-round safety limit.")
                return

    def _stream_with_retries(
        self,
        request: AgentRequest,
        *,
        legacy_system_instruction: str,
    ) -> Iterator[AgentStreamEvent]:
        current_request = request
        transient_attempt = 0
        overflow_retried = False
        while True:
            retry_side_effect = False
            try:
                for event in invoke_provider_stream(
                    self.provider,
                    current_request,
                    legacy_system_instruction=legacy_system_instruction,
                ):
                    _raise_if_cancelled(current_request.cancel_event)
                    retry_side_effect = retry_side_effect or event.kind != "usage"
                    yield event
                self._last_successful_request = current_request
                return
            except ProviderError as error:
                if error.category == "context_overflow":
                    if (
                        self.context_manager is not None
                        and not retry_side_effect
                        and not overflow_retried
                        and self.retry_provider_errors
                    ):
                        try:
                            recovered = self.context_manager.recover_overflow(current_request)
                        except ContextRecoveryError as recovery_error:
                            yield AgentStreamEvent(
                                "context_status",
                                context_status=recovery_error.status,
                            )
                            raise ProviderError("Provider context overflow recovery failed.") from error
                        except ContextLimitError as limit_error:
                            yield AgentStreamEvent(
                                "context_status",
                                context_status=limit_error.status,
                            )
                            raise ProviderError("Provider context overflow recovery failed.") from error
                        except ContextManagementError:
                            raise ProviderError("Provider context overflow recovery failed.") from error
                        current_request = recovered.request
                        overflow_retried = True
                        if recovered.status is not None:
                            yield AgentStreamEvent(
                                "context_status",
                                context_status=recovered.status,
                            )
                        continue
                    raise
                if self.retry_provider_errors and error.retryable and not retry_side_effect and transient_attempt < 2:
                    transient_attempt += 1
                    continue
                raise
            except RequestCancelled:
                raise
            except Exception as error:
                raise ProviderError("Provider stream failed.") from error

    def _request(
        self,
        history: tuple[AgentMessage, ...],
        tools: tuple[ToolDefinition, ...],
        *,
        cancel_event: Event | None,
        mode: AgentMode,
        round_number: int,
        turn_context: AgentTurnContext,
        consume_hook_prompts: bool = True,
    ) -> AgentRequest:
        if self.request_template is not None:
            return replace(
                self.request_template,
                messages=history,
                tools=tools,
                cancel_event=cancel_event,
            )
        memory_snapshot = turn_context.memory_snapshot
        automatic_memory_enabled = memory_snapshot is not None
        memory = memory_snapshot.rendered if memory_snapshot is not None else ""
        first_request_reminder = turn_context.first_request_reminder if round_number == 1 else ""
        skill_catalog = getattr(self.skill_manager, "catalog_text", "")
        active_skill_prompt = getattr(self.skill_manager, "active_prompt", "")
        hook_prompts = ()
        if self.hook_engine is not None:
            hook_prompts = (
                self.hook_engine.consume_prompts()
                if consume_hook_prompts
                else self.hook_engine.peek_prompts()
            )
        if (
            not self.custom_instructions
            and not memory
            and not automatic_memory_enabled
            and not first_request_reminder
            and not skill_catalog
            and not active_skill_prompt
            and not hook_prompts
            and not _supports_structured_request(self.provider)
        ):
            return AgentRequest(history, tools, cancel_event=cancel_event)
        workspace = getattr(getattr(self.tools, "policy", None), "workspace", None)
        from pathlib import Path

        safe_workspace = workspace if isinstance(workspace, Path) else Path.cwd()
        config = getattr(self.provider, "config", None)
        model = getattr(config, "model", "unknown")
        envelope = build_request_envelope(
            workspace=safe_workspace,
            model=model if isinstance(model, str) else "unknown",
            reminder="\n\n".join(
                item
                for item in (_plan_instruction(mode, round_number), first_request_reminder)
                if item
            ),
            custom_instructions=self.custom_instructions,
            skill_catalog=skill_catalog if isinstance(skill_catalog, str) else "",
            active_skills=(active_skill_prompt,) if isinstance(active_skill_prompt, str) else (),
            long_term_memory=memory,
            automatic_memory_enabled=automatic_memory_enabled,
        )
        return _append_hook_prompts(
            AgentRequest(history, tools, envelope.stable, envelope.supplement, cancel_event),
            hook_prompts,
        )

    def _dispatch_hook(
        self,
        event: HookEvent,
        payload: dict[str, object],
        mode: AgentMode,
    ) -> None:
        if self.hook_engine is not None:
            self.hook_engine.dispatch(event, payload, plan_mode=mode == "plan")

    def _post_model_hook(
        self,
        round_number: int,
        outcome: str,
        collector: StreamCollector,
        mode: AgentMode,
    ) -> None:
        message: dict[str, object] = {
            "round": round_number,
            "outcome": outcome,
            "tool_call_count": len(collector.tool_calls),
        }
        if collector.text:
            message["text"] = collector.text
        self._dispatch_hook(HookEvent.POST_MODEL_RESPONSE, {"message": message}, mode)

    def build_request(
        self,
        history: Sequence[AgentMessage],
        *,
        cancel_event: Event | None = None,
        mode: AgentMode = "execute",
        round_number: int = 1,
        turn_context: AgentTurnContext | None = None,
    ) -> AgentRequest:
        """Build the same provider envelope used by a normal model round."""

        return self._request(
            tuple(history),
            tuple(self._definitions(read_only_only=mode == "plan")),
            cancel_event=cancel_event,
            mode=mode,
            round_number=round_number,
            turn_context=turn_context or AgentTurnContext(),
            consume_hook_prompts=False,
        )

    def _definitions(self, *, read_only_only: bool) -> list[ToolDefinition]:
        if not read_only_only:
            return self.tools.definitions()
        try:
            return self.tools.definitions(read_only_only=True)
        except TypeError:
            return [definition for definition in self.tools.definitions() if self._is_read_only(definition.name)]

    def _tool_batches(self, calls: Sequence[ToolCall]) -> Iterator[tuple[ToolCall, ...]]:
        pending_read_only: list[ToolCall] = []
        for call in calls:
            if self._is_known(call.name) and self._is_read_only(call.name):
                pending_read_only.append(call)
                continue
            if pending_read_only:
                yield tuple(pending_read_only)
                pending_read_only.clear()
            yield (call,)
        if pending_read_only:
            yield tuple(pending_read_only)

    def _execute_batch(
        self,
        calls: Sequence[ToolCall],
        cancel_event: Event | None,
        *,
        read_only_only: bool,
    ) -> tuple[list[ToolResult], bool]:
        if _is_cancelled(cancel_event):
            return ([_cancelled_result(call) for call in calls], True)
        if len(calls) == 1:
            call = calls[0]
            try:
                return ([self._execute_tool(call, cancel_event, read_only_only=read_only_only)], False)
            except RequestCancelled:
                return ([_cancelled_result(call)], True)
            except Exception:
                return ([_failed_result(call)], False)

        executor = ThreadPoolExecutor(max_workers=len(calls), thread_name_prefix="fakuicode-read")
        futures: list[Future[ToolResult]] = [
            executor.submit(
                self._execute_tool,
                call,
                cancel_event,
                read_only_only=read_only_only,
            )
            for call in calls
        ]
        results: list[ToolResult] = []
        cancelled = False
        try:
            for call, future in zip(calls, futures, strict=True):
                if _is_cancelled(cancel_event):
                    cancelled = True
                    break
                try:
                    results.append(future.result())
                except RequestCancelled:
                    cancelled = True
                    break
                except Exception:
                    results.append(_failed_result(call))
        finally:
            if cancelled:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
        if cancelled:
            results.extend(_cancelled_result(call) for call in calls[len(results) :])
        return results, cancelled

    def _execute_tool(
        self,
        call: ToolCall,
        cancel_event: Event | None,
        *,
        read_only_only: bool,
    ) -> ToolResult:
        import inspect

        if call.argument_error == "invalid_json":
            return ToolResult(
                call.id,
                call.name,
                False,
                _INVALID_TOOL_ARGUMENTS_OUTPUT,
                "incomplete or invalid tool arguments",
            )
        try:
            supports_context = "read_only_only" in inspect.signature(self.tools.execute).parameters
        except (TypeError, ValueError):
            supports_context = False
        if supports_context:
            return self.tools.execute(
                call,
                cancel_event=cancel_event,
                read_only_only=read_only_only,
            )
        return self.tools.execute(call, cancel_event=cancel_event)

    def _is_known(self, name: str) -> bool:
        classifier = getattr(self.tools, "is_known", None)
        return bool(classifier(name)) if callable(classifier) else True

    def _is_read_only(self, name: str) -> bool:
        classifier = getattr(self.tools, "is_read_only", None)
        return bool(classifier(name)) if callable(classifier) else False

    def cancel(self) -> None:
        """Close an active provider stream when the UI requests cancellation."""
        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            cancel()


def _cancelled_result(call: ToolCall) -> ToolResult:
    return ToolResult(call.id, call.name, False, _CANCELLED_TOOL_OUTPUT, "tool action cancelled")


def _failed_result(call: ToolCall) -> ToolResult:
    return ToolResult(call.id, call.name, False, "Tool execution failed.", "tool action failed")


def _is_cancelled(cancel_event: Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if _is_cancelled(cancel_event):
        raise RequestCancelled()


def _append_hook_prompts(
    request: AgentRequest,
    prompts: tuple[str, ...],
) -> AgentRequest:
    if not prompts:
        return request
    hook_prompt = "\n\n".join(prompts)
    supplement = "\n\n".join(
        item
        for item in (
            request.system_supplement,
            f"## 生命周期 Hook 注入\n{hook_prompt}",
        )
        if item
    )
    return replace(request, system_supplement=supplement)


def _provider_error_message(error: ProviderError) -> str:
    """Render only bounded diagnostic fields, never the Provider's raw detail."""

    details: list[str] = []
    if error.status_code is not None:
        details.append(f"HTTP {error.status_code}")
    if error.error_type in PROVIDER_ERROR_TYPE_VALUES:
        details.append(f"type={error.error_type}")
    if error.failure_phase in PROVIDER_FAILURE_PHASE_VALUES:
        details.append(f"phase={error.failure_phase}")
    request_id = normalize_provider_request_id(error.request_id)
    if request_id is not None:
        details.append(f"request={request_id}")
    if not details:
        return "Agent stopped because the provider stream failed."
    return f"Agent stopped because the provider request failed ({'; '.join(details)})."


def _plan_instruction(mode: AgentMode, round_number: int) -> str:
    if mode != "plan":
        return ""
    return PLAN_MODE_SYSTEM_INSTRUCTION if round_number == 1 or round_number % 3 == 0 else PLAN_MODE_CONCISE_REMINDER


def _supports_structured_request(provider: object) -> bool:
    """Keep old custom providers working while built-in providers use AgentRequest."""

    import inspect

    try:
        return "request" in inspect.signature(provider.stream_agent).parameters  # type: ignore[attr-defined]
    except (TypeError, ValueError, AttributeError):
        return False
