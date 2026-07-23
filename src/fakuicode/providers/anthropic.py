from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from threading import Event, RLock

import httpx

from fakuicode.errors import (
    PROVIDER_ERROR_TYPE_VALUES,
    ProviderError,
    ProviderErrorType,
    RequestCancelled,
    normalize_provider_request_id,
)
from fakuicode.models import (
    AgentMessage,
    AgentStreamEvent,
    Message,
    ProviderConfig,
    ProviderMessageState,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from fakuicode.providers.base import AGENT_SYSTEM_PROMPT, AgentRequest, ProviderCapabilities
from fakuicode.providers.dsml import ToolMarkupAccumulator
from fakuicode.providers.sse import parse_sse


RESPONSE_FAILED = "Anthropic response failed."
REQUEST_FAILED = "Anthropic request failed."
STREAM_FORMAT_FAILED = "Anthropic stream format failed."
STREAM_REPORTED_ERROR = "Anthropic stream reported an error."
_CAPABILITIES = ProviderCapabilities(supports_output_token_limit=True)
_CONTEXT_OVERFLOW_PHRASES = (
    "prompt is too long",
    "context window",
    "maximum context length",
    "too many tokens",
)
_RETRYABLE_ERROR_TYPES: frozenset[ProviderErrorType] = frozenset(
    {"rate_limit_error", "api_error", "overloaded_error"}
)


class AnthropicProvider:
    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None) -> None:
        self.config, self.client = config, client or httpx.Client(timeout=60.0)
        self._active_response: httpx.Response | None = None
        self._response_lock = RLock()

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _CAPABILITIES

    def cancel(self) -> None:
        """Interrupt the active HTTP stream, if this provider owns one."""
        with self._response_lock:
            response = self._active_response
        if response is not None:
            response.close()

    def stream_chat(self, messages: Sequence[Message]) -> Iterator[StreamEvent]:
        body: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "stream": True,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
        }
        thinking = _thinking_request(self.config)
        if thinking is not None:
            body["thinking"] = thinking

        thinking_open = False
        try:
            with self.client.stream(
                "POST",
                f"{self.config.base_url}/messages",
                headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"},
                json=body,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    raise _response_error(response) from error

                try:
                    for event in parse_sse(response.iter_lines()):
                        if event.name == "error":
                            raise _stream_event_error(event.data, response)
                        payload = _parse_payload(event.data)
                        if event.name == "content_block_start" and _is_thinking_block(payload):
                            thinking_open = True
                            yield StreamEvent("thinking_start")
                        elif event.name == "content_block_stop" and thinking_open:
                            thinking_open = False
                            yield StreamEvent("thinking_end")
                        elif event.name == "content_block_delta":
                            delta = payload.get("delta")
                            if not isinstance(delta, Mapping):
                                raise ProviderError(STREAM_FORMAT_FAILED)
                            if delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str):
                                yield StreamEvent("thinking_delta", delta["thinking"])
                            elif delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                                yield StreamEvent("text_delta", delta["text"])
                        elif event.name == "message_stop":
                            yield StreamEvent("completed")
                            return
                except ProviderError as error:
                    if error.failure_phase in {"stream_event", "stream_format"}:
                        raise
                    raise ProviderError(STREAM_FORMAT_FAILED, failure_phase="stream_format") from error
        except ProviderError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise ProviderError(REQUEST_FAILED, retryable=True, failure_phase="stream_transport") from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError(REQUEST_FAILED, failure_phase="request") from error
        raise ProviderError("Anthropic stream ended before completion.", failure_phase="stream_transport")

    def stream_agent(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
        *,
        cancel_event: Event | None = None,
        system_instruction: str = "",
        request: AgentRequest | None = None,
    ) -> Iterator[AgentStreamEvent]:
        agent_request = request or AgentRequest(
            tuple(messages), tuple(tools), _system_prompt(system_instruction), cancel_event=cancel_event
        )
        body: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": agent_request.output_token_limit or 4096,
            "stream": True,
            "system": _anthropic_system(agent_request, use_cache=request is not None),
            "messages": _agent_messages(agent_request.messages),
        }
        if agent_request.tools:
            body["tools"] = [
                {"name": tool.name, "description": tool.description, "input_schema": dict(tool.input_schema)}
                for tool in agent_request.tools
            ]
        thinking = _thinking_request(self.config)
        if thinking is not None and (
            thinking.get("type") == "disabled"
            or not _has_tool_history(agent_request.messages)
            or _has_anthropic_thinking_state(agent_request.messages)
        ):
            body["thinking"] = thinking

        thinking_block: dict[str, object] | None = None
        latest_usage = TokenUsage()
        tool_blocks: dict[int, dict[str, object]] = {}
        tool_markup = ToolMarkupAccumulator()
        try:
            with self.client.stream(
                "POST",
                f"{self.config.base_url}/messages",
                headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"},
                json=body,
            ) as response:
                with self._response_lock:
                    self._active_response = response
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    with self._response_lock:
                        if self._active_response is response:
                            self._active_response = None
                    raise _response_error(response) from error

                try:
                    for event in parse_sse(response.iter_lines()):
                        _raise_if_cancelled(agent_request.cancel_event)
                        if event.name == "error":
                            raise _stream_event_error(event.data, response)
                        payload = _parse_payload(event.data)
                        usage = _anthropic_usage(payload, latest_usage)
                        if usage is not None:
                            latest_usage = usage
                            yield AgentStreamEvent("usage", usage=usage)
                        if event.name == "content_block_start":
                            if _is_thinking_block(payload):
                                thinking_block = _start_thinking_block(payload)
                                yield AgentStreamEvent("thinking_start")
                            elif _is_redacted_thinking_block(payload):
                                thinking_block = _start_thinking_block(payload)
                            else:
                                _begin_tool_block(payload, tool_blocks)
                        elif event.name == "content_block_stop":
                            if thinking_block is not None:
                                yield AgentStreamEvent(
                                    "thinking_end",
                                    provider_state=ProviderMessageState(
                                        "anthropic",
                                        (dict(thinking_block),),
                                    ),
                                )
                                thinking_block = None
                            else:
                                call = _finish_tool_block(payload, tool_blocks)
                                if call is not None:
                                    yield AgentStreamEvent("tool_call", tool_call=call)
                        elif event.name == "content_block_delta":
                            delta = payload.get("delta")
                            if not isinstance(delta, Mapping):
                                raise ProviderError(STREAM_FORMAT_FAILED)
                            if delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str):
                                if thinking_block is None:
                                    raise ProviderError(STREAM_FORMAT_FAILED)
                                thinking_block["thinking"] = str(thinking_block.get("thinking", "")) + delta["thinking"]
                                yield AgentStreamEvent("thinking_delta", delta["thinking"])
                            elif delta.get("type") == "signature_delta" and isinstance(delta.get("signature"), str):
                                if thinking_block is None:
                                    raise ProviderError(STREAM_FORMAT_FAILED)
                                thinking_block["signature"] = str(thinking_block.get("signature", "")) + delta["signature"]
                            elif delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                                for chunk in tool_markup.append(delta["text"]):
                                    yield AgentStreamEvent("text_delta", chunk)
                            elif delta.get("type") == "input_json_delta":
                                _append_tool_arguments(payload, delta, tool_blocks)
                        elif event.name == "message_stop":
                            visible_text, raw_calls = tool_markup.finish()
                            for chunk in visible_text:
                                yield AgentStreamEvent("text_delta", chunk)
                            if raw_calls is not None:
                                for call in raw_calls:
                                    yield AgentStreamEvent("tool_call", tool_call=call)
                            yield AgentStreamEvent("completed")
                            return
                except ProviderError as error:
                    if error.failure_phase in {"stream_event", "stream_format"}:
                        raise
                    raise ProviderError(STREAM_FORMAT_FAILED, failure_phase="stream_format") from error
                finally:
                    with self._response_lock:
                        if self._active_response is response:
                            self._active_response = None
        except ProviderError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as error:
            _raise_if_cancelled(agent_request.cancel_event)
            raise ProviderError(REQUEST_FAILED, retryable=True, failure_phase="stream_transport") from error
        except (httpx.HTTPError, ValueError) as error:
            _raise_if_cancelled(agent_request.cancel_event)
            raise ProviderError(REQUEST_FAILED, failure_phase="request") from error
        raise ProviderError("Anthropic stream ended before completion.", failure_phase="stream_transport")


def _parse_payload(data: str) -> Mapping[str, object]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise ProviderError(STREAM_FORMAT_FAILED, failure_phase="stream_format") from error
    if not isinstance(payload, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED, failure_phase="stream_format")
    return payload


def _system_prompt(instruction: str) -> str:
    return AGENT_SYSTEM_PROMPT if not instruction else f"{AGENT_SYSTEM_PROMPT}\n\n{instruction}"


def _thinking_request(config: ProviderConfig) -> dict[str, object] | None:
    if config.thinking is None:
        return None
    if config.model.casefold().startswith("deepseek-"):
        return {"type": "enabled" if config.thinking.enabled else "disabled"}
    if config.thinking.enabled:
        return {"type": "adaptive", "display": "summarized"}
    return None


def _anthropic_system(request: AgentRequest, *, use_cache: bool) -> str | list[dict[str, object]]:
    """Use Anthropic's explicit cache boundary only for structured Agent requests."""

    if not use_cache:
        return request.system_prompt
    stable: dict[str, object] = {
        "type": "text",
        "text": request.system_prompt,
        "cache_control": {"type": "ephemeral"},
    }
    return [stable, *([{ "type": "text", "text": request.system_supplement }] if request.system_supplement else [])]


def _anthropic_usage(payload: Mapping[str, object], previous: TokenUsage) -> TokenUsage | None:
    raw_usage = payload.get("usage")
    if raw_usage is None:
        message = payload.get("message")
        raw_usage = message.get("usage") if isinstance(message, Mapping) else None
    if not isinstance(raw_usage, Mapping):
        return None
    input_tokens = _optional_int(raw_usage.get("input_tokens"))
    output_tokens = _optional_int(raw_usage.get("output_tokens"))
    cache_read_tokens = _optional_int(raw_usage.get("cache_read_input_tokens"))
    cache_write_tokens = _optional_int(raw_usage.get("cache_creation_input_tokens"))
    if input_tokens is None and output_tokens is None and cache_read_tokens is None and cache_write_tokens is None:
        return None
    normalized_input = previous.input_tokens if input_tokens is None else input_tokens
    normalized_output = previous.output_tokens if output_tokens is None else output_tokens
    normalized_cache_read = (
        previous.cache_read_tokens if cache_read_tokens is None else cache_read_tokens
    )
    normalized_cache_write = (
        previous.cache_write_tokens if cache_write_tokens is None else cache_write_tokens
    )
    context_components = (normalized_input, normalized_cache_read, normalized_cache_write)
    context_input_tokens = (
        sum(value or 0 for value in context_components)
        if any(value is not None for value in context_components)
        else None
    )
    return TokenUsage(
        input_tokens=normalized_input,
        output_tokens=normalized_output,
        cache_read_tokens=normalized_cache_read,
        cache_write_tokens=normalized_cache_write,
        context_input_tokens=context_input_tokens,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _is_thinking_block(payload: Mapping[str, object]) -> bool:
    content_block = payload.get("content_block")
    return isinstance(content_block, Mapping) and content_block.get("type") == "thinking"


def _is_redacted_thinking_block(payload: Mapping[str, object]) -> bool:
    content_block = payload.get("content_block")
    return isinstance(content_block, Mapping) and content_block.get("type") == "redacted_thinking"


def _start_thinking_block(payload: Mapping[str, object]) -> dict[str, object]:
    content_block = payload.get("content_block")
    if not isinstance(content_block, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    return dict(content_block)


def _agent_messages(messages: Sequence[AgentMessage]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        if message.role == "assistant":
            content: list[dict[str, object]] = []
            if message.provider_state is not None and message.provider_state.protocol == "anthropic":
                content.extend(dict(block) for block in message.provider_state.thinking_blocks)
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
                for call in message.tool_calls
            )
            result.append({"role": "assistant", "content": content})
            continue

        if message.tool_results:
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_result.call_id,
                    "content": tool_result.to_model_content(),
                    "is_error": not tool_result.success,
                }
                for tool_result in message.tool_results
            ]
            if message.content:
                content.insert(0, {"type": "text", "text": message.content})
            result.append({"role": "user", "content": content})
        else:
            result.append({"role": "user", "content": message.content})
    return result


def _has_tool_history(messages: Sequence[AgentMessage]) -> bool:
    return any(message.tool_calls or message.tool_results for message in messages)


def _has_anthropic_thinking_state(messages: Sequence[AgentMessage]) -> bool:
    return any(
        message.provider_state is not None
        and message.provider_state.protocol == "anthropic"
        and message.provider_state.thinking_blocks
        for message in messages
    )


def _begin_tool_block(payload: Mapping[str, object], tool_blocks: dict[int, dict[str, object]]) -> None:
    content_block = payload.get("content_block")
    index = payload.get("index")
    if not isinstance(content_block, Mapping) or content_block.get("type") != "tool_use":
        return
    if not isinstance(index, int):
        raise ProviderError(STREAM_FORMAT_FAILED)
    call_id, name, initial_input = content_block.get("id"), content_block.get("name"), content_block.get("input")
    if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(initial_input, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    tool_blocks[index] = {"id": call_id, "name": name, "initial_input": dict(initial_input), "partial_json": ""}


def _append_tool_arguments(
    payload: Mapping[str, object], delta: Mapping[str, object], tool_blocks: dict[int, dict[str, object]]
) -> None:
    index, partial_json = payload.get("index"), delta.get("partial_json")
    if not isinstance(index, int) or not isinstance(partial_json, str) or index not in tool_blocks:
        raise ProviderError(STREAM_FORMAT_FAILED)
    tool_blocks[index]["partial_json"] = str(tool_blocks[index]["partial_json"]) + partial_json


def _finish_tool_block(payload: Mapping[str, object], tool_blocks: dict[int, dict[str, object]]) -> ToolCall | None:
    index = payload.get("index")
    if not isinstance(index, int):
        if not tool_blocks:
            return None
        raise ProviderError(STREAM_FORMAT_FAILED)
    block = tool_blocks.pop(index, None)
    if block is None:
        return None
    raw_arguments = block["partial_json"] or json.dumps(block["initial_input"])
    try:
        arguments = json.loads(str(raw_arguments))
    except json.JSONDecodeError:
        return ToolCall(
            str(block["id"]),
            str(block["name"]),
            {},
            argument_error="invalid_json",
        )
    if not isinstance(arguments, Mapping):
        return ToolCall(
            str(block["id"]),
            str(block["name"]),
            {},
            argument_error="invalid_json",
        )
    return ToolCall(str(block["id"]), str(block["name"]), dict(arguments))


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or status_code >= 500


def _safe_error_type(payload: object) -> ProviderErrorType:
    error = payload.get("error") if isinstance(payload, Mapping) else None
    value = error.get("type") if isinstance(error, Mapping) else None
    return value if value in PROVIDER_ERROR_TYPE_VALUES else "unknown_error"


def _safe_request_id(response: httpx.Response) -> str | None:
    value = response.headers.get("request-id") or response.headers.get("x-request-id")
    return normalize_provider_request_id(value)


def _stream_event_error(data: str, response: httpx.Response) -> ProviderError:
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        payload = None
    error_type = _safe_error_type(payload)
    return ProviderError(
        STREAM_REPORTED_ERROR,
        retryable=error_type in _RETRYABLE_ERROR_TYPES,
        error_type=error_type,
        failure_phase="stream_event",
        request_id=_safe_request_id(response),
    )


def _response_error(response: httpx.Response) -> ProviderError:
    category = "other"
    try:
        response.read()
        payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        payload = None
    error_type = _safe_error_type(payload)
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping):
        message = error.get("message")
        if error_type == "request_too_large" or (
            isinstance(message, str)
            and any(phrase in message.casefold() for phrase in _CONTEXT_OVERFLOW_PHRASES)
        ):
            category = "context_overflow"
    return ProviderError(
        RESPONSE_FAILED,
        retryable=_is_retryable_status(response.status_code),
        category=category,
        status_code=response.status_code,
        error_type=error_type,
        failure_phase="http_status",
        request_id=_safe_request_id(response),
    )


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RequestCancelled()
