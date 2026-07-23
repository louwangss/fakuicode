from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from threading import Event, RLock
from urllib.parse import urlparse

import httpx

from fakuicode.errors import ProviderError, RequestCancelled
from fakuicode.models import (
    AgentMessage,
    AgentStreamEvent,
    Message,
    ProviderConfig,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from fakuicode.providers.base import AGENT_SYSTEM_PROMPT, AgentRequest, ProviderCapabilities
from fakuicode.providers.dsml import ToolMarkupAccumulator
from fakuicode.providers.sse import parse_sse


RESPONSE_FAILED = "OpenAI response failed."
REQUEST_FAILED = "OpenAI request failed."
STREAM_FORMAT_FAILED = "OpenAI stream format failed."
STREAM_REPORTED_ERROR = "OpenAI stream reported an error."
_SUPPORTED_CONTEXT_ERROR_CODES = {"context_length_exceeded", "context_window_exceeded"}
_CONTEXT_OVERFLOW_PHRASES = (
    "maximum context length",
    "context length exceeded",
    "context window",
    "too many tokens",
)


class OpenAIProvider:
    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None) -> None:
        self.config, self.client = config, client or httpx.Client(timeout=60.0)
        self._active_response: httpx.Response | None = None
        self._response_lock = RLock()

    @property
    def capabilities(self) -> ProviderCapabilities:
        parsed = urlparse(self.config.base_url)
        is_official = parsed.scheme == "https" and parsed.hostname == "api.openai.com"
        return ProviderCapabilities(supports_output_token_limit=is_official)

    def cancel(self) -> None:
        """Interrupt the active HTTP stream, if this provider owns one."""
        with self._response_lock:
            response = self._active_response
        if response is not None:
            response.close()

    def stream_chat(self, messages: Sequence[Message]) -> Iterator[StreamEvent]:
        body = {
            "model": self.config.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": True,
        }
        try:
            with self.client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=body,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    raise _response_error(response) from error

                try:
                    for event in parse_sse(response.iter_lines()):
                        if event.name == "error":
                            raise ProviderError(STREAM_REPORTED_ERROR)
                        if event.data == "[DONE]":
                            yield StreamEvent("completed")
                            return
                        payload = _parse_payload(event.data)
                        if "error" in payload:
                            raise ProviderError(STREAM_REPORTED_ERROR)
                        text = _extract_text(payload)
                        if text:
                            yield StreamEvent("text_delta", text)
                except ProviderError as error:
                    if str(error) in {STREAM_REPORTED_ERROR, STREAM_FORMAT_FAILED}:
                        raise
                    raise ProviderError(STREAM_FORMAT_FAILED) from error
        except ProviderError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise ProviderError(REQUEST_FAILED, retryable=True) from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError(REQUEST_FAILED) from error
        raise ProviderError("OpenAI stream ended before completion.")

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
        body = {
            "model": self.config.model,
            "messages": _openai_messages(agent_request),
            "stream": True,
        }
        if agent_request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    },
                }
                for tool in agent_request.tools
            ]
        if (
            agent_request.output_token_limit is not None
            and self.capabilities.supports_output_token_limit
        ):
            body["max_completion_tokens"] = agent_request.output_token_limit
        tool_blocks: dict[int, dict[str, str]] = {}
        tool_markup = ToolMarkupAccumulator()
        try:
            with self.client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
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
                            raise ProviderError(STREAM_REPORTED_ERROR)
                        if event.data == "[DONE]":
                            visible_text, raw_calls = tool_markup.finish()
                            for chunk in visible_text:
                                yield AgentStreamEvent("text_delta", chunk)
                            if raw_calls is not None:
                                for call in raw_calls:
                                    yield AgentStreamEvent("tool_call", tool_call=call)
                            for call in _finish_tool_blocks(tool_blocks):
                                yield AgentStreamEvent("tool_call", tool_call=call)
                            yield AgentStreamEvent("completed")
                            return
                        payload = _parse_payload(event.data)
                        if "error" in payload:
                            raise ProviderError(STREAM_REPORTED_ERROR)
                        usage = _openai_usage(payload)
                        if usage is not None:
                            yield AgentStreamEvent("usage", usage=usage)
                        text = _extract_text(payload)
                        if text:
                            for chunk in tool_markup.append(text):
                                yield AgentStreamEvent("text_delta", chunk)
                        _collect_tool_blocks(payload, tool_blocks)
                except ProviderError as error:
                    if str(error) in {STREAM_REPORTED_ERROR, STREAM_FORMAT_FAILED}:
                        raise
                    raise ProviderError(STREAM_FORMAT_FAILED) from error
                finally:
                    with self._response_lock:
                        if self._active_response is response:
                            self._active_response = None
        except ProviderError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as error:
            _raise_if_cancelled(agent_request.cancel_event)
            raise ProviderError(REQUEST_FAILED, retryable=True) from error
        except (httpx.HTTPError, ValueError) as error:
            _raise_if_cancelled(agent_request.cancel_event)
            raise ProviderError(REQUEST_FAILED) from error
        raise ProviderError("OpenAI stream ended before completion.")


def _parse_payload(data: str) -> Mapping[str, object]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise ProviderError(STREAM_FORMAT_FAILED) from error
    if not isinstance(payload, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    return payload


def _system_prompt(instruction: str) -> str:
    return AGENT_SYSTEM_PROMPT if not instruction else f"{AGENT_SYSTEM_PROMPT}\n\n{instruction}"


def _openai_messages(request: AgentRequest) -> list[dict[str, object]]:
    """Keep the cacheable prefix first for OpenAI-compatible chat endpoints."""

    messages: list[dict[str, object]] = [{"role": "system", "content": request.system_prompt}]
    if request.system_supplement:
        messages.append({"role": "system", "content": request.system_supplement})
    messages.extend(_agent_messages(request.messages))
    return messages


def _openai_usage(payload: Mapping[str, object]) -> TokenUsage | None:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        return None
    input_tokens = _optional_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")))
    output_tokens = _optional_int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens")))
    details = raw_usage.get("prompt_tokens_details")
    cached_tokens = _optional_int(details.get("cached_tokens")) if isinstance(details, Mapping) else None
    if input_tokens is None and output_tokens is None and cached_tokens is None:
        return None
    return TokenUsage(
        input_tokens,
        output_tokens,
        cache_read_tokens=cached_tokens,
        context_input_tokens=input_tokens,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _extract_text(payload: Mapping[str, object]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise ProviderError(STREAM_FORMAT_FAILED)
    if not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    delta = first_choice.get("delta")
    if not isinstance(delta, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    content = delta.get("content")
    if content is not None and not isinstance(content, str):
        raise ProviderError(STREAM_FORMAT_FAILED)
    return content


def _agent_messages(messages: Sequence[AgentMessage]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        if message.role == "assistant":
            assistant: dict[str, object] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                    }
                    for call in message.tool_calls
                ]
            result.append(assistant)
            continue

        if message.content:
            result.append({"role": "user", "content": message.content})
        for tool_result in message.tool_results:
            result.append(
                {"role": "tool", "tool_call_id": tool_result.call_id, "content": tool_result.to_model_content()}
            )
    return result


def _collect_tool_blocks(payload: Mapping[str, object], tool_blocks: dict[int, dict[str, str]]) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        raise ProviderError(STREAM_FORMAT_FAILED)
    fragments = delta.get("tool_calls")
    if fragments is None:
        return
    if not isinstance(fragments, list):
        raise ProviderError(STREAM_FORMAT_FAILED)
    for fragment in fragments:
        if not isinstance(fragment, Mapping) or not isinstance(fragment.get("index"), int):
            raise ProviderError(STREAM_FORMAT_FAILED)
        index = fragment["index"]
        block = tool_blocks.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if isinstance(fragment.get("id"), str):
            block["id"] = fragment["id"]
        function = fragment.get("function")
        if function is None:
            continue
        if not isinstance(function, Mapping):
            raise ProviderError(STREAM_FORMAT_FAILED)
        if isinstance(function.get("name"), str):
            block["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            block["arguments"] += function["arguments"]


def _finish_tool_blocks(tool_blocks: dict[int, dict[str, str]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for _, block in sorted(tool_blocks.items()):
        if not block["id"] or not block["name"]:
            raise ProviderError(STREAM_FORMAT_FAILED)
        try:
            arguments = json.loads(block["arguments"])
        except json.JSONDecodeError as error:
            raise ProviderError(STREAM_FORMAT_FAILED) from error
        if not isinstance(arguments, Mapping):
            raise ProviderError(STREAM_FORMAT_FAILED)
        calls.append(ToolCall(block["id"], block["name"], dict(arguments)))
    return calls


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or status_code >= 500


def _response_error(response: httpx.Response) -> ProviderError:
    category = "other"
    try:
        response.read()
        payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        payload = None
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        if code in _SUPPORTED_CONTEXT_ERROR_CODES or (
            isinstance(message, str)
            and any(phrase in message.casefold() for phrase in _CONTEXT_OVERFLOW_PHRASES)
        ):
            category = "context_overflow"
    return ProviderError(
        RESPONSE_FAILED,
        retryable=_is_retryable_status(response.status_code),
        category=category,
    )


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RequestCancelled()
