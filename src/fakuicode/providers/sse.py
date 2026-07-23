"""Minimal Server-Sent Events framing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from fakuicode.errors import ProviderError


@dataclass(frozen=True)
class SSEEvent:
    name: str
    data: str


def parse_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    """Yield complete SSE frames from decoded response lines."""
    event_name = "message"
    data_lines: list[str] = []
    for line in lines:
        if not line:
            if data_lines:
                yield SSEEvent(event_name, "\n".join(data_lines))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.removeprefix(" ")
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        raise ProviderError("Provider stream ended with an unterminated event.")
