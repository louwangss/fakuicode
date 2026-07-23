from __future__ import annotations

import pytest


def test_parses_named_sse_events_in_order() -> None:
    from fakuicode.providers.sse import parse_sse

    events = list(parse_sse(["event: update", 'data: {"value": 1}', "", "data: [DONE]", ""]))

    assert [(event.name, event.data) for event in events] == [("update", '{"value": 1}'), ("message", "[DONE]")]


def test_rejects_unterminated_event() -> None:
    from fakuicode.errors import ProviderError
    from fakuicode.providers.sse import parse_sse

    with pytest.raises(ProviderError, match="unterminated"):
        list(parse_sse(['data: {"value": 1}']))
