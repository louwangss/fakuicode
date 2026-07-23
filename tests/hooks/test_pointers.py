from __future__ import annotations

import pytest

from fakuicode.hooks.pointers import JsonPointerError, parse_pointer, resolve_pointer


def test_json_pointer_resolves_nested_mappings_lists_and_escaped_components() -> None:
    payload = {
        "tool": {
            "arguments": {
                "items": [{"a/b": {"~name": "matched"}}],
            }
        }
    }

    path = parse_pointer("/tool/arguments/items/0/a~1b/~0name")

    assert path == ("tool", "arguments", "items", "0", "a/b", "~name")
    assert resolve_pointer(payload, path) == (True, "matched")


@pytest.mark.parametrize("pointer", ["", "/", "tool/name", "/tool/~2name", "/tool/name~"])
def test_json_pointer_rejects_root_relative_and_invalid_escapes(pointer: str) -> None:
    with pytest.raises(JsonPointerError):
        parse_pointer(pointer)


def test_json_pointer_reports_missing_or_out_of_range_values_without_matching() -> None:
    payload = {"tool": {"arguments": [{"path": "value"}]}}

    assert resolve_pointer(payload, parse_pointer("/tool/missing")) == (False, None)
    assert resolve_pointer(payload, parse_pointer("/tool/arguments/2")) == (False, None)
