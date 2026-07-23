"""Shared JSON Pointer parsing and resolution for Hook payloads."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


class JsonPointerError(ValueError):
    """Raised when a Hook field is not a supported non-root JSON Pointer."""


def parse_pointer(value: str) -> tuple[str, ...]:
    if not value.startswith("/") or value == "/":
        raise JsonPointerError("field must be a non-root JSON Pointer")
    components: list[str] = []
    for raw in value[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw):
            raise JsonPointerError("JSON Pointer escape is invalid")
        components.append(raw.replace("~1", "/").replace("~0", "~"))
    return tuple(components)


def resolve_pointer(payload: object, path: tuple[str, ...]) -> tuple[bool, Any]:
    current = payload
    for component in path:
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdecimal() and int(component) < len(current):
            current = current[int(component)]
        else:
            return False, None
    return True, current
