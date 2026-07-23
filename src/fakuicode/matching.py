"""Shared exact/glob matching primitives for policy-like configuration."""

from __future__ import annotations

import re


class GlobSyntaxError(ValueError):
    pass


def compile_glob(pattern: str) -> tuple[re.Pattern[str], bool]:
    """Compile FakuiCode's full-string glob grammar and report whether it is exact."""

    if not pattern:
        raise GlobSyntaxError("glob patterns must not be empty")
    pieces: list[str] = []
    exact = True
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            index += 1
            if index >= len(pattern):
                raise GlobSyntaxError("glob patterns must not end with an escape")
            escaped = pattern[index]
            if escaped not in {"\\", "*", "?", "[", "]"}:
                raise GlobSyntaxError("glob escapes may only quote metacharacters")
            pieces.append(re.escape(escaped))
        elif character == "*":
            pieces.append(".*")
            exact = False
        elif character == "?":
            pieces.append(".")
            exact = False
        elif character == "[":
            closing = pattern.find("]", index + 1)
            if closing < 0:
                raise GlobSyntaxError("glob character classes must be closed")
            content = pattern[index + 1 : closing]
            if not content or "\\" in content or "[" in content:
                raise GlobSyntaxError("glob character class is malformed")
            if content[0] in {"!", "^"}:
                prefix = "^"
                content = content[1:]
                if not content:
                    raise GlobSyntaxError("glob character classes must not be empty")
            else:
                prefix = ""
            pieces.append("[" + prefix + re.escape(content).replace(r"\-", "-") + "]")
            exact = False
            index = closing
        elif character == "]":
            raise GlobSyntaxError("glob contains an unmatched closing bracket")
        else:
            pieces.append(re.escape(character))
        index += 1
    return re.compile("".join(pieces)), exact
