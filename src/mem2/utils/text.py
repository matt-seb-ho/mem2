from __future__ import annotations

import ast
import re
from typing import Any


_LIST_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def extract_first_grid(text: str) -> list[list[Any]] | None:
    """Best-effort parser for first list-of-lists grid in model completion text."""
    match = _LIST_PATTERN.search(text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    if not parsed:
        return None
    if not all(isinstance(row, list) for row in parsed):
        return None
    return parsed
