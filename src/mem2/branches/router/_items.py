"""Shared helpers for per-item router implementations."""
from __future__ import annotations

from typing import Any

from mem2.core.entities import ProblemSpec, RetrievalBundle


def extract_problem_text(problem: ProblemSpec, domain: str) -> str:
    if domain == "math":
        return str(problem.metadata.get("problem_text", ""))
    if domain == "code":
        return str(problem.metadata.get("question_content", ""))
    return str(problem.train_pairs)


def split_concepts_from_hint(
    hint_text: str, concept_names: list[str]
) -> dict[str, str]:
    """Extract per-concept text blocks from rendered hint_text.

    Each concept block starts with ``- concept: {name}`` and extends until the
    next ``- concept:``, section header (``## ``), or structural line
    (``- lower usage``, ``- other concepts``).
    """
    blocks: dict[str, str] = {}
    for name in concept_names:
        marker = f"- concept: {name}\n"
        idx = hint_text.find(marker)
        if idx == -1:
            marker_end = f"- concept: {name}"
            idx = hint_text.find(marker_end)
            if idx == -1:
                continue
        start = idx
        search_from = start + len(f"- concept: {name}")
        end = len(hint_text)
        for boundary in ("- concept:", "## ", "### ", "- lower usage", "- other concepts"):
            pos = hint_text.find(boundary, search_from)
            if pos != -1 and pos < end:
                end = pos
        blocks[name] = hint_text[start:end].rstrip()
    return blocks


def extract_item_texts(
    items: list[dict[str, Any]], retrieval: RetrievalBundle
) -> list[tuple[int, str, str]]:
    """Extract (index, key, text) triples from retrieved items.

    Handles both oe_selector items (with ``hint`` field) and ps_selector
    items (with ``concept`` field, text extracted from hint_text).

    Returns empty list if items cannot be decomposed.
    """
    if not items:
        return []

    result: list[tuple[int, str, str]] = []

    if "hint" in items[0]:
        for i, item in enumerate(items):
            text = str(item.get("hint", "")).strip()
            if text:
                key = str(item.get("source_uid", i))
                result.append((i, key, text))
    elif "concept" in items[0] and retrieval.hint_text:
        names = [item["concept"] for item in items]
        blocks = split_concepts_from_hint(retrieval.hint_text, names)
        for i, item in enumerate(items):
            name = item["concept"]
            if name in blocks:
                result.append((i, name, blocks[name]))

    return result
