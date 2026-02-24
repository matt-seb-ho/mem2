"""LLM-based semantic routing — per-item filtering in a single call.

Presents all retrieved hints as a numbered list and asks the LLM which ones
are relevant.  One LLM call per problem.  Fail-open: on parse failure, keeps
all items.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext

from ._items import extract_item_texts, extract_problem_text

_DEFAULT_GEN_CFG: dict[str, Any] = {"n": 1, "temperature": 0, "max_tokens": 256}

_PROMPT_TEMPLATE = """\
Problem:
{problem_text}

Proposed hints:
{numbered_hints}

Which of the above hints are directly relevant to solving this problem?
Return ONLY the numbers of the relevant hints as a comma-separated list, e.g. "1, 3, 5".
If none are relevant, return "NONE"."""

_NUMBER_RE = re.compile(r"\d+")


class LlmRouter:
    name = "llm"

    def __init__(
        self,
        model: str = "",
        gen_cfg: dict[str, Any] | None = None,
        domain: str = "arc",
    ):
        self.model = model
        self.gen_cfg = gen_cfg or dict(_DEFAULT_GEN_CFG)
        self.domain = domain

    async def route(
        self,
        *,
        ctx: RunContext,
        provider: Any,
        problem: ProblemSpec,
        retrieval: RetrievalBundle,
    ) -> RetrievalBundle:
        if not retrieval.hint_text:
            return retrieval

        items = retrieval.retrieved_items
        if not items:
            return retrieval

        item_texts = extract_item_texts(items, retrieval)
        if not item_texts:
            return retrieval

        # ── Build numbered list ────────────────────────────────────────
        problem_text = extract_problem_text(problem, self.domain)
        numbered_lines: list[str] = []
        for num, (_, key, text) in enumerate(item_texts, start=1):
            # Truncate individual items to keep prompt reasonable
            truncated = text[:1000] if len(text) > 1000 else text
            numbered_lines.append(f"{num}. [{key}] {truncated}")

        prompt = _PROMPT_TEMPLATE.format(
            problem_text=problem_text[:2000],
            numbered_hints="\n".join(numbered_lines),
        )

        # ── Single LLM call ───────────────────────────────────────────
        completions = await provider.async_generate(prompt, self.model, self.gen_cfg)
        completion = completions[0] if completions else ""

        # ── Parse response ─────────────────────────────────────────────
        selected_numbers = _parse_selection(completion, len(item_texts))

        md: dict[str, Any] = dict(retrieval.metadata) if retrieval.metadata else {}
        md["routing_model"] = self.model
        md["routing_prompt"] = prompt
        md["routing_completion"] = completion

        # Fail-open: if we can't parse or got no valid numbers, keep all
        if selected_numbers is None:
            md["routing_included"] = True
            md["routing_parse_failure"] = True
            return replace(retrieval, metadata=md)

        # "NONE" → drop everything
        if len(selected_numbers) == 0:
            md["routing_included"] = False
            md["routing_included_items"] = []
            md["routing_excluded_items"] = [key for _, key, _ in item_texts]
            return replace(retrieval, hint_text=None, retrieved_items=[], metadata=md)

        # ── Filter to selected items ──────────────────────────────────
        # selected_numbers are 1-indexed into item_texts
        selected_set = set(selected_numbers)
        surviving_indices: list[int] = []
        surviving_texts: list[str] = []
        included_keys: list[str] = []
        excluded_keys: list[str] = []

        for num, (idx, key, text) in enumerate(item_texts, start=1):
            if num in selected_set:
                surviving_indices.append(idx)
                surviving_texts.append(text)
                included_keys.append(key)
            else:
                excluded_keys.append(key)

        md["routing_included"] = len(surviving_indices) > 0
        md["routing_included_items"] = included_keys
        md["routing_excluded_items"] = excluded_keys

        if not surviving_indices:
            return replace(retrieval, hint_text=None, retrieved_items=[], metadata=md)

        surviving_items = [retrieval.retrieved_items[i] for i in surviving_indices]
        hint_text = "\n".join(surviving_texts)

        # Update selected_names if present (ps_selector)
        if "selected_names" in md:
            surviving_names_set = {
                retrieval.retrieved_items[i].get("concept")
                for i in surviving_indices
                if "concept" in retrieval.retrieved_items[i]
            }
            if surviving_names_set:
                md["selected_names"] = [
                    n for n in md["selected_names"] if n in surviving_names_set
                ]
                md["selected_count"] = len(md["selected_names"])

        return replace(
            retrieval,
            hint_text=hint_text,
            retrieved_items=surviving_items,
            metadata=md,
        )


def _parse_selection(completion: str, num_items: int) -> list[int] | None:
    """Parse LLM response into list of 1-indexed item numbers.

    Returns:
        list[int]: Selected item numbers (may be empty for "NONE").
        None: Parse failure (fail-open signal).
    """
    text = completion.strip().upper()

    if "NONE" in text:
        return []

    numbers = [int(n) for n in _NUMBER_RE.findall(text)]
    # Filter to valid range
    valid = [n for n in numbers if 1 <= n <= num_items]

    if not valid and numbers:
        # Had numbers but none in valid range — parse failure
        return None
    if not valid:
        # No numbers at all — parse failure
        return None

    return valid
