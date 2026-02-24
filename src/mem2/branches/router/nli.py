"""Cross-encoder NLI routing gate — per-item filtering.

Scores entailment between problem text and each retrieved hint item
individually.  Items below the threshold are dropped; surviving items are
reassembled into a new hint_text.  This turns the router from a binary
keep/discard gate into a "which hints to keep" filter.

Per-item text extraction:
- oe_selector items: uses the ``hint`` field directly.
- ps_selector items: splits ``hint_text`` on ``- concept: {name}`` boundaries
  to recover per-concept blocks.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext

from ._items import extract_item_texts, extract_problem_text


def _softmax(logits: Any) -> Any:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


class NliRouter:
    name = "nli"

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.5,
        domain: str = "arc",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.entailment_threshold = float(entailment_threshold)
        self.domain = domain
        self.device = device
        self._cross_encoder: Any = None

    def _get_cross_encoder(self) -> Any:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name, device=self.device)
        return self._cross_encoder

    async def route(
        self,
        *,
        ctx: RunContext,
        provider: object,
        problem: ProblemSpec,
        retrieval: RetrievalBundle,
    ) -> RetrievalBundle:
        if not retrieval.hint_text:
            return retrieval

        items = retrieval.retrieved_items
        if not items:
            return retrieval

        problem_text = extract_problem_text(problem, self.domain)
        ce = self._get_cross_encoder()

        item_texts = extract_item_texts(items, retrieval)
        if not item_texts:
            return retrieval

        # ── Score each item ────────────────────────────────────────────
        pairs = [(problem_text, text) for _, _, text in item_texts]
        raw_logits = ce.predict(pairs)

        scores: dict[str, float] = {}
        surviving_indices: list[int] = []
        surviving_texts: list[str] = []

        for (idx, key, text), logit_row in zip(item_texts, raw_logits):
            probs = _softmax(np.array(logit_row))
            score = float(probs[2])  # entailment
            scores[key] = score
            if score >= self.entailment_threshold:
                surviving_indices.append(idx)
                surviving_texts.append(text)

        # ── Build filtered bundle ──────────────────────────────────────
        md: dict[str, Any] = dict(retrieval.metadata) if retrieval.metadata else {}
        md["routing_nli_scores"] = scores
        md["routing_included_items"] = [
            item_texts[j][1] for j in range(len(item_texts))
            if item_texts[j][0] in surviving_indices
        ]
        md["routing_excluded_items"] = [
            item_texts[j][1] for j in range(len(item_texts))
            if item_texts[j][0] not in surviving_indices
        ]
        md["routing_included"] = len(surviving_indices) > 0

        if not surviving_indices:
            return replace(
                retrieval,
                hint_text=None,
                retrieved_items=[],
                metadata=md,
            )

        surviving_items = [retrieval.retrieved_items[i] for i in surviving_indices]
        hint_text = "\n".join(surviving_texts)

        # Update selected_names in metadata if present (ps_selector)
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
