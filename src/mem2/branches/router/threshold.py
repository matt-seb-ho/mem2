"""Rule-based composite routing gate.

Wraps the existing ``RetrievalRouter`` from ``mem2.retrieval.routers`` and
exposes it as a pipeline-level Router component.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext
from mem2.retrieval.routers import RetrievalRouter


class ThresholdRouter:
    name = "threshold"

    def __init__(
        self,
        strategy: str = "none",
        frequency_threshold: float = 0.5,
        max_hint_chars: int = 0,
        max_concept_count: int = 0,
        max_pre_filter_count: int = 0,
        concept_frequency_file: str = "",
    ):
        frequencies: dict[str, float] = {}
        if concept_frequency_file:
            import json
            from pathlib import Path

            frequencies = json.loads(Path(concept_frequency_file).read_text())

        self._router = RetrievalRouter(
            strategy=strategy,
            frequency_threshold=frequency_threshold,
            max_hint_chars=max_hint_chars,
            max_concept_count=max_concept_count,
            max_pre_filter_count=max_pre_filter_count,
            frequencies=frequencies,
        )

    async def route(
        self,
        *,
        ctx: RunContext,
        provider: object,
        problem: ProblemSpec,
        retrieval: RetrievalBundle,
    ) -> RetrievalBundle:
        md: dict[str, Any] = dict(retrieval.metadata) if retrieval.metadata else {}
        names: list[str] | None = md.get("selected_names")
        pre_filter_count: int = md.get("pre_filter_count", 0)

        decision = self._router.should_include(
            names, retrieval.hint_text, pre_filter_count=pre_filter_count
        )

        md["routing_included"] = decision.include
        md["routing_reasons"] = decision.reasons

        if decision.include:
            return replace(retrieval, metadata=md)
        return replace(retrieval, hint_text=None, metadata=md)
