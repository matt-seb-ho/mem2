"""MDL / parsimony scorer for concept memory reorganization.

Approximates description length as:

    L(memory) = sum over concepts of len(rendered_concept_text)
              + penalty * num_concepts   (per-concept encoding overhead)

Higher L = costlier memory. Reorg aims to reduce L by aggregating redundant
concepts into a single parent while keeping coverage.

This scorer offers two modes:
  - ``score(mem)`` → scalar description length
  - ``local_diff(mem, keep_names, merge_into)`` → predicted L change if the
    named concepts were replaced by a single aggregate. No LLM calls.

The scorer is purely textual: encoding cost is rendered-string length in
characters, which is a rough but deterministic proxy for token cost. Future
work could swap in a tokenizer-based cost.
"""
from __future__ import annotations

from dataclasses import dataclass

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory


@dataclass(frozen=True, slots=True)
class MDLScore:
    total: float
    num_concepts: int
    content_chars: int
    overhead: float


def concept_length(c: Concept) -> int:
    """Rendered character count — stand-in for token cost."""
    return len(c.to_string(include_description=True))


class MDLScorer:
    def __init__(self, *, per_concept_overhead: float = 32.0) -> None:
        self.per_concept_overhead = float(per_concept_overhead)

    def score(self, mem: ConceptMemory) -> MDLScore:
        content = sum(concept_length(c) for c in mem.concepts.values())
        n = len(mem.concepts)
        overhead = self.per_concept_overhead * n
        return MDLScore(
            total=content + overhead,
            num_concepts=n,
            content_chars=content,
            overhead=overhead,
        )

    def local_diff(
        self,
        mem: ConceptMemory,
        merge_names: list[str],
        *,
        aggregate_length_estimate: int | None = None,
    ) -> float:
        """Return L_new - L_current if the listed concepts are replaced by a
        single aggregate.

        If ``aggregate_length_estimate`` is None, the estimate is the longest
        member's rendered length — a conservative lower-bound on the aggregate.
        Negative return value means reorg would shrink the library.
        """
        if not merge_names:
            return 0.0
        current_content = 0
        lens = []
        for name in merge_names:
            c = mem.concepts.get(name)
            if c is None:
                continue
            l = concept_length(c)
            current_content += l
            lens.append(l)
        if not lens:
            return 0.0
        agg = aggregate_length_estimate if aggregate_length_estimate is not None else max(lens)
        removed_overhead = self.per_concept_overhead * len(lens)
        added_overhead = self.per_concept_overhead  # one aggregate replaces many
        return (agg - current_content) + (added_overhead - removed_overhead)

    def score_delta(self, before: ConceptMemory, after: ConceptMemory) -> float:
        """Full before/after diff. Positive means memory grew in cost."""
        return self.score(after).total - self.score(before).total
