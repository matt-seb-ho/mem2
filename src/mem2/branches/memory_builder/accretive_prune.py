"""AccretivePruneMemoryBuilder — axis A.11 baseline.

Accretive ArcMemo-PS growth + periodic drop-lowest-score pruning. This is
the critical upper-bound baseline *without* reorg that stage-4b was missing:
reorg-on (A.1) must beat accretive-prune to justify the headline.

Mechanism (simple):
  - Inherit `arcmemo_ps` schema (270-concept compressed memory compatible).
  - Each call to `consolidate()`: if the memory has more concepts than
    `max_concepts`, drop the lowest-`used_in`-count concepts until we're
    at cap. Ties broken by name sort for determinism.
  - No LLM call, no MDL, no plateau trigger. Just capacity pressure → drop.

Paper reference: this is the "accretive with drop-lowest-score" policy
called out in `exploration_guide.md` axis A candidate list and
`06_ablation_plan.md` baseline row. No single canonical paper — it's the
straightforward capacity-pruned variant ArcMemo described in its runtime
paragraph.
"""
from __future__ import annotations

from dataclasses import asdict

from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.concepts.memory import ConceptMemory, ProblemSolution
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)


class AccretivePruneMemoryBuilder:
    """Wraps `ArcMemoPsMemoryBuilder` with a hard `max_concepts` cap enforced
    at `consolidate()` time, dropping the lowest-frequency concepts first."""

    name = "accretive_prune"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 200,
        prune_every_consolidate: bool = True,
        freeze_memory: bool = False,
    ) -> None:
        self.max_concepts = int(max_concepts)
        self.prune_every_consolidate = bool(prune_every_consolidate)
        self._frozen = bool(freeze_memory)
        self._inner = ArcMemoPsMemoryBuilder(
            seed_memory_file=seed_memory_file,
            seed_annotations_file=seed_annotations_file,
            domain=domain,
            max_concepts=0,  # cap enforced here, not in the inner builder
        )

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        return self._inner.initialize(ctx, problems)

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        if self._frozen:
            return memory
        # Delegate solution write to the inner PS builder — no concept-level
        # additions happen at update time (concept extraction is offline).
        solutions = memory.payload.get("solutions", {})
        for i, att in enumerate(attempts):
            is_correct = eval_records[i].is_correct if i < len(eval_records) else False
            if is_correct:
                solutions[att.problem_uid] = asdict(
                    ProblemSolution(
                        problem_id=att.problem_uid,
                        solution=(att.completion or "")[:2000],
                    )
                )
        memory.payload["solutions"] = solutions
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if self._frozen:
            return memory
        if not self.prune_every_consolidate:
            return memory
        mem = ConceptMemory.from_payload(memory.payload)
        if self.max_concepts <= 0 or len(mem.concepts) <= self.max_concepts:
            return memory
        # Rank concepts by (ascending) used_in count, break ties by name.
        # Drop until we're at cap.
        ranked = sorted(
            mem.concepts.values(),
            key=lambda c: (len(c.used_in or []), c.name),
        )
        to_drop = len(mem.concepts) - self.max_concepts
        dropped_names: list[str] = []
        for c in ranked[:to_drop]:
            del mem.concepts[c.name]
            cat = mem.categories.get(c.kind, [])
            if c.name in cat:
                cat.remove(c.name)
            dropped_names.append(c.name)
        # Stamp the prune event into metadata so the aggregator can surface
        # "how many concepts were pruned" per run.
        memory.payload = mem.to_payload()
        meta = memory.metadata
        meta["last_prune"] = {
            "concepts_before": len(mem.concepts) + len(dropped_names),
            "concepts_after": len(mem.concepts),
            "dropped_count": len(dropped_names),
            "dropped_sample": dropped_names[:10],
        }
        return memory
