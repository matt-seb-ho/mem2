"""RRMCInteractiveRetriever: multi-round probing with Coverage + convergence.

Axis C in the ablation plan. Compared against one-shot top-k retrievers.

Faithful simplification of ``workstation_00_RRMC`` for the Phase-1 axis-C
question ("does multi-round interactive retrieval help?"). The full RRMC
machinery (knowledge-graph priors, MI-estimator, structured probing) lives
elsewhere; porting it in total is out of scope for Phase-1. The core signal
— context division via coverage-gated iteration — is captured here.

Round structure:
  - Round 1: seed retrieval = top-k by co-activation degree.
  - Round t>1: pick concepts that maximize coverage (unseen concept kinds and
    unseen ``cues``), up to ``per_round_k``.
  - Stop when (a) coverage stops growing for ``convergence_patience`` rounds,
    (b) we reach ``max_rounds``, or (c) the candidate pool is exhausted.

This retriever is LLM-free at the retrieval step itself; it uses the graph
structure + concept metadata. Future RRMC variants can swap in the LLM-based
probing question generator; hook point is marked below.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


@dataclass(frozen=True, slots=True)
class CoverageState:
    kinds_seen: frozenset[str]
    cues_seen: frozenset[str]
    concepts_seen: frozenset[str]

    @property
    def score(self) -> int:
        return len(self.kinds_seen) + len(self.cues_seen)


class RRMCInteractiveRetriever:
    name = "rrmc_interactive"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        per_round_k: int = 3,
        max_rounds: int = 5,
        convergence_patience: int = 2,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
    ) -> None:
        self.top_k = int(top_k)
        self.per_round_k = int(per_round_k)
        self.max_rounds = int(max_rounds)
        self.convergence_patience = int(convergence_patience)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)

    # ----------------------------------------------------------------- #
    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        concept_mem = ConceptMemory.from_payload(memory.payload)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"reason": "empty_memory"},
            )

        graph = ConceptGraph.build_from_memory(concept_mem, min_co_overlap=1)

        # Round 1 — seed: top-k by co-activation degree
        degrees = [
            (graph.degree(n, kinds=["co_activation"]), n) for n in graph.nodes
        ]
        degrees.sort(reverse=True)
        ordered_by_degree = [n for _, n in degrees]
        seen: set[str] = set()
        rounds: list[list[str]] = []

        def bump(names: list[str]) -> list[str]:
            added = []
            for n in names:
                if n in seen or n not in concept_mem.concepts:
                    continue
                seen.add(n)
                added.append(n)
                if len(seen) >= self.top_k:
                    break
            return added

        seed = bump(ordered_by_degree[: self.per_round_k])
        if seed:
            rounds.append(seed)

        # Coverage-driven rounds
        prev_cov_score = self._coverage_score(concept_mem, seen)
        patience = 0
        while (
            len(seen) < self.top_k
            and len(rounds) < self.max_rounds
            and patience < self.convergence_patience
        ):
            candidate = self._next_round_candidates(concept_mem, graph, seen)
            added = bump(candidate)
            if not added:
                break
            rounds.append(added)
            cur_cov_score = self._coverage_score(concept_mem, seen)
            if cur_cov_score <= prev_cov_score:
                patience += 1
            else:
                patience = 0
            prev_cov_score = cur_cov_score

        names = list(seen)
        hint_text = concept_mem.to_string(
            concept_names=names,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            usage_threshold=self.usage_threshold,
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text or None,
            retrieved_items=[{"name": n} for n in names],
            metadata={
                "retriever": self.name,
                "scoring_mode": "rrmc_multi_round",
                "num_rounds": len(rounds),
                "per_round_counts": [len(r) for r in rounds],
                "rounds": rounds,
                "coverage_score": prev_cov_score,
                "converged": patience >= self.convergence_patience,
            },
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        return self.retrieve(ctx, memory, problem, previous_attempts)

    # ----------------------------------------------------------------- #
    def _coverage_score(
        self, mem: ConceptMemory, names: set[str]
    ) -> int:
        kinds = set()
        cues = set()
        for n in names:
            c = mem.concepts.get(n)
            if not c:
                continue
            kinds.add(c.kind)
            for cue in c.cues:
                cues.add(cue)
        return len(kinds) + len(cues)

    def _next_round_candidates(
        self,
        mem: ConceptMemory,
        graph: ConceptGraph,
        seen: set[str],
    ) -> list[str]:
        """Rank unseen concepts by (unseen-kinds-contributed, unseen-cues-contributed),
        breaking ties by graph degree.

        This is the "Coverage" signal — we prefer candidates that bring NEW
        concept kinds or NEW cues over ones that merely duplicate what's
        already in context.
        """
        kinds_seen: Counter[str] = Counter()
        cues_seen: set[str] = set()
        for n in seen:
            c = mem.concepts.get(n)
            if not c:
                continue
            kinds_seen[c.kind] += 1
            cues_seen.update(c.cues)

        scored: list[tuple[tuple[int, int, float], str]] = []
        for name, c in mem.concepts.items():
            if name in seen:
                continue
            new_kind = 1 if kinds_seen.get(c.kind, 0) == 0 else 0
            new_cues = sum(1 for cue in c.cues if cue not in cues_seen)
            degree = graph.degree(name, kinds=["co_activation"])
            scored.append(((new_kind, new_cues, degree), name))
        scored.sort(key=lambda r: r[0], reverse=True)
        return [name for _, name in scored[: self.per_round_k]]
