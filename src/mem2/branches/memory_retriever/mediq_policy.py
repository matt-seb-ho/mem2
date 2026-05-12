"""MediQ abstention-gated retriever — axis C.4.

Port of the abstention-gated interactive pattern from MediQ (Li et al., NeurIPS'24).

Paper: literature/2406.00922.pdf
Repo:  third_party/mediq/ (entry: src/expert.py::Expert + expert_functions.py::fixed_abstention_decision)

Specifically ported:
    - The multi-round "ask-or-commit" scheduling pattern where each round
      grows the context and an abstention gate decides whether to continue
      accumulating or commit to the current set.

Deliberate simplifications:
    - LLM-based confidence elicitation → rolling-window **coverage**
      improvement. If the last `window` rounds have not increased
      (unique_kinds + unique_cues) by at least `abstention_threshold`,
      abstain. Keeps retrieval deterministic-on-seed + LLM-free.
    - No patient/expert role separation.
    - No LLM confidence logprobs (paper's implicit-abstention strategy).

Distinct from `rrmc_interactive`:
    - RRMC uses "patience" (symmetric coverage patience).
    - MediQ here uses an explicit abstention threshold — tighter control
      over when to commit.
"""
from __future__ import annotations

from collections import Counter

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


class MediQPolicyRetriever:
    """Abstention-gated multi-round concept retrieval.

    Round 1 = seed with top-`per_round_k` concepts by co-activation degree.
    Round r>1 = next `per_round_k` concepts that bring new kind/cue coverage.
    After each round, compute coverage gain over the prior `window` rounds.
    Abstain when gain < `abstention_threshold` (commit to current set).
    """

    name = "mediq_policy"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        per_round_k: int = 2,
        max_rounds: int = 5,
        abstention_threshold: int = 1,
        window: int = 2,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
    ) -> None:
        self.top_k = int(top_k)
        self.per_round_k = int(per_round_k)
        self.max_rounds = int(max_rounds)
        self.abstention_threshold = int(abstention_threshold)
        self.window = int(window)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        mem = ConceptMemory.from_payload(memory.payload)
        if not mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid, hint_text=None, retrieved_items=[],
                metadata={"retriever": self.name, "reason": "empty_memory"},
            )

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        # Seed order: descending co-activation degree.
        degree_ranked = sorted(
            mem.concepts.keys(),
            key=lambda n: graph.degree(n, kinds=["co_activation"]),
            reverse=True,
        )

        seen: set[str] = set()
        rounds_log: list[dict] = []
        coverage_history: list[int] = []

        def compute_coverage(names: set[str]) -> int:
            kinds_seen: set[str] = set()
            cues_seen: set[str] = set()
            for n in names:
                c = mem.concepts.get(n)
                if not c:
                    continue
                kinds_seen.add(c.kind)
                for cue in (c.cues or []):
                    cues_seen.add(cue)
            return len(kinds_seen) + len(cues_seen)

        def pick_next_round() -> list[str]:
            """Pick `per_round_k` concepts ranked by (new_kind, new_cues, degree)."""
            kinds_seen: Counter[str] = Counter()
            cues_seen: set[str] = set()
            for n in seen:
                c = mem.concepts.get(n)
                if not c:
                    continue
                kinds_seen[c.kind] += 1
                cues_seen.update(c.cues or [])
            scored: list[tuple[tuple[int, int, float], str]] = []
            for name, c in mem.concepts.items():
                if name in seen:
                    continue
                new_kind = 1 if kinds_seen.get(c.kind, 0) == 0 else 0
                new_cues = sum(1 for cue in (c.cues or []) if cue not in cues_seen)
                degree = graph.degree(name, kinds=["co_activation"])
                scored.append(((new_kind, new_cues, degree), name))
            scored.sort(key=lambda r: r[0], reverse=True)
            return [name for _, name in scored[: self.per_round_k]]

        # Seed round (round 0)
        for name in degree_ranked[: self.per_round_k]:
            if name in mem.concepts:
                seen.add(name)
        rounds_log.append({"round": 0, "added": list(seen)[:], "coverage": compute_coverage(seen)})
        coverage_history.append(compute_coverage(seen))

        abstained = False
        abstain_reason = ""
        for r in range(1, self.max_rounds + 1):
            if len(seen) >= self.top_k:
                abstain_reason = f"reached top_k cap ({self.top_k})"
                break
            candidates = pick_next_round()
            if not candidates:
                abstain_reason = "no more candidates"
                break
            for name in candidates:
                if len(seen) >= self.top_k:
                    break
                seen.add(name)
            cov = compute_coverage(seen)
            rounds_log.append({"round": r, "added": candidates, "coverage": cov})
            coverage_history.append(cov)
            # Abstention check: rolling-window coverage gain
            if len(coverage_history) > self.window:
                gain = coverage_history[-1] - coverage_history[-1 - self.window]
                if gain < self.abstention_threshold:
                    abstained = True
                    abstain_reason = (
                        f"abstained at round {r}: coverage gain over last {self.window} rounds "
                        f"({gain}) < threshold ({self.abstention_threshold})"
                    )
                    break

        selected = sorted(seen)
        hint = mem.to_string(
            concept_names=selected,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            usage_threshold=self.usage_threshold,
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in selected],
            metadata={
                "retriever": self.name,
                "scoring_mode": "mediq_policy",
                "top_k": self.top_k,
                "num_rounds": len(rounds_log),
                "rounds": rounds_log,
                "abstained": abstained,
                "abstain_reason": abstain_reason or "max_rounds",
                "coverage_history": coverage_history,
                "num_selected": len(selected),
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
