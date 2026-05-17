"""Stitch-style compression — axis A.3.

Port of the top-down, frequency-ranked compression from Stitch (Bowers et al., POPL'23).

Paper: literature/2211.16605.pdf
Repo:  third_party/stitch/  (core algorithm is Rust; we re-implement in Python).

Specifically ported:
    - The "compress most-frequent patterns first" top-down strategy. Stitch
      is a faster DreamCoder that beats beam search via anti-unification +
      frequency priors.

Deliberate simplifications:
    - Anti-unification over program syntax trees (paper) → frequency-first
      ranking of single lines across the concept library. No syntax tree
      because concept memory doesn't have one; lines are the unit.
    - Stitch's Rust compressor → Python greedy loop.

Distinction from DreamCoder (A.2):
    - DreamCoder port: bottom-up pairwise fragment finder. Fragments can
      contain multiple lines (a *set* of shared lines).
    - Stitch port: top-down frequency-ranked fragment finder. Each fragment
      is a SINGLE high-frequency line, absorbed by every concept holding it.
    - Different granularity + different search direction.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from mem2.branches.feedback_engine.plateau_trigger import (
    detect_plateau,
    detect_plateau_every_k,
)
from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory, ProblemSolution
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)
from mem2.scoring.mdl import MDLScorer


class StitchReorgBuilder:
    """Frequency-ranked single-line compression à la Stitch."""

    name = "reorg_stitch"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 0,
        trigger: str = "every_k",
        plateau_window: int = 10,
        plateau_min_delta: float = 0.01,
        every_k: int = 20,
        scope: str = "accretive",
        min_line_frequency: int = 3,
        max_fragments_per_round: int = 10,
        mdl_per_concept_overhead: float = 32.0,
        freeze_memory: bool = False,
    ):
        self.seed_memory_file = seed_memory_file
        self.seed_annotations_file = seed_annotations_file
        self.domain = domain
        self.max_concepts = int(max_concepts)
        self._frozen = bool(freeze_memory)
        self.trigger = trigger
        self.plateau_window = int(plateau_window)
        self.plateau_min_delta = float(plateau_min_delta)
        self.every_k = int(every_k)
        self.scope = scope
        self.min_line_frequency = int(min_line_frequency)
        self.max_fragments_per_round = int(max_fragments_per_round)
        self.scorer = MDLScorer(per_concept_overhead=mdl_per_concept_overhead)

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
        base = ArcMemoPsMemoryBuilder(
            seed_memory_file=self.seed_memory_file,
            seed_annotations_file=self.seed_annotations_file,
            domain=self.domain,
            max_concepts=self.max_concepts,
        ).initialize(ctx, problems)
        base.payload.setdefault("stitch_reorg", {
            "history": [], "step": 0, "outcomes": [],
            "trigger": self.trigger, "scope": self.scope,
        })
        return base

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
        mem = ConceptMemory.from_payload(memory.payload)
        state = memory.payload.setdefault("stitch_reorg", {
            "history": [], "step": 0, "outcomes": [],
            "trigger": self.trigger, "scope": self.scope,
        })
        for i, att in enumerate(attempts):
            is_correct = eval_records[i].is_correct if i < len(eval_records) else False
            if is_correct:
                mem.solutions[att.problem_uid] = ProblemSolution(
                    problem_id=att.problem_uid,
                    solution=(att.completion or "")[:2000],
                )
            state["step"] += 1
            state["outcomes"].append(1.0 if is_correct else 0.0)
        memory.payload = mem.to_payload()
        memory.payload["stitch_reorg"] = state
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if self._frozen:
            return memory
        state = memory.payload.get("stitch_reorg")
        if not state or not self._should_trigger(state):
            return memory
        mem = ConceptMemory.from_payload(memory.payload)

        # Frequency-rank lines across the library
        cue_counts: Counter[str] = Counter()
        impl_counts: Counter[str] = Counter()
        cue_holders: dict[str, list[str]] = {}
        impl_holders: dict[str, list[str]] = {}
        for name, c in mem.concepts.items():
            for ln in (c.cues or []):
                cue_counts[ln] += 1
                cue_holders.setdefault(ln, []).append(name)
            for ln in (c.implementation or []):
                impl_counts[ln] += 1
                impl_holders.setdefault(ln, []).append(name)

        candidates: list[dict[str, Any]] = []
        for ln, count in cue_counts.most_common():
            if count < self.min_line_frequency:
                break
            candidates.append({
                "kind": "cue",
                "line": ln,
                "count": count,
                "children": list(cue_holders[ln]),
            })
        for ln, count in impl_counts.most_common():
            if count < self.min_line_frequency:
                break
            candidates.append({
                "kind": "impl",
                "line": ln,
                "count": count,
                "children": list(impl_holders[ln]),
            })
        candidates.sort(key=lambda c: c["count"], reverse=True)
        candidates = candidates[: self.max_fragments_per_round]

        if not candidates:
            state.setdefault("history", []).append({
                "step": state["step"],
                "action": "skipped",
                "reason": f"no lines above min_line_frequency={self.min_line_frequency}",
            })
            return memory

        # Greedy MDL-gated commit
        mdl_before = self.scorer.score(mem).total
        committed: list[dict[str, Any]] = []
        for cand in candidates:
            sim = ConceptMemory.from_payload(mem.to_payload())
            frag_name = f"stitch_{cand['kind']}__{cand['line'][:32]}".replace(" ", "_")
            # Sanitize to avoid catastrophic names
            frag_name = "".join(
                ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in frag_name
            )[:96]
            if frag_name in sim.concepts:
                continue
            fragment = Concept(
                name=frag_name,
                kind="routine",
                routine_subtype="stitch_fragment",
                description=f"Stitch fragment: {cand['kind']} line shared by {cand['count']} concepts",
                cues=[cand["line"]] if cand["kind"] == "cue" else [],
                implementation=[cand["line"]] if cand["kind"] == "impl" else [],
                used_in=sorted({
                    pid for n in cand["children"]
                    for pid in (mem.concepts[n].used_in or [])
                }),
            )
            sim.concepts[frag_name] = fragment
            sim.categories[fragment.kind].append(frag_name)
            if self.scope == "global_rebuild":
                for child in cand["children"]:
                    c = sim.concepts.get(child)
                    if c is None:
                        continue
                    if cand["kind"] == "cue":
                        c.cues = [ln for ln in (c.cues or []) if ln != cand["line"]]
                    else:
                        c.implementation = [ln for ln in (c.implementation or []) if ln != cand["line"]]
            sim_mdl = self.scorer.score(sim).total
            if sim_mdl >= mdl_before:
                continue
            # commit
            mem.concepts[frag_name] = fragment
            if frag_name not in mem.categories[fragment.kind]:
                mem.categories[fragment.kind].append(frag_name)
            if self.scope == "global_rebuild":
                for child in cand["children"]:
                    c = mem.concepts.get(child)
                    if c is None:
                        continue
                    if cand["kind"] == "cue":
                        c.cues = [ln for ln in (c.cues or []) if ln != cand["line"]]
                    else:
                        c.implementation = [ln for ln in (c.implementation or []) if ln != cand["line"]]
            committed.append({
                "fragment": frag_name,
                "kind": cand["kind"],
                "count": cand["count"],
                "children": cand["children"],
                "mdl_delta": sim_mdl - mdl_before,
            })
            mdl_before = sim_mdl

        if not committed:
            state.setdefault("history", []).append({
                "step": state["step"],
                "action": "skipped",
                "reason": "all candidates failed MDL gate",
            })
            return memory

        mdl_after = self.scorer.score(mem).total
        new_payload = mem.to_payload()
        state.setdefault("history", []).append({
            "step": state["step"],
            "action": "stitch_compress",
            "committed_count": len(committed),
            "committed": committed[:10],  # truncate for log
            "mdl_before_first": mdl_before,
            "mdl_after": mdl_after,
            "scope": self.scope,
        })
        new_payload["stitch_reorg"] = state
        memory.payload = new_payload
        return memory

    def _should_trigger(self, state: dict[str, Any]) -> bool:
        step = int(state.get("step", 0))
        # Outcome schema (post RN-005): list[dict] with `score` key.
        # Backward-compat: tolerate plain floats.
        scores = []
        for o in state.get("outcomes", []) or []:
            try:
                scores.append(float(o["score"]) if isinstance(o, dict) else float(o))
            except (TypeError, ValueError, KeyError):
                continue
        if self.trigger == "every_k":
            return detect_plateau_every_k(step, k=self.every_k).should_trigger
        return detect_plateau(
            scores, window=self.plateau_window, min_delta=self.plateau_min_delta,
        ).should_trigger
