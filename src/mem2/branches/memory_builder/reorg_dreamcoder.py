"""DreamCoder-style wake-sleep library compression — axis A.2.

Port of the compression ("sleep") phase from DreamCoder (Ellis et al., PLDI'21).

Paper: literature/2006.08381.pdf
Repo:  third_party/dreamcoder/
Specifically ported:
    - `dreamcoder/compression.py::induceGrammar` — the invented-primitive
      addition loop driven by MDL.
    - `dreamcoder/fragmentGrammar.py::FragmentGrammar.insideOutside` — the
      fragment-frequency + MDL-gain scoring pattern.

Deliberate simplifications (LLM-free, Python-only, no OCaml/PyPy backends):
    - "Frontiers" = concepts with shared `used_in` (problem IDs).
    - "Common subtrees" = shared cue/implementation lines across concepts
      that co-occur on training problems. Line-level granularity.
    - "Invented primitives" = new fragment concepts owning just the shared
      lines; original concepts keep their other content.
    - MDL gate via `MDLScorer` (shared with `arcmemo_reorg`). Fragments are
      only committed when total description length strictly decreases.
    - No recognition model / Helmholtz machine. No beam search. Greedy.

Distinct from `arcmemo_reorg` (A.1):
    - Reorg clusters WHOLE concepts into a single concatenated aggregate.
    - DreamCoder-port extracts SHARED LINES across concept pairs into a
      fragment concept; the children retain their private content. Finer
      granularity; closer to the paper's "compression-as-abstraction" story.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
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


class DreamCoderReorgBuilder:
    """DreamCoder-inspired reorganization: line-level fragment extraction.

    Trigger + scope + objective / trigger knobs mirror `arcmemo_reorg`:
      - `trigger`: plateau | every_k
      - `scope`: global_rebuild | accretive (whether we keep originals or
        drop them once their lines are covered by a fragment)
    """

    name = "reorg_dreamcoder"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 0,
        # --- Trigger / scope ---
        trigger: str = "every_k",
        plateau_window: int = 10,
        plateau_min_delta: float = 0.01,
        every_k: int = 20,
        scope: str = "accretive",
        # --- Fragment-extraction knobs ---
        min_shared_lines: int = 2,
        min_fragment_frequency: int = 2,
        mdl_per_concept_overhead: float = 32.0,
    ):
        self.seed_memory_file = seed_memory_file
        self.seed_annotations_file = seed_annotations_file
        self.domain = domain
        self.max_concepts = int(max_concepts)
        self.trigger = trigger
        self.plateau_window = int(plateau_window)
        self.plateau_min_delta = float(plateau_min_delta)
        self.every_k = int(every_k)
        self.scope = scope
        self.min_shared_lines = int(min_shared_lines)
        self.min_fragment_frequency = int(min_fragment_frequency)
        self.scorer = MDLScorer(per_concept_overhead=mdl_per_concept_overhead)

    # ----------------------------------------------------------------- #
    #                           Lifecycle                               #
    # ----------------------------------------------------------------- #
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
        base.payload.setdefault("dreamcoder_reorg", {
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
        mem = ConceptMemory.from_payload(memory.payload)
        state = memory.payload.setdefault("dreamcoder_reorg", {
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
        memory.payload["dreamcoder_reorg"] = state
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        state = memory.payload.get("dreamcoder_reorg")
        if not state or not self._should_trigger(state):
            return memory
        mem = ConceptMemory.from_payload(memory.payload)

        mdl_before = self.scorer.score(mem).total
        fragments = self._extract_fragments(mem)
        if not fragments:
            state.setdefault("history", []).append({
                "step": state["step"],
                "action": "skipped",
                "reason": "no fragments above min_shared_lines / min_fragment_frequency",
            })
            return memory

        committed: list[dict[str, Any]] = []
        for fragment in fragments:
            # MDL gate: commit only if adding this fragment strictly reduces total DL
            # Simulate: clone mem, add fragment, score, compare.
            sim = ConceptMemory.from_payload(mem.to_payload())
            sim.concepts[fragment["name"]] = fragment["concept"]
            sim.categories[fragment["concept"].kind].append(fragment["name"])
            if self.scope == "global_rebuild":
                # Remove shared lines from child concepts in the sim
                for child_name in fragment["children"]:
                    c = sim.concepts.get(child_name)
                    if c is None:
                        continue
                    c.cues = [ln for ln in (c.cues or []) if ln not in fragment["shared_cues"]]
                    c.implementation = [ln for ln in (c.implementation or []) if ln not in fragment["shared_impl"]]
            sim_mdl = self.scorer.score(sim).total
            if sim_mdl >= mdl_before:
                continue  # no MDL improvement

            # Commit
            mem.concepts[fragment["name"]] = fragment["concept"]
            if fragment["name"] not in mem.categories[fragment["concept"].kind]:
                mem.categories[fragment["concept"].kind].append(fragment["name"])
            if self.scope == "global_rebuild":
                for child_name in fragment["children"]:
                    c = mem.concepts.get(child_name)
                    if c is None:
                        continue
                    c.cues = [ln for ln in (c.cues or []) if ln not in fragment["shared_cues"]]
                    c.implementation = [ln for ln in (c.implementation or []) if ln not in fragment["shared_impl"]]
            committed.append({
                "fragment": fragment["name"],
                "children": fragment["children"],
                "shared_cues": list(fragment["shared_cues"]),
                "shared_impl": list(fragment["shared_impl"]),
                "mdl_delta": sim_mdl - mdl_before,
            })
            mdl_before = sim_mdl  # baseline shifts after each commit

        if not committed:
            state.setdefault("history", []).append({
                "step": state["step"],
                "action": "skipped",
                "reason": "all candidate fragments failed MDL gate",
            })
            return memory

        mdl_after = self.scorer.score(mem).total
        new_payload = mem.to_payload()
        state.setdefault("history", []).append({
            "step": state["step"],
            "action": "dreamcoder_compress",
            "committed_count": len(committed),
            "committed": committed,
            "mdl_before": mdl_before,  # final value from last iteration
            "mdl_after": mdl_after,
            "scope": self.scope,
        })
        new_payload["dreamcoder_reorg"] = state
        memory.payload = new_payload
        return memory

    # ----------------------------------------------------------------- #
    #                         Fragment extraction                       #
    # ----------------------------------------------------------------- #
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

    def _extract_fragments(self, mem: ConceptMemory) -> list[dict[str, Any]]:
        """Find shared-line fragments across concepts with overlapping used_in.

        Returns a list of fragment dicts (candidate sequence; caller MDL-gates
        each one). Greedy pairwise scan: for each pair of concepts sharing
        ≥1 `used_in` problem, compute the set of shared cue + implementation
        lines. If that set has ≥min_shared_lines entries AND the (line, count)
        frequency across all concepts is ≥min_fragment_frequency, create a
        fragment concept.
        """
        # Step 1: compute line-frequency across all concepts
        line_freq: Counter[tuple[str, str]] = Counter()  # (kind, line) → count
        for name, c in mem.concepts.items():
            for cue in (c.cues or []):
                line_freq[("cue", cue)] += 1
            for impl in (c.implementation or []):
                line_freq[("impl", impl)] += 1

        # Step 2: group concepts by used_in overlap
        by_problem: dict[str, list[str]] = defaultdict(list)
        for name, c in mem.concepts.items():
            for pid in (c.used_in or []):
                by_problem[pid].append(name)

        # Step 3: find candidate fragments (pairs with sufficient shared lines)
        candidates: list[dict[str, Any]] = []
        seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for pid, names in by_problem.items():
            names = sorted(set(names))
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    c_i = mem.concepts.get(names[i])
                    c_j = mem.concepts.get(names[j])
                    if c_i is None or c_j is None:
                        continue
                    cues_i = set(c_i.cues or [])
                    cues_j = set(c_j.cues or [])
                    impl_i = set(c_i.implementation or [])
                    impl_j = set(c_j.implementation or [])
                    shared_cues = {
                        ln for ln in (cues_i & cues_j)
                        if line_freq[("cue", ln)] >= self.min_fragment_frequency
                    }
                    shared_impl = {
                        ln for ln in (impl_i & impl_j)
                        if line_freq[("impl", ln)] >= self.min_fragment_frequency
                    }
                    if len(shared_cues) + len(shared_impl) < self.min_shared_lines:
                        continue
                    sig = (
                        tuple(sorted(shared_cues)),
                        tuple(sorted(shared_impl)),
                    )
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    # Expand children: find all concepts containing all shared
                    # lines (not just the pair that surfaced the fragment)
                    children = []
                    for n, c in mem.concepts.items():
                        c_cues = set(c.cues or [])
                        c_impl = set(c.implementation or [])
                        if shared_cues.issubset(c_cues) and shared_impl.issubset(c_impl):
                            children.append(n)
                    if len(children) < self.min_fragment_frequency:
                        continue

                    frag_name = f"fragment__{children[0][:24]}"
                    fragment_concept = Concept(
                        name=frag_name,
                        kind="routine",
                        routine_subtype="fragment",
                        description=f"DreamCoder fragment shared across {len(children)} concepts",
                        cues=sorted(shared_cues),
                        implementation=sorted(shared_impl),
                        used_in=sorted({pid for n in children for pid in (mem.concepts[n].used_in or [])}),
                    )
                    candidates.append({
                        "name": frag_name,
                        "concept": fragment_concept,
                        "children": children,
                        "shared_cues": shared_cues,
                        "shared_impl": shared_impl,
                        "signature": sig,
                    })

        # Rank by (total_shared_lines × num_children) — biggest MDL savings first
        candidates.sort(
            key=lambda f: (len(f["shared_cues"]) + len(f["shared_impl"])) * len(f["children"]),
            reverse=True,
        )
        return candidates
