"""ArcMemoReorgMemoryBuilder: memory builder that layers a reorganization
operation on top of ArcMemoPs.

Four sub-axes (A1-A4) are exposed as constructor arguments:

  - A1 input_basis     : "graph_intrinsic" | "trace_based"
  - A2 objective        : "mdl" | "implicit"
  - A3 scope           : "global_rebuild" | "accretive"
  - A4 trigger         : "plateau" | "every_k"

The reorg operation (``_reorg``) is called on ``consolidate()`` when the
trigger fires. It selects popular nodes, groups their neighbors, aggregates
each group into a new parent concept, records authorship-lineage edges, and
(optionally) rebuilds the flat concept library from scratch.

NOTE: We intentionally keep this LLM-free. Aggregation is a template-based
summary — cues/implementation union, description concatenation. A future
variant can swap in an LLM-based aggregator when we have a provider handle
in consolidate(), but that is out of scope for Phase-1 validation of whether
reorg-the-mechanism helps at all.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from mem2.branches.feedback_engine.plateau_trigger import (
    detect_plateau,
    detect_plateau_every_k,
)
from mem2.concepts.data import Concept, ParameterSpec
from mem2.concepts.graph import ConceptGraph
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


class ArcMemoReorgMemoryBuilder:
    """Memory builder with DreamCoder-style reorganization.

    Shares the ``arcmemo_ps`` schema so existing retrievers work unchanged
    on the flat concept library. The graph + reorg history are stored in
    ``memory.payload["reorg"]``.
    """

    name = "arcmemo_reorg"
    SCHEMA_NAME = "arcmemo_ps"  # consumer-compatible; retrievers see concepts

    def __init__(
        self,
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 0,
        # --- A1 input basis ---
        input_basis: str = "graph_intrinsic",  # or "trace_based"
        # --- A2 objective ---
        objective: str = "mdl",  # or "implicit"
        # --- A3 scope ---
        scope: str = "global_rebuild",  # or "accretive"
        # --- A4 trigger ---
        trigger: str = "plateau",  # or "every_k"
        plateau_window: int = 10,
        plateau_min_delta: float = 0.01,
        every_k: int = 20,
        # --- reorg knobs ---
        top_k_popular: int = 8,
        min_group_size: int = 2,
        mdl_per_concept_overhead: float = 32.0,
        freeze_memory: bool = False,
    ):
        self.seed_memory_file = seed_memory_file
        self.seed_annotations_file = seed_annotations_file
        self.domain = domain
        self.max_concepts = int(max_concepts)
        self._frozen = bool(freeze_memory)
        self.input_basis = input_basis
        self.objective = objective
        self.scope = scope
        self.trigger = trigger
        self.plateau_window = int(plateau_window)
        self.plateau_min_delta = float(plateau_min_delta)
        self.every_k = int(every_k)
        self.top_k_popular = int(top_k_popular)
        self.min_group_size = int(min_group_size)
        self.scorer = MDLScorer(per_concept_overhead=mdl_per_concept_overhead)

    # ----------------------------------------------------------------- #
    #                          Lifecycle                                #
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
        base.payload.setdefault("reorg", {
            "history": [],
            "step": 0,
            "outcomes": [],
            "input_basis": self.input_basis,
            "objective": self.objective,
            "scope": self.scope,
            "trigger": self.trigger,
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
        reorg = memory.payload.setdefault("reorg", {
            "history": [], "step": 0, "outcomes": [],
            "input_basis": self.input_basis, "objective": self.objective,
            "scope": self.scope, "trigger": self.trigger,
        })

        for i, att in enumerate(attempts):
            is_correct = eval_records[i].is_correct if i < len(eval_records) else False
            if is_correct:
                mem.solutions[att.problem_uid] = ProblemSolution(
                    problem_id=att.problem_uid,
                    solution=(att.completion or "")[:2000],
                )
            reorg["step"] += 1
            # Outcome schema: dict with problem_id + score (consumed by
            # reorg_memp / reorg_evolver / reorg_lrll). Older code wrote
            # plain floats; subclasses that read outcomes expected dicts,
            # so unfired triggers hid the bug. RN-005 finding 3 surfaced it.
            reorg["outcomes"].append({
                "problem_id": att.problem_uid,
                "score": 1.0 if is_correct else 0.0,
                "step": reorg["step"],
            })

        memory.payload = mem.to_payload()
        # re-attach reorg state after payload replacement
        memory.payload["reorg"] = reorg
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if self._frozen:
            return memory
        reorg = memory.payload.get("reorg")
        if not reorg:
            return memory
        if not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)

        groups = self._select_groups(mem, graph, reorg)
        if not groups:
            reorg["history"].append({
                "step": reorg["step"],
                "action": "skipped",
                "reason": "no groups passed min_group_size/objective",
            })
            return memory

        mdl_before = self.scorer.score(mem).total
        new_concepts: list[Concept] = []
        lineage: list[tuple[str, str]] = []

        for parent_name, members in groups.items():
            if self.objective == "mdl":
                gain = -self.scorer.local_diff(mem, members)
                if gain <= 0:
                    continue  # don't aggregate when MDL would grow
            aggregate = self._aggregate(parent_name, members, mem)
            new_concepts.append(aggregate)
            for m in members:
                lineage.append((parent_name, m))

        if not new_concepts:
            reorg["history"].append({
                "step": reorg["step"],
                "action": "skipped",
                "reason": "no groups reduced MDL",
            })
            return memory

        if self.scope == "global_rebuild":
            rebuilt = ConceptMemory()
            # keep solutions and custom types
            rebuilt.solutions = dict(mem.solutions)
            rebuilt.custom_types = dict(mem.custom_types)
            for c in new_concepts:
                rebuilt.concepts[c.name] = c
                rebuilt.categories[c.kind].append(c.name)
            # keep children that were not aggregated
            aggregated = {m for _, m in lineage}
            for name, c in mem.concepts.items():
                if name in aggregated or name in rebuilt.concepts:
                    continue
                rebuilt.concepts[name] = c
                rebuilt.categories[c.kind].append(name)
            out_mem = rebuilt
        else:  # accretive
            for c in new_concepts:
                if c.name in mem.concepts:
                    continue
                mem.concepts[c.name] = c
                mem.categories[c.kind].append(c.name)
            out_mem = mem

        mdl_after = self.scorer.score(out_mem).total
        new_payload = out_mem.to_payload()
        reorg["history"].append({
            "step": reorg["step"],
            "action": "reorg",
            "num_new_concepts": len(new_concepts),
            "mdl_before": mdl_before,
            "mdl_after": mdl_after,
            "mdl_delta": mdl_after - mdl_before,
            "lineage": lineage,
            "scope": self.scope,
            "objective": self.objective,
            "input_basis": self.input_basis,
        })
        new_payload["reorg"] = reorg
        memory.payload = new_payload
        return memory

    # ----------------------------------------------------------------- #
    #                          Reorg internals                          #
    # ----------------------------------------------------------------- #
    def _should_reorg(self, reorg: dict[str, Any]) -> bool:
        step = int(reorg.get("step", 0))
        # Outcome schema is dict with `score` key (post RN-005 fix). Extract
        # numeric scores for plateau detection. Backward-compat: tolerate
        # plain floats from legacy state files.
        raw = list(reorg.get("outcomes", []))
        scores: list[float] = []
        for o in raw:
            if isinstance(o, dict):
                try:
                    scores.append(float(o.get("score", 0.0)))
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    scores.append(float(o))
                except (TypeError, ValueError):
                    continue
        if self.trigger == "every_k":
            return detect_plateau_every_k(step, k=self.every_k).should_trigger
        return detect_plateau(
            scores,
            window=self.plateau_window,
            min_delta=self.plateau_min_delta,
        ).should_trigger

    def _select_groups(
        self,
        mem: ConceptMemory,
        graph: ConceptGraph,
        reorg: dict[str, Any],
    ) -> dict[str, list[str]]:
        """Return {aggregate_name: [member_names]}.

        graph_intrinsic: pick top-K by co-activation degree, cluster neighbors.
        trace_based: group by shared ``used_in`` problem sets.
        """
        if self.input_basis == "trace_based":
            groups_by_sig: dict[tuple, list[str]] = defaultdict(list)
            for name, c in mem.concepts.items():
                sig = tuple(sorted(c.used_in))
                if len(sig) < 2:
                    continue
                groups_by_sig[sig].append(name)
            named_groups: dict[str, list[str]] = {}
            for i, members in enumerate(groups_by_sig.values()):
                if len(members) < self.min_group_size:
                    continue
                parent = f"trace_group_{i}__{members[0]}"
                named_groups[parent] = members
            return named_groups

        # graph_intrinsic
        degrees = [
            (graph.degree(n, kinds=["co_activation"]), n) for n in graph.nodes
        ]
        degrees.sort(reverse=True)
        top = [n for _, n in degrees[: self.top_k_popular]]
        seen: set[str] = set()
        named_groups: dict[str, list[str]] = {}
        for anchor in top:
            if anchor in seen:
                continue
            cluster = [anchor]
            for neighbor, kind, _w in graph.neighbors(anchor, kinds=["co_activation"]):
                if neighbor in seen or neighbor == anchor:
                    continue
                cluster.append(neighbor)
            if len(cluster) < self.min_group_size:
                continue
            for c in cluster:
                seen.add(c)
            parent = f"reorg_{anchor}_cluster"
            named_groups[parent] = cluster
        return named_groups

    def _aggregate(
        self, parent_name: str, members: list[str], mem: ConceptMemory
    ) -> Concept:
        """Template-based aggregate: union cues/impl, concat descriptions,
        union parameters by name. No LLM."""
        cues: list[str] = []
        impl: list[str] = []
        desc_pieces: list[str] = []
        params: dict[str, ParameterSpec] = {}
        used: list[str] = []
        kinds: list[str] = []
        subtypes: list[str | None] = []
        for m in members:
            c = mem.concepts.get(m)
            if c is None:
                continue
            cues.extend(c.cues or [])
            impl.extend(c.implementation or [])
            if c.description:
                desc_pieces.append(f"{c.name}: {c.description}")
            for p in (c.parameters or []):
                if p.name not in params:
                    params[p.name] = p
            used.extend(c.used_in or [])
            kinds.append(c.kind)
            subtypes.append(c.routine_subtype)
        kind = max(set(kinds), key=kinds.count) if kinds else "routine"
        subtype = next((s for s in subtypes if s), None)
        return Concept(
            name=parent_name,
            kind=kind,
            routine_subtype=subtype,
            parameters=list(params.values()),
            description=" | ".join(desc_pieces)[:2000] or None,
            cues=list(dict.fromkeys(cues)),
            implementation=list(dict.fromkeys(impl)),
            used_in=list(dict.fromkeys(used)),
        )
