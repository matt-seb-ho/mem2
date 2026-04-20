"""GraphTraversalRetriever: hierarchical retrieval over ConceptGraph.

Axis B in the ablation plan: flat top-k (existing ``ps_selector`` /
``oe_topk``) vs graph-traversal.

Algorithm:
  1. Build a ConceptGraph from memory (co-activation edges).
  2. Find aggregate roots (nodes that are the ``src`` of ``authorship_lineage``
     edges — produced by reorg). Fall back to top-degree nodes.
  3. BFS from roots, weighted by edge strength. Preserves roots first, then
     breadth-ordered descendants.
  4. Cap at ``top_k`` concepts, render via ConceptMemory.to_string.

This is intentionally LLM-free: retrieval is a pure graph walk, reproducible
across seeds. The only randomness is optional tie-breaking (controlled by
``ctx.seed``).
"""
from __future__ import annotations

import random
from collections import deque

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


class GraphTraversalRetriever:
    name = "graph_traversal"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        bfs_depth: int = 3,
        prefer_aggregates: bool = True,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
    ) -> None:
        self.top_k = int(top_k)
        self.bfs_depth = int(bfs_depth)
        self.prefer_aggregates = bool(prefer_aggregates)
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
        # Lineage edges live on the reorg payload — re-attach them if present.
        for entry in memory.payload.get("reorg", {}).get("history", []):
            for parent, child in entry.get("lineage", []) or []:
                graph.add_authorship_lineage(parent, child)

        rng = random.Random(getattr(ctx, "seed", 0))
        roots = self._pick_roots(graph, rng)
        selected = self._bfs(graph, roots, rng)[: self.top_k]

        # Only render concepts that exist (aggregate names may be lineage-only).
        rendered_names = [n for n in selected if n in concept_mem.concepts]
        hint_text = concept_mem.to_string(
            concept_names=rendered_names,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            usage_threshold=self.usage_threshold,
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text or None,
            retrieved_items=[{"name": n} for n in rendered_names],
            metadata={
                "retriever": self.name,
                "top_k": self.top_k,
                "bfs_depth": self.bfs_depth,
                "num_roots": len(roots),
                "num_selected": len(rendered_names),
                "num_graph_nodes": graph.num_nodes(),
                "num_graph_edges": graph.num_edges(),
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
    def _pick_roots(self, g: ConceptGraph, rng: random.Random) -> list[str]:
        if self.prefer_aggregates:
            parents = sorted({e.src for e in g.edges() if e.kind == "authorship_lineage"})
            if parents:
                return parents
        # Fall back: top-K by weighted degree on co-activation.
        scored = [(g.degree(n, kinds=["co_activation"]), n) for n in g.nodes]
        scored.sort(key=lambda r: (r[0], rng.random()), reverse=True)
        return [n for _, n in scored[: max(self.top_k, 1)]]

    def _bfs(self, g: ConceptGraph, roots: list[str], rng: random.Random) -> list[str]:
        seen: set[str] = set()
        order: list[str] = []
        queue: deque[tuple[str, int]] = deque()
        for r in roots:
            if r not in seen:
                seen.add(r)
                order.append(r)
                queue.append((r, 0))
        while queue and len(order) < self.top_k:
            node, depth = queue.popleft()
            if depth >= self.bfs_depth:
                continue
            neighbors = g.neighbors(node)
            # weighted random order to avoid degenerate deterministic walks
            neighbors.sort(key=lambda nb: (nb[2], rng.random()), reverse=True)
            for nxt, _kind, _w in neighbors:
                if nxt in seen:
                    continue
                seen.add(nxt)
                order.append(nxt)
                queue.append((nxt, depth + 1))
                if len(order) >= self.top_k:
                    break
        return order
