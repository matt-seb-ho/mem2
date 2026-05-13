"""PathRAG path-based retrieval with flow-pruning — axis B.10.

Port of PathRAG (Chen, Guo, Yang et al., 2026 AAAI; arxiv 2502.14902).

Paper: literature/2502.14902.pdf
Repo:  https://github.com/BUPT-GAMMA/PathRAG (not locally mirrored).

Specifically ported:
    - The *path-based retrieval* pattern: identify query-relevant nodes,
      then find key PATHS between them in the indexing graph.
    - The *flow-based pruning* with reliability scores: enumerate candidate
      paths, score each by a flow-awareness metric (edge-weight product
      along the path, distance-decayed), prune low-reliability paths.
    - Render retrieved paths as TEXTUAL relational sequences in the prompt
      (ascending reliability order, mitigating "lost in the middle").

Deliberate simplifications (no repo locally; paper-only port):
    - Query-relevant nodes are the top-k by token-overlap against the
      query (same mechanic as B.8 ColBERT for first-stage identification).
    - "Path reliability" = product of co-activation edge weights along
      the path, distance-decayed by 1/(1+len).
    - BFS depth-limited enumeration — max_path_length=3 to avoid
      combinatorial blow-up. Paper uses learned distance awareness; we
      use graph-distance as a proxy.

B.10 vs B.2 / B.3 / B.7:
    - B.2 graph_traversal: BFS from seed nodes, no path reasoning.
    - B.3 GraphRAG: community summaries, no paths.
    - B.7 LightRAG: entity + edge dual, but flat.
    - B.10 PathRAG (this module): PATH between query-related nodes is the
      retrieval unit — the edges-between are part of the answer, not just
      the traversal scaffolding.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)

WORD_RE = re.compile(r"\w+")


class PathRAGRetriever:
    """Flow-pruned path retrieval over ConceptGraph."""

    name = "pathrag"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k_seeds: int = 4,
        max_path_length: int = 3,
        min_reliability: float = 0.1,
        max_paths_rendered: int = 5,
        edge_kinds: tuple[str, ...] = ("co_activation", "openie_fact"),
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = True,
        usage_threshold: int = 0,
    ) -> None:
        self.top_k_seeds = int(top_k_seeds)
        self.max_path_length = int(max_path_length)
        self.min_reliability = float(min_reliability)
        self.max_paths_rendered = int(max_paths_rendered)
        self.edge_kinds = tuple(edge_kinds)
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

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1, load_openie_edges=True)

        q_toks = self._query_toks(problem, previous_attempts)

        # Stage 1 — seed nodes by query token overlap.
        scored: list[tuple[float, str]] = []
        for name, c in mem.concepts.items():
            doc_toks = {name.lower()}
            if c.description:
                doc_toks.update(t.lower() for t in WORD_RE.findall(c.description))
            for cue in c.cues or []:
                doc_toks.update(t.lower() for t in WORD_RE.findall(cue))
            overlap = len(q_toks & doc_toks) if q_toks else len(c.used_in or [])
            scored.append((float(overlap), name))
        scored.sort(reverse=True)
        seeds = [name for _, name in scored[: self.top_k_seeds] if name in mem.concepts]

        # Stage 2 — enumerate paths between seed pairs (length ≤ max_path_length).
        all_paths: list[tuple[float, list[str]]] = []
        for i, src in enumerate(seeds):
            for dst in seeds[i + 1:]:
                for path in self._enumerate_paths(graph, src, dst, self.max_path_length):
                    rel = self._path_reliability(graph, path)
                    if rel >= self.min_reliability:
                        all_paths.append((rel, path))

        # Sort ascending by reliability (paper: place best at end, "lost in the middle").
        all_paths.sort()
        top_paths = all_paths[-self.max_paths_rendered:]

        # Nodes across all rendered paths.
        selected_nodes: list[str] = []
        seen: set[str] = set()
        for _, path in top_paths:
            for n in path:
                if n not in seen:
                    seen.add(n)
                    selected_nodes.append(n)
        if not selected_nodes:
            # No paths — fall back to raw seeds.
            selected_nodes = seeds

        # Render: base hint (concept details) + path summary block at top.
        base_hint = mem.to_string(
            concept_names=selected_nodes,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            usage_threshold=self.usage_threshold,
        )
        path_block = self._render_paths(graph, top_paths)
        hint = (path_block + "\n\n" + (base_hint or "")) if path_block else base_hint

        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in selected_nodes],
            metadata={
                "retriever": self.name,
                "seeds": seeds,
                "paths_found": len(all_paths),
                "paths_rendered": len(top_paths),
                "num_selected": len(selected_nodes),
                "min_reliability": self.min_reliability,
                "top_reliability": top_paths[-1][0] if top_paths else 0.0,
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
    def _query_toks(
        self, problem: ProblemSpec, previous_attempts: list[AttemptRecord],
    ) -> set[str]:
        parts: list[str] = [str(getattr(problem, "uid", ""))]
        meta = getattr(problem, "metadata", {}) or {}
        for key in ("description", "instructions", "prompt", "query"):
            if meta.get(key):
                parts.append(str(meta[key]))
        return {t.lower() for t in WORD_RE.findall(" ".join(parts))}

    def _enumerate_paths(
        self, graph: ConceptGraph, src: str, dst: str, max_len: int,
    ) -> list[list[str]]:
        """BFS-enumerate simple paths from src to dst (length ≤ max_len)."""
        if src == dst:
            return [[src]]
        # BFS queue of (current, path).
        paths: list[list[str]] = []
        queue: list[list[str]] = [[src]]
        while queue:
            path = queue.pop(0)
            if len(path) > max_len:
                continue
            current = path[-1]
            for nbr, kind, _w in graph.neighbors(current, kinds=self.edge_kinds):
                if nbr in path:
                    continue
                new_path = path + [nbr]
                if nbr == dst:
                    paths.append(new_path)
                elif len(new_path) < max_len:
                    queue.append(new_path)
            if len(paths) >= 10:  # cap per-pair enumeration
                break
        return paths

    def _path_reliability(self, graph: ConceptGraph, path: list[str]) -> float:
        """Product of edge weights along path, distance-decayed."""
        if len(path) < 2:
            return 0.0
        prod = 1.0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            w = 0.0
            for nbr, kind, weight in graph.neighbors(src, kinds=self.edge_kinds):
                if nbr == dst:
                    w = max(w, float(weight))
                    break
            prod *= w
        return prod / (1.0 + len(path))

    def _render_paths(self, graph: ConceptGraph, paths: list[tuple[float, list[str]]]) -> str:
        if not paths:
            return ""
        lines = ["## key relational paths (ascending reliability)"]
        for rel, p in paths:
            lines.append(f"  [{rel:.3f}] " + self._render_path(graph, p))
        return "\n".join(lines)

    def _render_path(self, graph: ConceptGraph, path: list[str]) -> str:
        if len(path) < 2:
            return " -> ".join(path)
        parts = [path[0]]
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            edge = graph.edge_between(src, dst, kinds=("openie_fact", "co_activation"))
            if edge and edge.kind == "openie_fact":
                predicate = str((edge.metadata or {}).get("predicate") or "relates_to").strip()
                parts.append(f"--{predicate}--")
            else:
                parts.append("->")
            parts.append(dst)
        return " ".join(parts)
