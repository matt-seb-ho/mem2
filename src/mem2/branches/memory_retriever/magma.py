"""MAGMA multi-graph policy-guided retrieval — axis B.11.

Port of MAGMA (Jiang, Li, Li, Li, 2026; arxiv 2601.03236).

Paper: literature/2601.03236.pdf
Repo:  https://github.com/FredJiang0324/MAGMA (not locally mirrored).

Specifically ported:
    - The *multi-graph orthogonal views*: represent memory as 4 relational
      views — SEMANTIC (concept-concept co-activation), TEMPORAL
      (authorship-lineage edges), CAUSAL (shared-problem usage), and
      ENTITY (kind-membership). Each view exposes a different aspect of
      the same substrate.
    - The *adaptive traversal policy*: at retrieval time, score the query
      against each view; select the subset of views that are "active"
      for this query intent, traverse each independently, fuse the
      resulting subgraphs into the final context.

Deliberate simplifications (no repo locally; paper-only port):
    - Views are built over the same ConceptGraph but filter/re-weight
      edges by kind. Semantic = co_activation. Temporal = authorship_lineage.
      Causal = a view we synthesize by connecting concepts that appear in
      the same `used_in` problems (we materialize this as a set of
      "problem-cluster" implicit edges at retrieve time). Entity = kind
      groupings from `mem.categories`.
    - The "adaptive policy" is template-based: count query-token hits in
      each view's entries; a view is active if it has ≥1 hit. LLM mode
      (via `_meta_edit_provider`) upgrades to a proper policy call.
    - Fusion is type-aligned concatenation (§3.3 "type-aligned context"):
      we render a separate block per active view.

B.11 vs B.3 / B.4 / B.10:
    - B.3 GraphRAG: single view (co-activation communities).
    - B.4 HippoRAG PPR: single view (full graph PPR).
    - B.10 PathRAG: paths in a single view.
    - B.11 MAGMA (this module): *multiple orthogonal views*; policy selects
      which views to activate per query; no single-view can answer cross-
      modal queries (e.g., temporal + causal).
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"\w+")

VIEWS = ("semantic", "temporal", "causal", "entity")


class MAGMAMultiGraphRetriever:
    """Multi-view retrieval with adaptive policy selection."""

    name = "magma_multigraph"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k_per_view: int = 2,
        max_active_views: int = 3,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = True,
        usage_threshold: int = 0,
    ) -> None:
        self.top_k_per_view = int(top_k_per_view)
        self.max_active_views = int(max_active_views)
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

        graph = ConceptGraph.build_from_memory(
            mem,
            min_co_overlap=1,
            load_openie_edges=True,
            load_entity_edges=True,
        )
        provider = self._resolve_provider(ctx)
        q_toks = self._query_toks(problem, previous_attempts)

        # Per-view candidate sets.
        view_hits: dict[str, list[str]] = {}
        view_hits["semantic"] = self._semantic_view(graph, mem, q_toks)
        view_hits["temporal"] = self._temporal_view(graph, mem, q_toks)
        view_hits["causal"] = self._causal_view(graph, mem, q_toks)
        view_hits["entity"] = self._entity_view(graph, mem, q_toks)

        # Adaptive policy: rank views by hit count; LLM can override.
        policy_ranking = sorted(
            view_hits.items(), key=lambda kv: len(kv[1]), reverse=True,
        )
        if provider is not None:
            override = self._policy_via_llm(provider, problem, view_hits)
            if override:
                policy_ranking = [(v, view_hits[v]) for v in override if v in view_hits]

        active_views = [
            (v, cands) for v, cands in policy_ranking[: self.max_active_views]
            if cands
        ]

        # Fuse: render per-view block.
        view_blocks: list[str] = []
        all_selected: list[str] = []
        seen: set[str] = set()
        for v, cands in active_views:
            picks = [c for c in cands if c in mem.concepts][: self.top_k_per_view]
            if not picks:
                continue
            hint_block = mem.to_string(
                concept_names=picks,
                include_description=self.include_description,
                skip_cues=self.skip_cues,
                skip_implementation=self.skip_implementation,
                usage_threshold=self.usage_threshold,
            )
            view_blocks.append(f"## view: {v}\n{hint_block}")
            for p in picks:
                if p not in seen:
                    seen.add(p)
                    all_selected.append(p)

        hint = "\n\n".join(view_blocks) if view_blocks else None
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint,
            retrieved_items=[{"name": n} for n in all_selected],
            metadata={
                "retriever": self.name,
                "scoring_mode": "magma_multigraph",
                "active_views": [v for v, _ in active_views],
                "view_hit_counts": {v: len(c) for v, c in view_hits.items()},
                "num_selected": len(all_selected),
                "used_llm_policy": provider is not None,
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
    def _resolve_provider(self, ctx: RunContext):
        try:
            return (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            return None

    def _query_toks(
        self, problem: ProblemSpec, previous_attempts: list[AttemptRecord],
    ) -> set[str]:
        parts: list[str] = [str(getattr(problem, "uid", ""))]
        meta = getattr(problem, "metadata", {}) or {}
        for key in ("description", "instructions", "prompt", "query"):
            if meta.get(key):
                parts.append(str(meta[key]))
        return {t.lower() for t in WORD_RE.findall(" ".join(parts))}

    def _concept_toks(self, c) -> set[str]:
        toks: set[str] = {c.name.lower()}
        if c.description:
            toks.update(t.lower() for t in WORD_RE.findall(c.description))
        for cue in c.cues or []:
            toks.update(t.lower() for t in WORD_RE.findall(cue))
        return toks

    def _semantic_view(
        self, graph: ConceptGraph, mem: ConceptMemory, q_toks: set[str],
    ) -> list[str]:
        """Co-activation and fact-edge nodes scored by token overlap."""
        scored: list[tuple[float, str]] = []
        for name, c in mem.concepts.items():
            overlap = len(q_toks & self._concept_toks(c)) if q_toks else 0
            fact_overlap = self._fact_edge_overlap(graph, name, q_toks)
            if overlap == 0 and fact_overlap == 0:
                continue
            deg = graph.degree(name, kinds=["co_activation", "openie_fact", "entity_relation"])
            scored.append((overlap + fact_overlap + 0.01 * deg, name))
        scored.sort(reverse=True)
        return [n for _, n in scored[: self.top_k_per_view * 2]]

    def _temporal_view(
        self, graph: ConceptGraph, mem: ConceptMemory, q_toks: set[str],
    ) -> list[str]:
        """Nodes with many authorship_lineage edges (proxy for temporal depth)."""
        scored: list[tuple[float, str]] = []
        for name in mem.concepts:
            deg = graph.degree(name, kinds=["authorship_lineage"])
            if deg > 0:
                scored.append((deg, name))
        scored.sort(reverse=True)
        return [n for _, n in scored[: self.top_k_per_view * 2]]

    def _causal_view(
        self, graph: ConceptGraph, mem: ConceptMemory, q_toks: set[str],
    ) -> list[str]:
        """Nodes in dense problem-clusters or fact-linked causal relations."""
        scored: list[tuple[float, str]] = []
        for name, c in mem.concepts.items():
            fact_overlap = self._fact_edge_overlap(graph, name, q_toks)
            fact_degree = graph.degree(name, kinds=["openie_fact"])
            if not c.used_in and fact_degree <= 0:
                continue
            overlap = len(q_toks & self._concept_toks(c))
            if overlap == 0 and fact_overlap == 0 and len(c.used_in) < 3:
                continue
            scored.append((len(c.used_in) + overlap + fact_overlap + 0.05 * fact_degree, name))
        scored.sort(reverse=True)
        return [n for _, n in scored[: self.top_k_per_view * 2]]

    def _fact_edge_overlap(
        self, graph: ConceptGraph, name: str, q_toks: set[str],
    ) -> int:
        if not q_toks:
            return 0
        score = 0
        for nbr, kind, _weight in graph.neighbors(name, kinds=["openie_fact"]):
            edge = graph.edge_between(name, nbr, kinds=["openie_fact"])
            if not edge:
                continue
            text = " ".join(
                str((edge.metadata or {}).get(k) or "")
                for k in ("subject", "predicate", "object", "supporting_text")
            )
            score += len(q_toks & {t.lower() for t in WORD_RE.findall(text)})
        return score

    def _entity_view(
        self, graph: ConceptGraph, mem: ConceptMemory, q_toks: set[str],
    ) -> list[str]:
        """Entity-graph view: prioritize concepts with matching entity relations."""
        by_kind = defaultdict(list)
        for name, c in mem.concepts.items():
            overlap = len(q_toks & self._concept_toks(c))
            entity_overlap = self._entity_edge_overlap(graph, name, q_toks)
            entity_degree = graph.degree(name, kinds=["entity_relation"])
            score = overlap + entity_overlap + 0.05 * entity_degree
            by_kind[c.kind].append((score, name))
        picks: list[str] = []
        for kind, items in by_kind.items():
            items.sort(reverse=True)
            if items and items[0][0] > 0:
                picks.extend(name for _, name in items[: self.top_k_per_view])
        return picks[: self.top_k_per_view * 2]

    def _entity_edge_overlap(
        self, graph: ConceptGraph, name: str, q_toks: set[str],
    ) -> int:
        if not q_toks:
            return 0
        score = 0
        for nbr, kind, _weight in graph.neighbors(name, kinds=["entity_relation"]):
            edge = graph.edge_between(name, nbr, kinds=["entity_relation"])
            if not edge:
                continue
            text = " ".join(
                str((edge.metadata or {}).get(k) or "")
                for k in ("src_mention", "dst_mention", "edge_type", "supporting_text")
            )
            score += len(q_toks & {t.lower() for t in WORD_RE.findall(text)})
        return score

    def _policy_via_llm(
        self, provider, problem: ProblemSpec, view_hits: dict[str, list[str]],
    ) -> list[str] | None:
        q = str(getattr(problem, "uid", ""))
        meta = getattr(problem, "metadata", {}) or {}
        if meta.get("description"):
            q += " | " + str(meta["description"])[:200]
        summary = "\n".join(
            f"- {v}: {len(cands)} candidates — top: {cands[:3]}"
            for v, cands in view_hits.items()
        )
        prompt = (
            "You are a MAGMA view policy. Select the best 1-3 views to "
            "activate for this query, ordered by priority. Views: semantic, "
            "temporal, causal, entity.\n\n"
            f"Query: {q}\n\n{summary}\n\n"
            'Output JSON list of view names: ["semantic", ...]'
        )
        try:
            out = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = out[0] if out else "[]"
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [v for v in parsed if isinstance(v, str) and v in VIEWS]
        except Exception as exc:  # pragma: no cover
            logger.warning("magma policy LLM call failed: %s", exc)
        return None
