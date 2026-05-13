"""HippoRAG PPR retriever — axis B.4.

Port of the Personalized PageRank core mechanism from HippoRAG
(Gutierrez et al., NeurIPS'24 and "From RAG to Memory", ICML'25).

Papers:
    - literature/2405.14831.pdf   HippoRAG 1 (PPR over document-entity graph)
    - literature/2502.14802.pdf   HippoRAG 2 (same PPR core, continual learning)

Reference repo:
    third_party/hipporag/  (https://github.com/OSU-NLP-Group/HippoRAG)

Specifically ported:
    - `src/hipporag/HippoRAG.py::run_ppr` — the igraph.personalized_pagerank
      call with reset vector + damping, NaN clamping.
    - `src/hipporag/HippoRAG.py::graph_search_with_fact_entities` — the
      reset-vector construction pattern (query-relevant facts/phrases seed
      the personalization).

Deliberate simplifications (what we do NOT port):
    - No OpenIE fact extraction at query time (HippoRAG uses an LLM to
      extract (subject, predicate, object) tuples from the query). We use
      simple token-overlap between the ARC problem text and each concept's
      rendered text — keeps retrieval deterministic-on-seed and LLM-free
      like `ps_topk`. The paper's OpenIE is document-RAG-specific; we're
      retrieving concepts, not passages.
    - No dense passage retrieval signal merged in. We rank purely by PPR.
    - Swap igraph → networkx (`personalized` kwarg to `nx.pagerank`). Same
      algorithm, different library dep. Adds `networkx>=3.x` to mem2 env.

Interface: matches `ps_topk` + `graph_traversal` (axis-B-compatible
retrievers on arcmemo_ps schema).
"""
from __future__ import annotations

import re

from mem2.concepts.artifacts import load_entity_graph, load_openie_facts
from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


def _problem_text(problem: ProblemSpec) -> str:
    """Concatenate all textual content in a ProblemSpec for keyword overlap."""
    parts: list[str] = []
    if problem.metadata:
        for k, v in problem.metadata.items():
            if isinstance(v, str):
                parts.append(v)
    for p in (problem.train_pairs or []):
        if isinstance(p, dict):
            for k, v in p.items():
                if isinstance(v, str):
                    parts.append(v)
    for p in (problem.test_pairs or []):
        if isinstance(p, dict):
            for k, v in p.items():
                if isinstance(v, str):
                    parts.append(v)
    parts.append(problem.uid)
    return "\n".join(parts)


class HippoRAGPPRRetriever:
    """Personalized-PageRank retriever over a `ConceptGraph`.

    Algorithm:
      1. Build the `ConceptGraph` from the current `ConceptMemory`.
      2. For each concept, compute a **reset weight** = token-overlap between
         the ARC problem text and the concept's rendered text (name +
         description + cues).
      3. Normalize reset weights to a probability distribution (sum to 1).
         Concepts with zero overlap get 0 mass — they can still be picked
         up as PPR neighbors of seed nodes.
      4. Run networkx `pagerank(G, personalization=reset_dict, alpha=damping)`.
      5. Sort concepts by PPR score, take top-k, render.

    `COMPATIBLE_SCHEMAS = {"arcmemo_ps"}` — reads ConceptMemory payload.
    """

    name = "hipporag_ppr"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 3,
        damping: float = 0.5,
        min_reset_overlap: int = 1,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
        edge_kinds: tuple[str, ...] = (
            "co_activation",
            "openie_fact",
            "entity_relation",
            "uses", "is_a", "specializes", "opposite_of", "composed_of",
        ),
    ) -> None:
        self.top_k = int(top_k)
        self.damping = float(damping)
        self.min_reset_overlap = int(min_reset_overlap)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)
        self.edge_kinds = tuple(edge_kinds)

    # ----------------------------------------------------------------- #
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

        # Build graph (co-activation edges only by default; embedding-sim
        # edges would need an embed_fn which we don't wire at retrieval
        # time).
        graph = ConceptGraph.build_from_memory(
            mem,
            min_co_overlap=1,
            load_openie_edges=True,
            load_entity_edges=True,
        )

        # Compute reset-probability vector via query-concept keyword overlap.
        q_tokens = _tokenize(_problem_text(problem))
        reset: dict[str, float] = {}
        for name, concept in mem.concepts.items():
            c_text = concept.to_string(
                include_description=True, skip_cues=False, skip_implementation=False,
            )
            c_tokens = _tokenize(c_text)
            overlap = len(q_tokens & c_tokens)
            if overlap >= self.min_reset_overlap:
                reset[name] = float(overlap)
        fact_reset, fact_reset_matches = self._fact_reset_weights(mem, q_tokens)
        for name, weight in fact_reset.items():
            reset[name] = reset.get(name, 0.0) + weight
        entity_reset, entity_reset_matches = self._entity_reset_weights(mem, q_tokens)
        for name, weight in entity_reset.items():
            reset[name] = reset.get(name, 0.0) + weight
        total_mass = sum(reset.values())
        if total_mass <= 0:
            # Fall back to uniform: every node seeds PPR equally. Equivalent
            # to un-personalized PageRank — still better than random over
            # the graph.
            reset = {n: 1.0 for n in mem.concepts.keys()}
            total_mass = float(len(reset))
        for n in list(reset.keys()):
            reset[n] /= total_mass

        # Run PPR via networkx. Add nodes without edges so they exist in G
        # (isolated concepts still get their reset mass back each step).
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "hipporag_ppr requires networkx; install with `pip install networkx`."
            ) from exc

        G = nx.Graph()
        for n in mem.concepts.keys():
            G.add_node(n)
        fact_graph_edges_used = 0
        entity_graph_edges_used = 0
        for edge in graph.edges():
            if edge.kind not in self.edge_kinds:
                continue
            G.add_edge(edge.src, edge.dst, weight=float(edge.weight or 1.0))
            if edge.kind == "openie_fact":
                fact_graph_edges_used += 1
            elif edge.kind == "entity_relation":
                entity_graph_edges_used += 1

        # Only include concepts that are actually in the graph in the reset
        # map; extras raise NetworkXError.
        reset_clipped = {k: v for k, v in reset.items() if k in G}
        if not reset_clipped:
            # Edge case: graph has no nodes (shouldn't happen if mem has concepts)
            return RetrievalBundle(
                problem_uid=problem.uid, hint_text=None, retrieved_items=[],
                metadata={"retriever": self.name, "reason": "empty_graph"},
            )

        pagerank_scores = nx.pagerank(
            G,
            alpha=self.damping,
            personalization=reset_clipped,
            weight="weight",
        )

        # Sort + take top-k
        ranked = sorted(pagerank_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_names = [n for n, _ in ranked[: max(self.top_k, 0)]]

        hint = mem.to_string(
            concept_names=top_names,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            usage_threshold=self.usage_threshold,
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n, "ppr_score": float(pagerank_scores[n])} for n in top_names],
            metadata={
                "retriever": self.name,
                "scoring_mode": "ppr",
                "top_k": self.top_k,
                "damping": self.damping,
                "num_concepts_total": len(mem.concepts),
                "num_selected": len(top_names),
                "num_graph_edges": G.number_of_edges(),
                "fact_graph_edges_used": fact_graph_edges_used,
                "entity_graph_edges_used": entity_graph_edges_used,
                "num_reset_seeds": sum(1 for v in reset_clipped.values() if v > 0),
                "num_fact_reset_matches": fact_reset_matches,
                "num_entity_reset_matches": entity_reset_matches,
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

    def _fact_reset_weights(
        self,
        mem: ConceptMemory,
        q_tokens: set[str],
    ) -> tuple[dict[str, float], int]:
        if not q_tokens:
            return {}, 0
        weights: dict[str, float] = {}
        matches = 0
        for fact in load_openie_facts(valid_concepts=mem.concepts.keys()):
            text = " ".join(
                str(fact.get(k) or "")
                for k in ("subject", "predicate", "object", "supporting_text")
            )
            overlap = len(q_tokens & _tokenize(text))
            if overlap < self.min_reset_overlap:
                continue
            matches += 1
            linked = fact.get("linked_concepts") or []
            for name in linked:
                if name in mem.concepts:
                    weights[name] = weights.get(name, 0.0) + float(overlap)
        return weights, matches

    def _entity_reset_weights(
        self,
        mem: ConceptMemory,
        q_tokens: set[str],
    ) -> tuple[dict[str, float], int]:
        if not q_tokens:
            return {}, 0
        weights: dict[str, float] = {}
        matches = 0
        entity_graph = load_entity_graph(valid_concepts=mem.concepts.keys())
        for entity in entity_graph.get("entities") or []:
            source = entity.get("source_concept")
            if not isinstance(source, str) or source not in mem.concepts:
                continue
            text = " ".join([
                str(entity.get("mention_text") or ""),
                str(entity.get("entity_type") or ""),
                str(entity.get("supporting_text") or ""),
                " ".join(str(v) for v in (entity.get("attributes") or {}).values()),
            ])
            overlap = len(q_tokens & _tokenize(text))
            if overlap < self.min_reset_overlap:
                continue
            matches += 1
            weights[source] = weights.get(source, 0.0) + float(overlap)
        return weights, matches
