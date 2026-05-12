"""LightRAG dual-level retriever — axis B.7.

Port of the dual-level retrieval pattern from LightRAG (Guo et al.).

Paper: literature/2410.05779.pdf
Repo:  third_party/lightrag/ (entry: lightrag/operate.py::kg_query)

Specifically ported:
    - Dual-level retrieval: "local" (entity-level) + "global" (relationship-level),
      merged into a single context.

Deliberate simplifications (LLM-free, embed-free):
    - Entity scoring: paper uses entity-embedding cosine sim → token overlap.
    - Relationship scoring: paper uses per-edge LLM summary + embedding sim →
      edge score = product of endpoint token-overlap times edge weight.
    - No vector DBs, no LLM summary generation.
    - Hint shape: two blocks, `## entities (local)` + `## relationships (global)`,
      distinguishing this axis-B condition from node-only or community-only
      retrievers.
"""
from __future__ import annotations

import re

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
    parts: list[str] = []
    if problem.metadata:
        parts.extend(v for v in problem.metadata.values() if isinstance(v, str))
    for p in (problem.train_pairs or []):
        if isinstance(p, dict):
            parts.extend(v for v in p.values() if isinstance(v, str))
    for p in (problem.test_pairs or []):
        if isinstance(p, dict):
            parts.extend(v for v in p.values() if isinstance(v, str))
    parts.append(problem.uid)
    return "\n".join(parts)


class LightRAGRetriever:
    """Dual-level: top-k entities (concepts) + top-m relationships (edges)."""

    name = "lightrag"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k_entities: int = 3,
        top_m_relationships: int = 3,
        min_edge_weight: float = 1.0,
    ) -> None:
        self.top_k_entities = int(top_k_entities)
        self.top_m_relationships = int(top_m_relationships)
        self.min_edge_weight = float(min_edge_weight)

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

        q_tokens = _tokenize(_problem_text(problem))

        # --- Local (entity-level) scoring -------------------------------
        entity_scores: dict[str, int] = {}
        for name, concept in mem.concepts.items():
            c_tokens = _tokenize(concept.to_string(include_description=True))
            entity_scores[name] = len(q_tokens & c_tokens)
        ranked_entities = sorted(entity_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_entities = [name for name, _ in ranked_entities[: self.top_k_entities]]

        # --- Global (relationship-level) scoring ------------------------
        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        edge_scores: list[tuple[str, str, float]] = []
        for edge in graph.edges():
            if edge.kind != "co_activation":
                continue
            if edge.weight < self.min_edge_weight:
                continue
            e1_score = entity_scores.get(edge.src, 0)
            e2_score = entity_scores.get(edge.dst, 0)
            if e1_score == 0 and e2_score == 0:
                continue
            # product of endpoint relevance × edge weight; add 1 to endpoint
            # scores to avoid killing asymmetric edges where one endpoint
            # has zero overlap.
            combined = float((e1_score + 1) * (e2_score + 1) * edge.weight)
            edge_scores.append((edge.src, edge.dst, combined))
        edge_scores.sort(key=lambda t: t[2], reverse=True)
        top_edges = edge_scores[: self.top_m_relationships]

        # --- Render hint ------------------------------------------------
        lines: list[str] = []
        if top_entities:
            lines.append("## entities (local)")
            for name in top_entities:
                c = mem.concepts.get(name)
                desc = (c.description or "") if c else ""
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        if top_edges:
            lines.append("")
            lines.append("## relationships (global)")
            for src, dst, score in top_edges:
                lines.append(f"- {src} ↔ {dst}  (co-activation strength {score:.1f})")
        hint = "\n".join(lines) if lines else None

        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint,
            retrieved_items=[
                *[{"type": "entity", "name": n} for n in top_entities],
                *[{"type": "edge", "src": s, "dst": d, "score": sc} for s, d, sc in top_edges],
            ],
            metadata={
                "retriever": self.name,
                "scoring_mode": "lightrag_dual",
                "top_k_entities": self.top_k_entities,
                "top_m_relationships": self.top_m_relationships,
                "num_concepts_total": len(mem.concepts),
                "num_edges_considered": len(graph.edges()),
                "num_edges_ranked": len(edge_scores),
                "num_entities_selected": len(top_entities),
                "num_edges_selected": len(top_edges),
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
