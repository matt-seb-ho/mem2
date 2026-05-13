"""GraphRAG community-summary retriever — axis B.3.

Port of the "global search" flavor from Microsoft GraphRAG (Edge et al.).

Paper: literature/2404.16130.pdf
Repo:  third_party/graphrag/ (entry: packages/graphrag/graphrag/query/structured_search/global_search/)

Specifically ported:
    - The community-report retrieval pattern: score community reports
      against the query, return top-k reports as the hint.

Deliberate simplifications (LLM-free, embed-free at retrieve time):
    - Community detection: hierarchical Leiden (paper) → Louvain (here).
    - Community report: LLM-summarized (paper) → template concat of member
      (name, description) pairs.
    - Scoring: embedding cosine (paper) → token-overlap vs query text.
    - No map-reduce "ask-LLM-per-community" flow.

Distinguishing behavior vs `raptor.py` (which also uses Louvain + template
summaries): RAPTOR returns a HYBRID of leaves + community-seed proxies.
`graphrag` returns COMMUNITY REPORTS ONLY — the hint is a block of
"cluster K: member1 (desc), member2 (desc), ..." for the top-k clusters.
This is what distinguishes the axis-B ablation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mem2.concepts.artifacts import load_community_summaries
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


@dataclass
class _CommunityReport:
    """Pseudo-LLM summary of a community: title + member list + concat text."""

    community_id: str
    title: str
    members: list[str]
    body_tokens: set[str] = field(default_factory=set)
    body_text: str = ""
    summary_source: str = "template"


class GraphRAGRetriever:
    """Global-search-style retriever: top-k community reports as the hint.

    Algorithm:
      1. Build ConceptGraph; project to networkx with co-activation edges.
      2. Louvain community detection.
      3. Per community: build a template report = "cluster_N\\n- name: description ...".
      4. Score each community's report by token-overlap with query.
      5. Return top-k community reports concatenated as the hint.
    """

    name = "graphrag"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k_communities: int = 2,
        min_community_size: int = 2,
        max_members_per_community: int = 5,
        include_description: bool = True,
        community_summaries_path: str | Path | None = None,
    ) -> None:
        self.top_k_communities = int(top_k_communities)
        self.min_community_size = int(min_community_size)
        self.max_members_per_community = int(max_members_per_community)
        self.include_description = bool(include_description)
        self.community_summaries_path = community_summaries_path

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

        try:
            import networkx as nx
            from networkx.algorithms import community as nx_community
        except ImportError as exc:
            raise RuntimeError(
                "graphrag retriever requires networkx; install with `pip install networkx`."
            ) from exc

        concept_graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        G = nx.Graph()
        for n in mem.concepts.keys():
            G.add_node(n)
        for edge in concept_graph.edges():
            if edge.kind != "co_activation":
                continue
            G.add_edge(edge.src, edge.dst, weight=float(edge.weight or 1.0))

        reports: list[_CommunityReport] = []
        summary_source = "template"
        artifact_communities = load_community_summaries(
            self.community_summaries_path,
            valid_concepts=mem.concepts.keys(),
        )
        if artifact_communities:
            summary_source = "llm_summaries_v1"
            for raw in artifact_communities:
                members = list(raw["member_concepts"])
                if len(members) < self.min_community_size:
                    continue
                community_id = raw["community_id"]
                seed = raw["seed_concept"]
                digest = raw.get("member_digest") or ""
                summary = raw["llm_summary"]
                body_text = (
                    f"## {community_id} ({seed})\n"
                    f"{summary}\n\n"
                    f"Members: {', '.join(members[: self.max_members_per_community])}"
                )
                body_tokens = _tokenize(" ".join([community_id, seed, digest, summary, *members]))
                reports.append(_CommunityReport(
                    community_id=community_id,
                    title=f"{community_id} ({seed})",
                    members=members,
                    body_tokens=body_tokens,
                    body_text=body_text,
                    summary_source=summary_source,
                ))
            communities = [set(r.members) for r in reports]
        else:
            communities = list(nx_community.louvain_communities(G, seed=int(getattr(ctx, "seed", 0))))
        for i, comm in enumerate(communities if summary_source == "template" else []):
            if len(comm) < self.min_community_size:
                continue
            # Rank members by degree desc, take top-N
            ordered_members = sorted(
                comm,
                key=lambda n: (G.degree(n) if n in G else 0, n),
                reverse=True,
            )[: self.max_members_per_community]
            seed = ordered_members[0]
            lines = [f"## community_{i} ({seed})"]
            body_text_parts = [f"community_{i}", seed]
            for m in ordered_members:
                c = mem.concepts.get(m)
                if c is None:
                    continue
                desc = c.description or ""
                lines.append(f"- {m}: {desc}" if desc else f"- {m}")
                body_text_parts.extend([m, desc])
            body_text = "\n".join(lines)
            body_tokens = _tokenize(" ".join(body_text_parts))
            reports.append(_CommunityReport(
                community_id=f"community_{i}",
                title=f"community_{i} ({seed})",
                members=ordered_members,
                body_tokens=body_tokens,
                body_text=body_text,
                summary_source=summary_source,
            ))

        # Score community reports vs query
        q_tokens = _tokenize(_problem_text(problem))
        if not q_tokens:
            # Fall back: take the largest communities
            reports.sort(key=lambda r: len(r.members), reverse=True)
        else:
            reports.sort(key=lambda r: len(q_tokens & r.body_tokens), reverse=True)

        selected = reports[: self.top_k_communities]
        if not selected:
            return RetrievalBundle(
                problem_uid=problem.uid, hint_text=None, retrieved_items=[],
                metadata={"retriever": self.name, "reason": "no_communities_above_min_size"},
            )

        # Render hint: concatenation of the selected community reports
        hint = "\n\n".join(r.body_text for r in selected)

        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"community": r.title, "members": r.members} for r in selected],
            metadata={
                "retriever": self.name,
                "scoring_mode": "graphrag_community",
                "top_k_communities": self.top_k_communities,
                "num_concepts_total": len(mem.concepts),
                "num_communities_all": len(communities),
                "num_communities_eligible": len(reports),
                "num_graph_edges": G.number_of_edges(),
                "summary_source": selected[0].summary_source if selected else summary_source,
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
