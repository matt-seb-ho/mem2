"""RAPTOR hierarchical tree retriever — axis B.6.

Port of the core mechanism from RAPTOR (Sarthi et al., ICLR'24).

Paper: literature/2401.18059.pdf
Repo:  third_party/raptor/
Specifically ported:
    - `raptor/cluster_tree_builder.py::ClusterTreeBuilder.construct_tree`
      — the cluster → summarize → create parent node recursion.
    - `raptor/tree_retriever.py::TreeRetriever.retrieve_information_collapse_tree`
      — the "flatten nodes + top-k by query similarity" retrieve path.

Deliberate simplifications (LLM-free + embedding-free, per the mem2
retrieve-time contract):
    - **Clustering:** UMAP + GMM (paper) → Louvain community detection over
      co-activation edges (local). Principle is the same: group concepts
      that *co-occur across problems*. Needs `networkx.community`.
    - **Cluster summarization:** LLM (paper) → template-concat of member
      names + descriptions. Creates virtual "pseudo-parent" entries under
      names like `cluster_<seed>`, not written back to ConceptMemory.
    - **Tree depth:** multi-layer recursion (paper) → single 2-layer tree
      (leaves + communities). Multi-layer would need embedding-sim on the
      pseudo-parents which we don't have.
    - **Retrieval scoring:** cosine similarity over embeddings (paper) →
      token-overlap vs query text (same trick `hipporag_ppr` uses for
      determinism-on-seed and no-LLM-at-retrieve).
    - **Top-k mixing:** hybrid scoring across leaves + pseudo-parents; a
      `parent_ratio` knob controls how many top-k slots pseudo-parents
      may claim (default 0.4).

Interface: matches `ps_topk` + `hipporag_ppr`.
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
class _Cluster:
    """Virtual pseudo-parent: seed concept + community members + summary text."""

    seed: str
    members: list[str]
    summary_tokens: set[str] = field(default_factory=set)
    community_id: str = ""
    summary_text: str = ""
    summary_source: str = "template"

    @property
    def label(self) -> str:
        return f"cluster_{self.seed}"


class RAPTORRetriever:
    """Hierarchical retriever: leaves (concepts) + community-summarized parents.

    Two-layer tree:
      - Level 0 (leaves): individual concepts from ConceptMemory.
      - Level 1 (parents): Louvain communities on the co-activation graph,
        each labeled by its highest-degree seed, "summary" = concat of
        member descriptions.

    Query-time scoring: token overlap between the ARC problem text and
    (leaf / parent) rendered text. Top-k mixes both levels via
    `parent_ratio`.
    """

    name = "raptor"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 3,
        parent_ratio: float = 0.4,
        min_community_size: int = 2,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
        community_summaries_path: str | Path | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.parent_ratio = float(parent_ratio)
        self.min_community_size = int(min_community_size)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)
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
                "raptor retriever requires networkx; install with `pip install networkx`."
            ) from exc

        # Build ConceptGraph; project to networkx for community detection
        concept_graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        G = nx.Graph()
        for n in mem.concepts.keys():
            G.add_node(n)
        for edge in concept_graph.edges():
            if edge.kind != "co_activation":
                continue
            G.add_edge(edge.src, edge.dst, weight=float(edge.weight or 1.0))

        # Level-1: community detection via Louvain, or the shared LLM summary
        # artifact when present.
        summary_source = "template"
        artifact_communities = load_community_summaries(
            self.community_summaries_path,
            valid_concepts=mem.concepts.keys(),
        )
        clusters: list[_Cluster] = []
        if artifact_communities:
            summary_source = "llm_summaries_v1"
            raw_communities = [set(raw["member_concepts"]) for raw in artifact_communities]
            for raw in artifact_communities:
                members = list(raw["member_concepts"])
                if len(members) < self.min_community_size:
                    continue
                seed = raw["seed_concept"]
                summary = raw["llm_summary"]
                summary_tokens = _tokenize(" ".join([
                    raw["community_id"], seed, summary, raw.get("member_digest") or "", *members,
                ]))
                clusters.append(_Cluster(
                    seed=seed,
                    members=members,
                    summary_tokens=summary_tokens,
                    community_id=raw["community_id"],
                    summary_text=summary,
                    summary_source=summary_source,
                ))
        else:
            raw_communities = list(nx_community.louvain_communities(G, seed=int(getattr(ctx, "seed", 0))))
            for comm in raw_communities:
                if len(comm) < self.min_community_size:
                    continue
                # Pick seed = highest-degree member in this community
                seed = max(comm, key=lambda n: G.degree(n) if n in G else 0)
                members = sorted(comm, key=lambda n: (G.degree(n) if n in G else 0, n), reverse=True)
                summary_text_parts = []
                for m in members:
                    c = mem.concepts.get(m)
                    if c is not None and c.description:
                        summary_text_parts.append(c.description)
                    summary_text_parts.append(m)
                summary_tokens = _tokenize(" ".join(summary_text_parts))
                clusters.append(_Cluster(
                    seed=seed,
                    members=members,
                    summary_tokens=summary_tokens,
                    community_id=f"cluster_{seed}",
                    summary_source=summary_source,
                ))

        # Query-time scoring: token overlap
        q_tokens = _tokenize(_problem_text(problem))
        if not q_tokens:
            # Fall back to frequency-based ranking over leaves
            ranked = sorted(
                mem.concepts.values(), key=lambda c: (len(c.used_in or []), c.name), reverse=True,
            )
            top_names = [c.name for c in ranked[: self.top_k]]
            return _render_bundle(
                self, problem, mem, top_names,
                metadata={"reason": "empty_query_tokens", "summary_source": summary_source},
                graph_edges=G.number_of_edges(), n_clusters=len(clusters),
            )

        leaf_scores: list[tuple[str, int]] = []
        for name, concept in mem.concepts.items():
            c_tokens = _tokenize(concept.to_string(include_description=True))
            leaf_scores.append((name, len(q_tokens & c_tokens)))

        parent_scores: list[tuple[_Cluster, int]] = []
        for cl in clusters:
            parent_scores.append((cl, len(q_tokens & cl.summary_tokens)))

        leaf_scores.sort(key=lambda kv: kv[1], reverse=True)
        parent_scores.sort(key=lambda kv: kv[1], reverse=True)

        # Top-k mixing: reserve `parent_ratio` × top_k slots for pseudo-parents.
        # A pseudo-parent "occupies" its seed leaf (we render the seed, not a
        # synthetic concept that isn't in ConceptMemory).
        n_parent_slots = max(0, min(self.top_k, int(round(self.parent_ratio * self.top_k))))
        n_leaf_slots = self.top_k - n_parent_slots

        selected: list[str] = []
        selected_clusters: list[_Cluster] = []
        used_leaves: set[str] = set()
        for cl, score in parent_scores[:n_parent_slots]:
            if score <= 0:
                break
            if cl.seed not in used_leaves and cl.seed in mem.concepts:
                selected.append(cl.seed)
                selected_clusters.append(cl)
                used_leaves.add(cl.seed)
        for name, score in leaf_scores:
            if len(selected) >= self.top_k:
                break
            if name in used_leaves:
                continue
            selected.append(name)
            used_leaves.add(name)

        for name, score in leaf_scores:
            if len(selected) >= self.top_k:
                break
            if name not in used_leaves:
                selected.append(name)
                used_leaves.add(name)

        return _render_bundle(
            self, problem, mem, selected,
            clusters=selected_clusters,
            metadata={
                "n_clusters": len(clusters),
                "n_parent_slots": n_parent_slots,
                "n_leaf_slots": n_leaf_slots,
                "summary_source": summary_source,
            },
            graph_edges=G.number_of_edges(),
            n_clusters=len(clusters),
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


def _render_bundle(
    r: "RAPTORRetriever",
    problem: ProblemSpec,
    mem: ConceptMemory,
    selected: list[str],
    *,
    clusters: list[_Cluster] | None = None,
    metadata: dict,
    graph_edges: int,
    n_clusters: int,
) -> RetrievalBundle:
    hint = mem.to_string(
        concept_names=selected,
        include_description=r.include_description,
        skip_cues=r.skip_cues,
        skip_implementation=r.skip_implementation,
        usage_threshold=r.usage_threshold,
    )
    if clusters:
        cluster_blocks: list[str] = []
        for cl in clusters:
            member_descs = []
            for m in cl.members:
                c = mem.concepts.get(m)
                if c and c.description:
                    member_descs.append(f"  - {m}: {c.description}")
            summary_text = f"Summary: {cl.summary_text}\n" if cl.summary_text else ""
            block = (
                f"\n[Hierarchical summary: {cl.label}]\n"
                f"{summary_text}"
                f"Members: {', '.join(cl.members)}\n"
                + "\n".join(member_descs)
            )
            cluster_blocks.append(block)
        if cluster_blocks:
            hint = (hint or "") + "\n\n--- Hierarchical cluster summaries ---" + "".join(cluster_blocks)
    meta = {
        "retriever": r.name,
        "scoring_mode": "raptor_summary",
        "top_k": r.top_k,
        "parent_ratio": r.parent_ratio,
        "num_concepts_total": len(mem.concepts),
        "num_selected": len(selected),
        "num_graph_edges": graph_edges,
        "num_clusters": n_clusters,
        "summary_source": (
            clusters[0].summary_source if clusters else (metadata or {}).get("summary_source", "template")
        ),
    }
    meta.update(metadata or {})
    items: list[dict] = [{"name": n} for n in selected]
    if clusters:
        items.extend(
            {"name": cl.label, "type": "cluster_summary", "members": cl.members}
            for cl in clusters
        )
    return RetrievalBundle(
        problem_uid=problem.uid,
        hint_text=hint or None,
        retrieved_items=items,
        metadata=meta,
    )
