"""ConceptGraph: graph substrate shared by reorg (write-side), hierarchical
retrieval (read-side structure), and RRMC-style probing (read-side policy).

Nodes are concept names (keys into a ConceptMemory). Edges carry a single
`kind` plus a scalar weight:

  - co-activation: two concepts appeared on the same problem (`used_in` overlap)
  - embedding-sim: similarity of rendered concept text (cached externally)
  - authorship-lineage: parent/child when a concept was produced by reorg
    aggregation (parent nodes are the popular aggregates)

Design notes:
  - The graph does not own concept storage; it is an index over ConceptMemory.
    Rebuild by calling ``build_from_memory(mem)`` whenever the underlying
    memory changes.
  - All persistence goes through ``to_payload`` / ``from_payload`` so that the
    graph can round-trip through ``MemoryState.payload``.
  - Nothing here calls an LLM or touches the network. Embedding-similarity
    edges are optional and accept a precomputed callable.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from mem2.concepts.memory import ConceptMemory

EdgeKind = str  # "co_activation" | "embedding_sim" | "authorship_lineage"


@dataclass
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class ConceptGraph:
    """Undirected-by-kind concept graph.

    Co-activation and embedding-sim edges are undirected (stored canonically
    with src <= dst). Authorship-lineage edges are directed (parent → child).
    """

    def __init__(self) -> None:
        self.nodes: set[str] = set()
        # (src, dst, kind) -> Edge ; canonical ordering enforced per-kind
        self._edges: dict[tuple[str, str, EdgeKind], Edge] = {}

    # ----------------------------------------------------------------- #
    #                             Construction                          #
    # ----------------------------------------------------------------- #
    def add_node(self, name: str) -> None:
        self.nodes.add(name)

    def add_edge(
        self,
        src: str,
        dst: str,
        kind: EdgeKind,
        weight: float = 1.0,
        *,
        directed: bool | None = None,
        metadata: dict | None = None,
    ) -> None:
        if src == dst:
            return
        self.nodes.add(src)
        self.nodes.add(dst)
        if directed is None:
            directed = kind == "authorship_lineage"
        if not directed:
            src, dst = _sorted_pair(src, dst)
        key = (src, dst, kind)
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = Edge(
                src=src, dst=dst, kind=kind, weight=weight, metadata=metadata or {}
            )
        else:
            # accumulate weight (useful for co-activation counts)
            existing.weight += weight
            if metadata:
                existing.metadata.update(metadata)

    # ----------------------------------------------------------------- #
    #                         Edge-kind builders                        #
    # ----------------------------------------------------------------- #
    def add_co_activation_edges(self, mem: ConceptMemory, *, min_overlap: int = 1) -> None:
        """One +1 per shared problem_id in `used_in`. Weight = shared count."""
        by_problem: dict[str, list[str]] = defaultdict(list)
        for name, concept in mem.concepts.items():
            self.nodes.add(name)
            for pid in concept.used_in:
                by_problem[pid].append(name)
        for pid, names in by_problem.items():
            # unordered pairs
            names = sorted(set(names))
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    self.add_edge(a, b, kind="co_activation", weight=1.0)
        if min_overlap > 1:
            to_drop = [
                k for k, e in self._edges.items()
                if e.kind == "co_activation" and e.weight < min_overlap
            ]
            for k in to_drop:
                del self._edges[k]

    def add_embedding_edges(
        self,
        mem: ConceptMemory,
        embed_fn: Callable[[str], list[float]],
        *,
        threshold: float = 0.7,
        max_per_node: int = 8,
    ) -> None:
        """Cosine-similarity edges over rendered concept descriptions.

        ``embed_fn`` must be deterministic; callers are expected to cache.
        Only edges with sim >= threshold are kept, and each node keeps its
        top-``max_per_node`` neighbors.
        """
        names = sorted(mem.concepts.keys())
        if not names:
            return
        vecs: dict[str, list[float]] = {}
        for n in names:
            text = mem.concepts[n].to_string(include_description=True)
            vecs[n] = embed_fn(text)
            self.nodes.add(n)

        def cos(u: list[float], v: list[float]) -> float:
            import math
            du = math.sqrt(sum(x * x for x in u)) or 1.0
            dv = math.sqrt(sum(x * x for x in v)) or 1.0
            return sum(x * y for x, y in zip(u, v)) / (du * dv)

        per_node: dict[str, list[tuple[float, str]]] = {n: [] for n in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                s = cos(vecs[a], vecs[b])
                if s < threshold:
                    continue
                per_node[a].append((s, b))
                per_node[b].append((s, a))
        for n, neigh in per_node.items():
            neigh.sort(reverse=True)
            for s, other in neigh[:max_per_node]:
                self.add_edge(n, other, kind="embedding_sim", weight=float(s))

    def add_authorship_lineage(self, parent: str, child: str, *, weight: float = 1.0) -> None:
        """Directed edge: parent → child. Parent is the aggregate, child is the
        concept that contributed to the aggregation."""
        self.add_edge(parent, child, kind="authorship_lineage",
                      weight=weight, directed=True)

    # ----------------------------------------------------------------- #
    #                              Queries                              #
    # ----------------------------------------------------------------- #
    def neighbors(
        self, node: str, *, kinds: Iterable[EdgeKind] | None = None,
    ) -> list[tuple[str, EdgeKind, float]]:
        out: list[tuple[str, EdgeKind, float]] = []
        kind_filter = set(kinds) if kinds else None
        for (src, dst, kind), edge in self._edges.items():
            if kind_filter and kind not in kind_filter:
                continue
            if kind == "authorship_lineage":
                if src == node:
                    out.append((dst, kind, edge.weight))
            else:
                if src == node:
                    out.append((dst, kind, edge.weight))
                elif dst == node:
                    out.append((src, kind, edge.weight))
        return out

    def degree(self, node: str, *, kinds: Iterable[EdgeKind] | None = None) -> float:
        """Weighted degree — used by reorg to pick 'popular' nodes."""
        return sum(w for _, _, w in self.neighbors(node, kinds=kinds))

    def edges(self) -> list[Edge]:
        return list(self._edges.values())

    def num_edges(self) -> int:
        return len(self._edges)

    def num_nodes(self) -> int:
        return len(self.nodes)

    # ----------------------------------------------------------------- #
    #                            Persistence                            #
    # ----------------------------------------------------------------- #
    def to_payload(self) -> dict:
        return {
            "nodes": sorted(self.nodes),
            "edges": [asdict(e) for e in self._edges.values()],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ConceptGraph":
        g = cls()
        for n in payload.get("nodes", []):
            g.add_node(str(n))
        for raw in payload.get("edges", []):
            g.add_edge(
                src=raw["src"],
                dst=raw["dst"],
                kind=raw["kind"],
                weight=float(raw.get("weight", 1.0)),
                directed=(raw["kind"] == "authorship_lineage"),
                metadata=raw.get("metadata") or {},
            )
        return g

    @classmethod
    def build_from_memory(
        cls,
        mem: ConceptMemory,
        *,
        embed_fn: Callable[[str], list[float]] | None = None,
        embed_threshold: float = 0.7,
        embed_max_per_node: int = 8,
        min_co_overlap: int = 1,
        load_typed_edges: bool = True,
    ) -> "ConceptGraph":
        """Convenience: build a graph from a ConceptMemory using co-activation
        (always) + embedding-sim (if ``embed_fn`` provided) + typed semantic
        edges from the prereq concept_graph_v1.json (when present and
        ``load_typed_edges`` is True).

        Authorship-lineage edges are added by the reorg builder at reorg time.
        """
        g = cls()
        for name in mem.concepts.keys():
            g.add_node(name)
        g.add_co_activation_edges(mem, min_overlap=min_co_overlap)
        if embed_fn is not None:
            g.add_embedding_edges(
                mem, embed_fn,
                threshold=embed_threshold,
                max_per_node=embed_max_per_node,
            )
        if load_typed_edges:
            g._maybe_load_typed_edges(mem)
        return g

    def _maybe_load_typed_edges(self, mem: ConceptMemory) -> None:
        """Augment the graph with typed semantic edges from the prereq file
        ``data/arc_agi/concept_memory/concept_graph_v1.json`` if present.

        Edges loaded as kind in {"uses", "is_a", "specializes",
        "opposite_of", "composed_of"} — preserving the relation type
        from the LLM-extracted graph rather than collapsing into
        co-activation. Retrievers that consume these edges (axis-1 graph
        retrievers, axis-2 graph-MDL reorg) get a richer substrate.

        Silent fallback if file is absent or malformed — co-activation
        graph still works.
        """
        import json
        from pathlib import Path
        # Resolve repo root via this module's location.
        repo_root = Path(__file__).resolve().parents[3]  # mem2/
        graph_path = repo_root / "data" / "arc_agi" / "concept_memory" / "concept_graph_v1.json"
        if not graph_path.exists():
            return
        try:
            data = json.loads(graph_path.read_text())
        except Exception:
            return
        valid_names = set(mem.concepts.keys())
        for e in data.get("edges", []) or []:
            src = e.get("src")
            tgt = e.get("tgt")
            relation = e.get("relation")
            weight = float(e.get("weight", 1.0))
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            if src not in valid_names or tgt not in valid_names:
                continue
            if not isinstance(relation, str):
                continue
            # Add as a typed edge — kind = the relation string.
            # Most relations are directional in semantic intent
            # (uses, is_a, specializes, composed_of); opposite_of is
            # symmetric. For graph traversal we treat all as
            # undirected by default (kind != authorship_lineage).
            self.add_edge(src, tgt, kind=relation, weight=weight)
