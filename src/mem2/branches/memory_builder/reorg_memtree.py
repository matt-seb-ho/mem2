"""MemTree hierarchical-schema memory — axis A.9.

Port of MemTree (Rezazadeh et al., ICLR 2025; arxiv 2410.14052).

Paper: literature/2410.14052.pdf
Repo:  no public official implementation; ported from paper §3 Method.

Specifically ported:
    - The dynamic TREE schema: each node has [content, embedding, parent,
      children, depth]. On new info, walk from root; at each node, compare
      new content against existing children's content; if similar (above a
      threshold), route INTO that child; otherwise, create a new leaf under
      the current node.
    - Ancestor aggregation: a parent node's content is the summarization of
      all its descendants, built on-the-fly as the tree grows.
    - Online update semantics (vs RAPTOR/GraphRAG's offline batch rebuild).

Deliberate simplifications (no embedding model, paper-only port):
    - "Similarity" uses ConceptGraph co-activation as a cheap proxy for
      embedding similarity. Co-activation correlates with "concepts used on
      the same problems" which is a reasonable proxy for semantic closeness
      in the ARC domain.
    - "Ancestor content summary" is the concatenation of descendant
      descriptions (truncated) rather than an LLM-generated summary. LLM
      mode (via `_meta_edit_provider`) would upgrade this to true
      summarization, matching the paper's LLM-based aggregation.
    - Tree is rebuilt each consolidate round (not strictly online) because
      mem2's `consolidate` is the natural batch boundary. The paper's true
      innovation — no-full-rebuild updates — maps cleanly to mem2 only when
      we track incremental additions, which we do via `used_in` and the
      reorg step counter.

A.9 vs A.1 / A.2 / A.3 / A.4 / A.6 / A.7:
    - All previous A.x are FLAT structures (single aggregate, or line-level
      fragments, or per-note rewiring).
    - A.9 is genuinely HIERARCHICAL: root → kind-group nodes → concept
      leaves, with ancestor summaries at every level. A retrieval step
      benefits from knowing BOTH leaf-level detail and ancestor-level
      summaries — the paper's core contribution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.data import Concept
from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    name: str
    content: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    descendant_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "content": self.content[:500],
            "parent": self.parent, "children": list(self.children),
            "depth": self.depth, "descendants": len(self.descendant_names),
        }


class MemTreeHierarchicalBuilder(ArcMemoReorgMemoryBuilder):
    """Hierarchical tree over concepts with ancestor summaries."""

    name = "reorg_memtree"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        similarity_threshold: float = 1.0,
        max_children_per_node: int = 6,
        max_depth: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.similarity_threshold = float(similarity_threshold)
        self.max_children_per_node = int(max_children_per_node)
        self.max_depth = int(max_depth)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        if not mem.concepts:
            return memory

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        provider = self._resolve_provider(ctx)

        tree: dict[str, TreeNode] = {}
        root_name = "__memtree_root__"
        tree[root_name] = TreeNode(
            name=root_name, content="MemTree root", parent=None, depth=0,
        )

        # Level 1: kind-group nodes.
        for kind in sorted(mem.categories.keys()):
            if not mem.categories[kind]:
                continue
            kind_node = f"__kind:{kind}__"
            tree[kind_node] = TreeNode(
                name=kind_node,
                content=f"Kind group: {kind}",
                parent=root_name, depth=1,
            )
            tree[root_name].children.append(kind_node)

        # Level 2+: route each concept into its kind-subtree; within a kind,
        # either attach as direct child OR further subdivide via co-activation
        # (paper: "similar to existing leaf" → merge; else → new leaf).
        for name, concept in mem.concepts.items():
            kind_node = f"__kind:{concept.name}__" if False else f"__kind:{concept.kind}__"
            if kind_node not in tree:
                tree[kind_node] = TreeNode(
                    name=kind_node,
                    content=f"Kind group: {concept.kind}",
                    parent=root_name, depth=1,
                )
                tree[root_name].children.append(kind_node)

            # Find best-match within the kind-subtree.
            kind_children = [
                ch for ch in tree[kind_node].children if ch in tree
            ]
            best_match = self._find_similar_child(
                name, kind_children, graph, tree,
            )
            if best_match is not None and tree[best_match].depth < self.max_depth:
                # Route INTO the existing subtree: attach as grandchild.
                if len(tree[best_match].children) < self.max_children_per_node:
                    tree[name] = TreeNode(
                        name=name, content=concept.description or "",
                        parent=best_match,
                        depth=tree[best_match].depth + 1,
                    )
                    tree[best_match].children.append(name)
                else:
                    # Existing subtree full → fallback: attach to kind parent.
                    tree[name] = TreeNode(
                        name=name, content=concept.description or "",
                        parent=kind_node, depth=2,
                    )
                    tree[kind_node].children.append(name)
            else:
                # No similar existing child → new leaf under kind-group.
                tree[name] = TreeNode(
                    name=name, content=concept.description or "",
                    parent=kind_node, depth=2,
                )
                tree[kind_node].children.append(name)

        # Compute descendant lists + ancestor summaries (bottom-up).
        self._compute_descendants(tree, root_name)
        self._summarize_ancestors(tree, mem, provider)

        # Write the hierarchy into memory payload as a sidecar structure.
        hierarchy_payload = {n: node.to_dict() for n, node in tree.items()}
        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "memtree_hierarchical_build",
            "tree_size": len(tree),
            "max_depth_reached": max(node.depth for node in tree.values()),
            "used_llm": provider is not None,
            "kind_groups": len(tree[root_name].children),
            "leaf_count": sum(
                1 for n in tree.values() if not n.children and n.name != root_name
            ),
        })
        new_payload["reorg"] = reorg
        new_payload["memtree_hierarchy"] = hierarchy_payload
        memory.payload = new_payload
        return memory

    # ----------------------------------------------------------------- #
    def _resolve_provider(self, ctx: RunContext):
        try:
            return (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            return None

    def _find_similar_child(
        self, new_name: str, children: list[str],
        graph: ConceptGraph, tree: dict[str, TreeNode],
    ) -> str | None:
        """Return child name with highest co-activation to new_name, OR None
        if none exceeds threshold."""
        best_name: str | None = None
        best_score = 0.0
        for ch in children:
            if ch.startswith("__"):
                continue  # skip system nodes
            # Sum of co-activation weights between new_name and ch.
            score = 0.0
            for dst, kind, w in graph.neighbors(new_name, kinds=["co_activation"]):
                if dst == ch:
                    score += float(w)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_name = ch
        return best_name

    def _compute_descendants(
        self, tree: dict[str, TreeNode], root: str,
    ) -> list[str]:
        """Populate `descendant_names` bottom-up."""
        node = tree[root]
        descendants: list[str] = []
        for child in node.children:
            if child in tree:
                sub = self._compute_descendants(tree, child)
                descendants.extend(sub)
                descendants.append(child)
        node.descendant_names = descendants
        return descendants

    def _summarize_ancestors(
        self, tree: dict[str, TreeNode], mem: ConceptMemory, provider,
    ) -> None:
        """For each internal node, aggregate descendant content.

        Template mode: join first 80 chars of each concept-leaf description.
        LLM mode: call provider to summarize descendant text blob.
        """
        for node in tree.values():
            if node.name.startswith("__"):
                # System node (root or kind-group); aggregate concept children.
                concept_descendants = [
                    d for d in node.descendant_names if not d.startswith("__")
                ]
                if not concept_descendants:
                    continue
                snippets = []
                for cn in concept_descendants[:8]:
                    c = mem.concepts.get(cn)
                    if c and c.description:
                        snippets.append(f"{cn}: {c.description[:80]}")
                blob = " | ".join(snippets)
                if provider is not None:
                    try:
                        prompt = (
                            "Summarize this set of concepts in under 200 chars. "
                            f"Concepts: {blob}\n\nSummary:"
                        )
                        out = provider.generate(prompt, model=getattr(provider, "model", ""))
                        if out and isinstance(out[0], str):
                            node.content = out[0][:500]
                            continue
                    except Exception as exc:  # pragma: no cover
                        logger.warning("memtree LLM summary failed: %s", exc)
                # Fall through to template.
                node.content = f"[{len(concept_descendants)} concepts] {blob}"[:500]
