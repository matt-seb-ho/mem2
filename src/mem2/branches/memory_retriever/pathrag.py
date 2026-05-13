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

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from mem2.concepts.artifacts import CONCEPT_MEMORY_DIR
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
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "pathrag_memory_v1.json"


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
        edge_kinds: tuple[str, ...] = ("co_activation", "openie_fact", "entity_relation"),
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = True,
        usage_threshold: int = 0,
        adapted_memory_path: str | Path | None = None,
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
        self.adapted_memory_path = self._resolve_path(
            adapted_memory_path,
            _DEFAULT_ADAPTED_MEMORY_PATH,
        )

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
        adapted_records, adapted_source = self._load_adapted_records(mem)

        graph = ConceptGraph.build_from_memory(
            mem,
            min_co_overlap=1,
            load_openie_edges=True,
            load_entity_edges=True,
        )

        q_toks = self._query_toks(problem, previous_attempts)

        # Stage 1 — seed nodes by query token overlap.
        scored: list[tuple[float, str]] = []
        for name, c in mem.concepts.items():
            doc_toks = {name.lower()}
            if name in adapted_records:
                doc_toks.update(t.lower() for t in WORD_RE.findall(
                    self._adapted_record_text(adapted_records[name])
                ))
            else:
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
        adapted_path_block = self._render_adapted_paths(selected_nodes, adapted_records)
        blocks = [block for block in (path_block, adapted_path_block, base_hint) if block]
        hint = "\n\n".join(blocks)

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
                "adapted_memory_source": adapted_source,
                "adapted_records_loaded": len(adapted_records),
                "adapted_paths_rendered": self._count_renderable_adapted_paths(
                    selected_nodes,
                    adapted_records,
                ),
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
            edge = graph.edge_between(src, dst, kinds=("entity_relation", "openie_fact", "co_activation"))
            if edge and edge.kind == "entity_relation":
                label = str((edge.metadata or {}).get("edge_type") or "entity_relates_to").strip()
                parts.append(f"--{label}--")
            if edge and edge.kind == "openie_fact":
                predicate = str((edge.metadata or {}).get("predicate") or "relates_to").strip()
                parts.append(f"--{predicate}--")
            elif not edge or edge.kind != "entity_relation":
                parts.append("->")
            parts.append(dst)
        return " ".join(parts)

    @staticmethod
    def _resolve_path(path: str | Path | None, default: Path) -> Path:
        if path is None:
            return default
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _load_adapted_records(
        self,
        mem: ConceptMemory,
    ) -> tuple[dict[str, dict], str]:
        path = self.adapted_memory_path
        if not path.exists():
            return {}, "flat"
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - corrupted local artifact should not be silent
            raise RuntimeError(f"invalid PathRAG adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != self.name:
            raise RuntimeError(f"invalid PathRAG adapted memory schema: {path}")
        records: dict[str, dict] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            paths = raw.get("entity_paths")
            if not isinstance(paths, list) or not paths:
                raise RuntimeError(f"adapted memory missing entity_paths for {concept_id}")
            records[concept_id] = raw
        if not records:
            return {}, "flat"
        return records, "pathrag_memory_v1"

    @staticmethod
    def _adapted_record_text(record: dict) -> str:
        parts: list[str] = []
        parts.extend(str(item) for item in record.get("query_keywords") or [])
        for node in record.get("path_nodes") or []:
            if isinstance(node, dict):
                parts.append(" ".join([
                    str(node.get("label") or ""),
                    str(node.get("text_chunk") or ""),
                    str(node.get("node_type") or ""),
                ]))
        for path in record.get("entity_paths") or []:
            if isinstance(path, dict):
                parts.append(str(path.get("textual_path") or ""))
                for edge in path.get("edges") or []:
                    if isinstance(edge, dict):
                        parts.append(" ".join([
                            str(edge.get("relation") or ""),
                            str(edge.get("text_chunk") or ""),
                        ]))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _count_renderable_adapted_paths(
        selected_nodes: list[str],
        adapted_records: dict[str, dict],
    ) -> int:
        return sum(
            len(adapted_records.get(name, {}).get("entity_paths") or [])
            for name in selected_nodes
        )

    def _render_adapted_paths(
        self,
        selected_nodes: list[str],
        adapted_records: dict[str, dict],
    ) -> str:
        rows: list[tuple[float, str, str]] = []
        for name in selected_nodes:
            record = adapted_records.get(name)
            if not record:
                continue
            for path in record.get("entity_paths") or []:
                if not isinstance(path, dict):
                    continue
                text = str(path.get("textual_path") or "").strip()
                if not text:
                    continue
                try:
                    reliability = float(path.get("reliability_hint", 0.0))
                except (TypeError, ValueError):
                    reliability = 0.0
                rows.append((reliability, name, text))
        if not rows:
            return ""
        rows.sort(key=lambda item: item[0])
        rendered = ["## adapted PathRAG relational paths (ascending reliability)"]
        for reliability, name, text in rows[-self.max_paths_rendered:]:
            rendered.append(f"  [{reliability:.3f}] {name}: {text}")
        return "\n".join(rendered)
