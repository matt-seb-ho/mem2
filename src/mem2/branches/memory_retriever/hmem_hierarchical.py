"""H-MEM hierarchical multi-layer retrieval — axis B.9.

Port of H-MEM (Sun & Zeng, 2025; arxiv 2507.22925).

Paper: literature/2507.22925.pdf
Repo:  no public official implementation; ported from paper §3 Method.

Specifically ported:
    - The *multi-layer routing* pattern: divide memory into layers by
      semantic abstraction (paper: Domain → Category → Trace → Episode).
      Query walks layer-by-layer: at each layer, compute similarity to
      the layer's entries, keep top-k, drop into the next layer's
      submemories pointed to by the kept entries.
    - The *position-index encoding*: each upper-layer entry stores pointers
      to its sub-memories in the next layer, enabling efficient
      layer-by-layer retrieval without exhaustive similarity computation.

Hierarchy mode (preferred, when prereq file exists):
    - Loads `data/arc_agi/concept_memory/concept_hierarchy_v1.json` (built
      by `scripts/prereq/shared/hmem_hierarchical_prereq/build_hierarchy.py`).
    - Layer 1 = the LLM-built broad themes ("spatial transformations",
      "object detection", etc.).
    - Layer 2 = sub-themes within each picked category.
    - Layer 3 = individual concepts within picked sub-themes.
    - True 3-level routing matching the paper's Domain → Cat → Trace
      → Episode shape.

Fallback mode (when prereq file absent):
    - Layer 1 = `kind` categories (routine vs structure).
    - Layer 2 = concepts grouped by shared `used_in` problems.
    - Layer 3 = individual concepts.
    - "Similarity" = token overlap between query and entry text.

B.9 vs B.4 / B.6 / A.9 MemTree:
    - B.4 HippoRAG PPR: flat-graph PPR with single-stage scoring.
    - B.6 RAPTOR: community hierarchy with leaf+summary hybrid retrieval
      (recursive Louvain, not kind-based).
    - A.9 MemTree (AXIS A, builder): hierarchical REORG that builds a
      persistent tree in the memory payload.
    - B.9 (this module, AXIS B, retriever): hierarchical RETRIEVAL — the
      hierarchy is a routing device, not a memory mutation. Layers are
      traversed fresh each query.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

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

# Resolve repo root and prereq path.
# parents: 0=memory_retriever, 1=branches, 2=mem2, 3=src, 4=mem2 root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_HIERARCHY_PATH = _REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "concept_hierarchy_v1.json"
_DEFAULT_ADAPTED_MEMORY_PATH = (
    _REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "hmem_memory_v1.json"
)

_HIERARCHY_CACHE: dict | None = None


def _load_prereq_hierarchy() -> dict | None:
    """Lazy-load the prebuilt 3-level hierarchy from disk. Returns None on absence."""
    global _HIERARCHY_CACHE
    if _HIERARCHY_CACHE is not None:
        return _HIERARCHY_CACHE if _HIERARCHY_CACHE.get("ok") else None
    if not _HIERARCHY_PATH.exists():
        _HIERARCHY_CACHE = {"ok": False}
        return None
    try:
        data = json.loads(_HIERARCHY_PATH.read_text())
        _HIERARCHY_CACHE = {"ok": True, "data": data}
        logger.info(f"hmem_hierarchical: loaded prebuilt hierarchy from {_HIERARCHY_PATH.name}")
        return _HIERARCHY_CACHE
    except Exception as e:
        logger.warning(f"hmem_hierarchical: failed to load hierarchy → fallback. {e}")
        _HIERARCHY_CACHE = {"ok": False}
        return None


def _memtree_to_prereq_shape(memtree_payload: dict) -> dict | None:
    """Convert reorg_memtree's flat {node_name: TreeNode.to_dict()} payload
    into the prereq categories→subcategories→concepts shape so the existing
    retrieval walk can consume it.

    Memtree structure: depth 0 root (structural), depth 1 kind-groups,
    depth 2 mid-nodes, depth 3 leaves. We map depth 1 → categories,
    depth 2 → subcategories, depth 3 → concepts. depth-2 nodes that are
    themselves leaves (no depth-3 children) become single-concept subcategories.
    """
    if not memtree_payload:
        return None
    by_depth: dict[int, list[tuple[str, dict]]] = {}
    for n, node in memtree_payload.items():
        d = int(node.get("depth", 0))
        by_depth.setdefault(d, []).append((n, node))
    if 1 not in by_depth:
        return None
    categories = []
    for name, node in by_depth.get(1, []):
        cat = {
            "name": name,
            "description": str(node.get("content", ""))[:200],
            "subcategories": [],
        }
        for sub_name, sub_node in by_depth.get(2, []):
            if sub_node.get("parent") != name:
                continue
            sub = {
                "name": sub_name,
                "description": str(sub_node.get("content", ""))[:200],
                "concepts": list(sub_node.get("children", [])) or [sub_name],
            }
            cat["subcategories"].append(sub)
        categories.append(cat)
    if not categories:
        return None
    return {"categories": categories}


def _resolve_hierarchy(memory: MemoryState) -> tuple[dict | None, str]:
    """Prefer fresh memtree payload when present; fall back to prereq file.

    Returns (hierarchy_data, source_label) where source_label ∈
    {'memtree_fresh', 'prebuilt_v1', 'none'}. Audit purpose: T3
    retrieval_metadata captures source_label as scoring_mode.
    """
    payload_tree = memory.payload.get("memtree_hierarchy") if memory and memory.payload else None
    if payload_tree:
        adapted = _memtree_to_prereq_shape(payload_tree)
        if adapted:
            return adapted, "memtree_fresh"
        logger.warning("hmem_hierarchical: memtree_hierarchy payload present but adapter returned empty; falling back to prereq")
    cached = _load_prereq_hierarchy()
    if cached and cached.get("ok"):
        return cached["data"], "prebuilt_v1"
    return None, "none"


def _toks(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in WORD_RE.findall(text)}


class HMEMHierarchicalRetriever:
    """Layer-by-layer routing retrieval."""

    name = "hmem_hierarchical"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 3,
        per_layer_top_k: int = 2,
        trace_group_min_overlap: int = 1,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 0,
        adapted_memory_path: str | Path | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.per_layer_top_k = int(per_layer_top_k)
        self.trace_group_min_overlap = int(trace_group_min_overlap)
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

        query_text = self._query_text(problem, previous_attempts)
        q_toks = _toks(query_text)

        adapted_hierarchy, adapted_records, adapted_source = self._load_adapted_hierarchy(mem)
        if adapted_hierarchy is not None:
            return self._retrieve_with_prebuilt_hierarchy(
                adapted_hierarchy,
                mem,
                q_toks,
                query_text,
                problem,
                source="adapted_memory_v1",
                adapted_records=adapted_records,
                adapted_memory_source=adapted_source,
            )

        # Prefer freshly-written memtree payload when present; fall back to prereq.
        hdata, hsource = _resolve_hierarchy(memory)
        if hdata is not None:
            return self._retrieve_with_prebuilt_hierarchy(
                hdata,
                mem,
                q_toks,
                query_text,
                problem,
                source=hsource,
            )

        layer_trace: list[dict] = []

        # Layer 1: Category (by kind).
        kinds = sorted(mem.categories.keys())
        kind_scores = []
        for k in kinds:
            if not mem.categories[k]:
                continue
            # Kind-entry text = kind name + concatenated first-100-chars of members.
            kind_text = k + " " + " ".join(
                (mem.concepts[n].description or "")[:60]
                for n in mem.categories[k][:5] if n in mem.concepts
            )
            kind_scores.append((self._overlap(q_toks, kind_text), k))
        kind_scores.sort(reverse=True)
        picked_kinds = [k for _, k in kind_scores[: self.per_layer_top_k]] or kinds
        layer_trace.append({"layer": 1, "kept": picked_kinds, "total": len(kinds)})

        # Layer 2: Trace groups within each picked kind — cluster by shared used_in.
        trace_groups: list[tuple[frozenset[str], list[str]]] = []
        for kind in picked_kinds:
            groups_by_sig: dict[frozenset[str], list[str]] = defaultdict(list)
            for name in mem.categories.get(kind, []):
                c = mem.concepts.get(name)
                if c is None:
                    continue
                sig = frozenset(c.used_in or ())
                groups_by_sig[sig].append(name)
            for sig, members in groups_by_sig.items():
                if len(members) >= 1:
                    trace_groups.append((sig, members))

        trace_scores = []
        for sig, members in trace_groups:
            group_text = " ".join(
                (mem.concepts[n].description or "")[:60] for n in members[:5]
            )
            trace_scores.append((
                self._overlap(q_toks, group_text),
                sum(len(mem.concepts[n].used_in or []) for n in members),
                members,
            ))
        trace_scores.sort(reverse=True)
        picked_groups = trace_scores[: self.per_layer_top_k * 3]
        layer_trace.append({
            "layer": 2, "kept_groups": len(picked_groups),
            "total_groups": len(trace_groups),
        })

        # Layer 3: Episodes (individual concepts within picked groups).
        candidate_concepts: list[str] = []
        for _, _, members in picked_groups:
            candidate_concepts.extend(members)
        # Dedupe while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for n in candidate_concepts:
            if n not in seen:
                seen.add(n)
                deduped.append(n)

        concept_scores = []
        for name in deduped:
            c = mem.concepts[name]
            c_text = " ".join([name, c.description or "", " ".join(c.cues or [])])
            concept_scores.append((self._overlap(q_toks, c_text), len(c.used_in or []), name))
        concept_scores.sort(reverse=True)
        top = [name for _, _, name in concept_scores[: self.top_k]]
        layer_trace.append({"layer": 3, "kept_concepts": len(top), "pool_size": len(deduped)})

        hint = mem.to_string(
            concept_names=top,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            usage_threshold=self.usage_threshold,
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in top],
            metadata={
                "retriever": self.name,
                "layer_trace": layer_trace,
                "num_selected": len(top),
                "top_k": self.top_k,
                "per_layer_top_k": self.per_layer_top_k,
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

    def _retrieve_with_prebuilt_hierarchy(
        self,
        hdata: dict,
        mem: ConceptMemory,
        q_toks: set[str],
        query_text: str,
        problem: ProblemSpec,
        source: str = "prebuilt_v1",
        adapted_records: dict[str, dict] | None = None,
        adapted_memory_source: str = "flat",
    ) -> RetrievalBundle:
        """3-level walk: Category → Sub-category → concept."""
        adapted_records = adapted_records or {}
        layer_trace: list[dict] = []
        cats = hdata.get("categories", []) or []

        # Layer 1: pick top categories by token overlap (name+description).
        cat_scores: list[tuple[int, dict]] = []
        for cat in cats:
            cat_text = (cat.get("name", "") + " " + cat.get("description", "")).strip()
            cat_scores.append((self._overlap(q_toks, cat_text), cat))
        cat_scores.sort(reverse=True, key=lambda kv: kv[0])
        picked_cats = [c for s, c in cat_scores[: self.per_layer_top_k]] or cats[: self.per_layer_top_k]
        layer_trace.append({
            "layer": 1, "kept": [c.get("name") for c in picked_cats],
            "total": len(cats),
        })

        # Layer 2: pick top sub-categories within each picked category.
        picked_subs: list[dict] = []
        for cat in picked_cats:
            subs = cat.get("subcategories", []) or []
            sub_scores: list[tuple[int, dict]] = []
            for sub in subs:
                sub_text = (sub.get("name", "") + " " + sub.get("description", "")).strip()
                sub_scores.append((self._overlap(q_toks, sub_text), sub))
            sub_scores.sort(reverse=True, key=lambda kv: kv[0])
            picked_subs.extend([s for _, s in sub_scores[: self.per_layer_top_k]])
        layer_trace.append({
            "layer": 2, "kept_subs": [s.get("name") for s in picked_subs],
            "total_subs_in_picked_cats": sum(len(c.get("subcategories", []) or []) for c in picked_cats),
        })

        # Layer 3: rank concepts within picked sub-categories by query overlap.
        candidates: list[str] = []
        for sub in picked_subs:
            for cn in sub.get("concepts", []) or []:
                if cn in mem.concepts and cn not in candidates:
                    candidates.append(cn)
        # If no candidates (e.g. hierarchy doesn't cover this memory's concepts),
        # broaden to ALL concepts as a safety net.
        if not candidates:
            candidates = sorted(mem.concepts.keys())
        concept_scores: list[tuple[int, int, str]] = []
        for name in candidates:
            c = mem.concepts[name]
            c_text = " ".join([
                name, c.description or "",
                " ".join(str(x) for x in (c.cues or [])),
            ])
            concept_scores.append((
                self._overlap(q_toks, c_text),
                len(c.used_in or []),
                name,
            ))
        concept_scores.sort(reverse=True)
        top = [name for _, _, name in concept_scores[: self.top_k]]
        layer_trace.append({
            "layer": 3, "kept_concepts": len(top), "pool_size": len(candidates),
        })

        hint = self._render_adapted_hint(top, adapted_records) if adapted_records else ""
        if not hint:
            hint = mem.to_string(
                concept_names=top,
                include_description=self.include_description,
                skip_cues=self.skip_cues,
                skip_implementation=self.skip_implementation,
                usage_threshold=self.usage_threshold,
            )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in top],
            metadata={
                "retriever": self.name,
                "scoring_mode": f"hmem_{source}_3level",
                "hierarchy_source": source,
                "layer_trace": layer_trace,
                "num_selected": len(top),
                "top_k": self.top_k,
                "per_layer_top_k": self.per_layer_top_k,
                "adapted_memory_source": adapted_memory_source,
                "adapted_records_loaded": len(adapted_records),
            },
        )

    # ----------------------------------------------------------------- #
    def _query_text(
        self, problem: ProblemSpec, previous_attempts: list[AttemptRecord],
    ) -> str:
        parts: list[str] = [str(getattr(problem, "uid", ""))]
        meta = getattr(problem, "metadata", {}) or {}
        for key in ("description", "instructions", "prompt", "query"):
            if meta.get(key):
                parts.append(str(meta[key]))
        return " ".join(parts)

    def _overlap(self, q_toks: set[str], text: str) -> float:
        d_toks = _toks(text)
        if not q_toks or not d_toks:
            return 0.0
        return len(q_toks & d_toks)

    @staticmethod
    def _resolve_path(path: str | Path | None, default: Path) -> Path:
        if path is None:
            return default
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _load_adapted_hierarchy(
        self,
        mem: ConceptMemory,
    ) -> tuple[dict | None, dict[str, dict], str]:
        path = self.adapted_memory_path
        if not path.exists():
            return None, {}, "flat"
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - corrupted local artifact should not be silent
            raise RuntimeError(f"invalid H-MEM adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != self.name:
            raise RuntimeError(f"invalid H-MEM adapted memory schema: {path}")
        records: dict[str, dict] = {}
        category_order: list[str] = []
        sub_order: dict[str, list[str]] = {}
        grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        descriptions: dict[tuple[str, str | None], str] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            category = str(raw.get("category") or "").strip()
            subcategory = str(raw.get("subcategory") or "").strip()
            if not category or not subcategory:
                raise RuntimeError(f"adapted memory missing route for {concept_id}")
            records[concept_id] = raw
            if category not in category_order:
                category_order.append(category)
            if subcategory not in sub_order.setdefault(category, []):
                sub_order[category].append(subcategory)
            grouped[category][subcategory].append(concept_id)
            trace = raw.get("memory_trace") if isinstance(raw.get("memory_trace"), dict) else {}
            episode = raw.get("episode") if isinstance(raw.get("episode"), dict) else {}
            descriptions[(category, None)] = str(raw.get("retrieval_notes") or category)
            descriptions[(category, subcategory)] = str(
                trace.get("trace_summary") or episode.get("when_to_route_here") or subcategory
            )
        if not records:
            return None, {}, "flat"
        categories = []
        for category in category_order:
            subs = []
            for subcategory in sub_order.get(category, []):
                subs.append({
                    "name": subcategory,
                    "description": descriptions.get((category, subcategory), subcategory),
                    "concepts": grouped[category][subcategory],
                })
            categories.append({
                "name": category,
                "description": descriptions.get((category, None), category),
                "subcategories": subs,
            })
        return {"categories": categories}, records, "hmem_memory_v1"

    @staticmethod
    def _render_adapted_hint(
        top: list[str],
        adapted_records: dict[str, dict],
    ) -> str:
        blocks: list[str] = []
        for name in top:
            record = adapted_records.get(name)
            if not record:
                continue
            trace = record.get("memory_trace") if isinstance(record.get("memory_trace"), dict) else {}
            episode = record.get("episode") if isinstance(record.get("episode"), dict) else {}
            lines = [f"- concept: {name}"]
            lines.append(
                "  hmem_route: "
                f"{record.get('domain', 'ARC-AGI')} / {record.get('category')} / {record.get('subcategory')}"
            )
            trace_summary = str(trace.get("trace_summary") or "").strip()
            if trace_summary:
                lines.append(f"  memory_trace: {trace_summary}")
            episode_summary = str(episode.get("summary") or "").strip()
            if episode_summary:
                lines.append(f"  episode: {episode_summary}")
            route_here = str(episode.get("when_to_route_here") or "").strip()
            if route_here:
                lines.append(f"  route_when: {route_here}")
            keywords = [str(k).strip() for k in record.get("routing_keywords") or [] if str(k).strip()]
            if keywords:
                lines.append("  routing_keywords: " + ", ".join(keywords[:8]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
