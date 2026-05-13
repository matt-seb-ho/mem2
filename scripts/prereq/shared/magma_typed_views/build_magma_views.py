"""Build typed MAGMA view graphs from ARC concept and entity substrates.

Output:
  data/arc_agi/concept_memory/magma_typed_views_v1.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.concepts.memory import ConceptMemory


SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
ENTITY_GRAPH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "entity_graph_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "magma_typed_views_v1.json"
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
VERBS = {
    "align",
    "apply",
    "combine",
    "compare",
    "copy",
    "count",
    "crop",
    "detect",
    "draw",
    "extract",
    "filter",
    "find",
    "fill",
    "group",
    "identify",
    "match",
    "mirror",
    "overlay",
    "place",
    "recolor",
    "reflect",
    "remove",
    "repeat",
    "rotate",
    "scale",
    "select",
    "sort",
    "split",
    "transform",
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _concept_text(raw: dict[str, Any]) -> str:
    parts = [
        str(raw.get("description") or ""),
        str(raw.get("routine_subtype") or ""),
        str(raw.get("output_typing") or ""),
        " ".join(str(c) for c in (raw.get("cues") or [])[:4]),
        " ".join(str(c) for c in (raw.get("implementation") or [])[:4]),
    ]
    return " ".join(parts)


def _verb_signature(name: str, raw: dict[str, Any]) -> set[str]:
    toks = _tokens(name) | _tokens(_concept_text(raw))
    stems: set[str] = set()
    for tok in toks:
        base = tok
        for suffix in ("ing", "ed", "s"):
            if base.endswith(suffix) and len(base) > len(suffix) + 3:
                base = base[:-len(suffix)]
                break
        if base in VERBS:
            stems.add(base)
    return stems


def _node(node_id: str, label: str, node_type: str, **extra) -> dict[str, Any]:
    out = {"node_id": node_id, "label": label, "node_type": node_type}
    out.update(extra)
    return out


def _edge(src: str, dst: str, edge_type: str, weight: float, **extra) -> dict[str, Any]:
    out = {"src": src, "dst": dst, "edge_type": edge_type, "weight": round(float(weight), 4)}
    out.update(extra)
    return out


def build_views() -> dict[str, Any]:
    seed = json.loads(SEED_MEM.read_text())
    mem = ConceptMemory.from_payload(seed)
    concepts = seed.get("concepts", {})
    entity_graph = json.loads(ENTITY_GRAPH.read_text())
    entities = [e for e in entity_graph.get("entities", []) or [] if isinstance(e, dict)]
    raw_edges = [e for e in entity_graph.get("edges", []) or [] if isinstance(e, dict)]
    by_entity = {e.get("entity_id"): e for e in entities if e.get("entity_id")}

    concept_nodes = {
        name: _node(f"concept::{name}", name, "concept", kind=concept.kind)
        for name, concept in mem.concepts.items()
    }
    semantic_nodes = dict(concept_nodes)
    semantic_edges: list[dict[str, Any]] = []
    for entity in entities:
        entity_id = str(entity.get("entity_id"))
        source = str(entity.get("source_concept") or "")
        if source not in mem.concepts:
            continue
        entity_node_id = f"entity::{entity_id}"
        semantic_nodes[entity_node_id] = _node(
            entity_node_id,
            str(entity.get("mention_text") or entity_id),
            str(entity.get("entity_type") or "entity"),
            source_concept=source,
        )
        semantic_edges.append(_edge(
            f"concept::{source}",
            entity_node_id,
            str(entity.get("entity_type") or "mentions_entity"),
            1.0,
            supporting_text=str(entity.get("supporting_text") or ""),
        ))

    causal_edges: list[dict[str, Any]] = []
    signatures = {name: _verb_signature(name, concepts[name]) for name in mem.concepts}
    names = sorted(mem.concepts.keys())
    for i, src in enumerate(names):
        for dst in names[i + 1:]:
            shared = signatures[src] & signatures[dst]
            if not shared:
                continue
            src_toks = _tokens(_concept_text(concepts[src]))
            dst_toks = _tokens(_concept_text(concepts[dst]))
            obj_overlap = len(src_toks & dst_toks)
            if obj_overlap < 2:
                continue
            causal_edges.append(_edge(
                f"concept::{src}",
                f"concept::{dst}",
                "shared_operation_predicate",
                len(shared) + 0.01 * obj_overlap,
                predicates=sorted(shared),
            ))

    structural_counts: dict[tuple[str, str], float] = defaultdict(float)
    structural_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for raw in raw_edges:
        src_entity = by_entity.get(raw.get("src_entity"))
        dst_entity = by_entity.get(raw.get("dst_entity"))
        if not src_entity or not dst_entity:
            continue
        src = str(src_entity.get("source_concept") or "")
        dst = str(dst_entity.get("source_concept") or "")
        if src == dst or src not in mem.concepts or dst not in mem.concepts:
            continue
        key = tuple(sorted((src, dst)))
        structural_counts[key] += float(raw.get("weight") or 1.0)
        structural_labels[key].add(str(raw.get("edge_type") or "related_to"))
    structural_edges = [
        _edge(
            f"concept::{src}",
            f"concept::{dst}",
            "entity_co_mention_strength",
            weight,
            relation_types=sorted(structural_labels[(src, dst)]),
        )
        for (src, dst), weight in sorted(structural_counts.items())
    ]

    return {
        "schema_version": "1",
        "model": "deterministic_entity_graph_projection",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "views": {
            "semantic": {
                "nodes": list(semantic_nodes.values()),
                "edges": semantic_edges,
            },
            "causal": {
                "nodes": list(concept_nodes.values()),
                "edges": causal_edges,
            },
            "structural": {
                "nodes": list(concept_nodes.values()),
                "edges": structural_edges,
            },
        },
        "stats": {
            "num_views": 3,
            "edges_per_view": {
                "semantic": len(semantic_edges),
                "causal": len(causal_edges),
                "structural": len(structural_edges),
            },
            "llm_calls": 0,
            "estimated_cost_usd": 0.0,
            "source_entity_graph_stats": entity_graph.get("stats") or {},
        },
    }


def main() -> int:
    args = parse_args()
    if OUT_FILE.exists() and not args.force:
        print(f"ERROR: output exists: {OUT_FILE}", file=sys.stderr)
        return 2
    out = build_views()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[magma_views] wrote {OUT_FILE}")
    print(f"[magma_views] edges_per_view={out['stats']['edges_per_view']} cost=$0")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
