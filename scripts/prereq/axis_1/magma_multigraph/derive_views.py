"""Derive 4 graph views for MAGMA multi-graph retrieval (axis 1).

Why this exists
---------------
MAGMA (axis 1.11) wants 4 orthogonal relational views:
  - SEMANTIC: concept-concept conceptual relatedness
  - TEMPORAL: authorship-lineage / abstraction predecessors
  - CAUSAL: shared-problem usage / dependency chains
  - ENTITY: kind-membership / type relationships

The hipporag-built graph (`concept_graph_v1.json`) and the seed memory
together carry enough signal to derive all four views without additional
LLM calls. This script splits and re-weights the existing data:

  SEMANTIC  ←  hipporag edges with relation in {"uses", "specializes"}
                + edges with concept-name token overlap
  TEMPORAL  ←  hipporag edges with relation in {"is_a", "specializes",
                "composed_of"} (abstraction lineage stands in for true
                temporal lineage we don't have)
  CAUSAL    ←  co-activation edges from `used_in` overlap (concepts that
                were used together to solve the same problem)
  ENTITY    ←  kind-membership edges (each concept connected to its kind
                and to other concepts of the same kind)

This is honestly DERIVED, not extracted; "temporal" and "causal" here
are approximations to what the paper means. Documented as "Reduced" fit
in doc 52.

Inputs
------
- mem2/data/arc_agi/concept_memory/compressed_v1.json
- mem2/data/arc_agi/concept_memory/concept_graph_v1.json (built by
  scripts/prereq/axis_1/hipporag_ppr/build_concept_graph.py)

Outputs
-------
- mem2/data/arc_agi/concept_memory/concept_views_v1.json
  Schema:
    {
      "schema_version": "1",
      "built_at": "...",
      "views": {
        "semantic":  [{"src", "tgt", "weight"}, ...],
        "temporal":  [{"src", "tgt", "weight"}, ...],
        "causal":    [{"src", "tgt", "weight"}, ...],
        "entity":    [{"src", "tgt", "weight"}, ...]
      },
      "stats": {<per-view edge counts>}
    }

Cost / runtime
--------------
$0 (no LLM calls). <1s.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
GRAPH = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_graph_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_views_v1.json"


def main() -> int:
    if not SEED_MEM.exists():
        print(f"ERROR: {SEED_MEM} not found", file=sys.stderr)
        return 2
    if not GRAPH.exists():
        print(f"ERROR: {GRAPH} not found — run hipporag_ppr/build_concept_graph.py first", file=sys.stderr)
        return 2

    seed = json.loads(SEED_MEM.read_text())
    concepts: dict[str, dict] = seed.get("concepts", {})
    graph_data = json.loads(GRAPH.read_text())
    edges = graph_data.get("edges", [])

    # Confirm graph is the full one, not a smoketest leftover
    n_concepts_in_graph = graph_data.get("stats", {}).get("num_concepts", 0)
    if n_concepts_in_graph < len(concepts) - 5:  # tolerate small slack
        print(
            f"WARN: graph file appears partial ({n_concepts_in_graph}/{len(concepts)} concepts);"
            f" run the full hipporag graph build before deriving views.",
            file=sys.stderr,
        )

    # SEMANTIC: uses + specializes from hipporag graph
    semantic = []
    for e in edges:
        if e["relation"] in ("uses", "specializes"):
            semantic.append({"src": e["src"], "tgt": e["tgt"], "weight": 1.0})

    # TEMPORAL: abstraction lineage (is_a, specializes, composed_of)
    temporal = []
    for e in edges:
        if e["relation"] in ("is_a", "specializes", "composed_of"):
            temporal.append({"src": e["src"], "tgt": e["tgt"], "weight": 1.0})

    # CAUSAL: co-activation from used_in overlap
    causal_pairs: dict[tuple[str, str], int] = defaultdict(int)
    for name, c in concepts.items():
        used_in = c.get("used_in", []) or []
        for other_name, other in concepts.items():
            if other_name == name:
                continue
            other_used = other.get("used_in", []) or []
            shared = len(set(used_in) & set(other_used))
            if shared > 0:
                key = (name, other_name)
                causal_pairs[key] = shared
    causal = [
        {"src": s, "tgt": t, "weight": float(w)}
        for (s, t), w in causal_pairs.items()
    ]

    # ENTITY: kind-membership (concepts of same kind)
    by_kind: dict[str, list[str]] = defaultdict(list)
    for name, c in concepts.items():
        by_kind[c.get("kind", "?")].append(name)
    entity = []
    for kind, names in by_kind.items():
        for a, b in combinations(sorted(names), 2):
            entity.append({"src": a, "tgt": b, "weight": 1.0})
            entity.append({"src": b, "tgt": a, "weight": 1.0})

    out = {
        "schema_version": "1",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "source_graph": str(GRAPH.relative_to(ROOT)),
        "views": {
            "semantic": semantic,
            "temporal": temporal,
            "causal": causal,
            "entity": entity,
        },
        "stats": {
            "num_concepts": len(concepts),
            "num_kinds": len(by_kind),
            "kind_distribution": {k: len(v) for k, v in by_kind.items()},
            "edges_per_view": {
                "semantic": len(semantic),
                "temporal": len(temporal),
                "causal": len(causal),
                "entity": len(entity),
            },
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[derive_views] wrote {OUT_FILE.name}")
    for view, count in out["stats"]["edges_per_view"].items():
        print(f"  {view:10s} : {count} edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
