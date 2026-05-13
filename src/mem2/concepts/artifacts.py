"""Load optional ARC concept-memory prep artifacts.

The core memory payload stays portable. These helpers let axis-specific
retrievers opt into richer prep artifacts when those files exist in the local
ARC concept-memory directory, while keeping the old template behavior as the
fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
CONCEPT_MEMORY_DIR = REPO_ROOT / "data" / "arc_agi" / "concept_memory"
COMMUNITY_SUMMARIES_PATH = CONCEPT_MEMORY_DIR / "community_summaries_v1.json"
OPENIE_FACTS_PATH = CONCEPT_MEMORY_DIR / "concept_facts_openie_v1.json"
ENTITY_GRAPH_PATH = CONCEPT_MEMORY_DIR / "concept_entity_graph_v1.json"
HIERARCHICAL_REPORTS_PATH = CONCEPT_MEMORY_DIR / "entity_hierarchical_reports_v1.json"
RAPTOR_TREE_PATH = CONCEPT_MEMORY_DIR / "raptor_tree_v1.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_community_summaries(
    path: str | Path | None = None,
    *,
    valid_concepts: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    artifact_path = _resolve(path, COMMUNITY_SUMMARIES_PATH)
    if not artifact_path.exists():
        return []
    data = _read_json(artifact_path)
    if not data or data.get("schema_version") != "1":
        return []

    valid = set(valid_concepts) if valid_concepts is not None else None
    out: list[dict[str, Any]] = []
    for raw in data.get("communities", []) or []:
        if not isinstance(raw, dict):
            continue
        members = raw.get("member_concepts") or []
        if not isinstance(members, list):
            continue
        cleaned_members = [m for m in members if isinstance(m, str) and (valid is None or m in valid)]
        if not cleaned_members:
            continue
        summary = raw.get("llm_summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        seed = raw.get("seed_concept")
        if not isinstance(seed, str) or seed not in cleaned_members:
            seed = cleaned_members[0]
        out.append({
            "community_id": str(raw.get("community_id") or f"community_{len(out)}"),
            "seed_concept": seed,
            "member_concepts": cleaned_members,
            "member_digest": str(raw.get("member_digest") or ""),
            "llm_summary": summary.strip(),
            "summary_tokens": raw.get("summary_tokens"),
        })
    return out


def load_openie_facts(
    path: str | Path | None = None,
    *,
    valid_concepts: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    artifact_path = _resolve(path, OPENIE_FACTS_PATH)
    if not artifact_path.exists():
        return []
    data = _read_json(artifact_path)
    if not data or data.get("schema_version") != "1":
        return []

    valid = set(valid_concepts) if valid_concepts is not None else None
    out: list[dict[str, Any]] = []
    for raw in data.get("facts", []) or []:
        if not isinstance(raw, dict):
            continue
        source = raw.get("source_concept")
        if not isinstance(source, str) or (valid is not None and source not in valid):
            continue
        linked_raw = raw.get("linked_concepts") or []
        linked = [
            c for c in linked_raw
            if isinstance(c, str) and (valid is None or c in valid)
        ]
        for field in ("subject", "object"):
            value = raw.get(field)
            if isinstance(value, str) and valid is not None and value in valid and value not in linked:
                linked.append(value)
        if source not in linked:
            linked.insert(0, source)
        if len(set(linked)) < 2:
            continue
        out.append({
            "fact_id": str(raw.get("fact_id") or f"openie_{len(out):05d}"),
            "source_concept": source,
            "subject": str(raw.get("subject") or ""),
            "predicate": str(raw.get("predicate") or ""),
            "object": str(raw.get("object") or ""),
            "confidence": raw.get("confidence", 1.0),
            "supporting_text": str(raw.get("supporting_text") or ""),
            "linked_concepts": list(dict.fromkeys(linked)),
            "relation_kind": str(raw.get("relation_kind") or "other"),
        })
    return out


def load_entity_graph(
    path: str | Path | None = None,
    *,
    valid_concepts: Iterable[str] | None = None,
) -> dict[str, Any]:
    artifact_path = _resolve(path, ENTITY_GRAPH_PATH)
    if not artifact_path.exists():
        return {"entities": [], "edges": [], "stats": {}}
    data = _read_json(artifact_path)
    if not data or data.get("schema_version") != "1":
        return {"entities": [], "edges": [], "stats": {}}

    valid = set(valid_concepts) if valid_concepts is not None else None
    entities: list[dict[str, Any]] = []
    entity_ids: set[str] = set()
    for raw in data.get("entities", []) or []:
        if not isinstance(raw, dict):
            continue
        entity_id = raw.get("entity_id")
        source = raw.get("source_concept")
        mention = raw.get("mention_text")
        if not isinstance(entity_id, str) or not isinstance(source, str):
            continue
        if valid is not None and source not in valid:
            continue
        if not isinstance(mention, str) or not mention.strip():
            continue
        cleaned = {
            "entity_id": entity_id,
            "mention_text": mention.strip(),
            "source_concept": source,
            "entity_type": str(raw.get("entity_type") or "other"),
            "attributes": raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {},
            "supporting_text": str(raw.get("supporting_text") or ""),
        }
        entities.append(cleaned)
        entity_ids.add(entity_id)

    edges: list[dict[str, Any]] = []
    for raw in data.get("edges", []) or []:
        if not isinstance(raw, dict):
            continue
        src = raw.get("src_entity")
        dst = raw.get("dst_entity")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if src not in entity_ids or dst not in entity_ids or src == dst:
            continue
        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        edges.append({
            "src_entity": src,
            "dst_entity": dst,
            "edge_type": str(raw.get("edge_type") or "related_to"),
            "weight": weight,
            "supporting_text": str(raw.get("supporting_text") or ""),
        })

    return {
        "schema_version": "1",
        "source_seed": data.get("source_seed"),
        "model": data.get("model"),
        "entities": entities,
        "edges": edges,
        "stats": data.get("stats") or {},
    }


def load_hierarchical_reports(
    path: str | Path | None = None,
    *,
    valid_concepts: Iterable[str] | None = None,
) -> dict[str, Any]:
    artifact_path = _resolve(path, HIERARCHICAL_REPORTS_PATH)
    if not artifact_path.exists():
        return {"hierarchy": {}}
    data = _read_json(artifact_path)
    if not data or data.get("schema_version") != "1":
        return {"hierarchy": {}}

    valid = set(valid_concepts) if valid_concepts is not None else None
    hierarchy: dict[str, list[dict[str, Any]]] = {}
    for level, reports in (data.get("hierarchy") or {}).items():
        if not isinstance(level, str) or not isinstance(reports, list):
            continue
        cleaned_reports: list[dict[str, Any]] = []
        for raw in reports:
            if not isinstance(raw, dict):
                continue
            summary = raw.get("llm_summary")
            if not isinstance(summary, str) or not summary.strip():
                continue
            source_concepts = [
                c for c in (raw.get("source_concepts") or [])
                if isinstance(c, str) and (valid is None or c in valid)
            ]
            if valid is not None and not source_concepts:
                continue
            cleaned_reports.append({
                "community_id": str(raw.get("community_id") or f"{level}_{len(cleaned_reports)}"),
                "level": int(raw.get("level", 0) or 0),
                "entities": [str(e) for e in (raw.get("entities") or []) if isinstance(e, str)],
                "source_concepts": list(dict.fromkeys(source_concepts)),
                "child_communities": [
                    str(c) for c in (raw.get("child_communities") or [])
                    if isinstance(c, str)
                ],
                "member_digest": str(raw.get("member_digest") or ""),
                "llm_summary": summary.strip(),
                "summary_tokens": raw.get("summary_tokens"),
            })
        if cleaned_reports:
            hierarchy[level] = cleaned_reports

    return {
        "schema_version": "1",
        "source_graph": data.get("source_graph"),
        "model": data.get("model"),
        "hierarchy": hierarchy,
        "stats": data.get("stats") or {},
    }


def load_raptor_tree(
    path: str | Path | None = None,
    *,
    valid_concepts: Iterable[str] | None = None,
) -> dict[str, Any]:
    artifact_path = _resolve(path, RAPTOR_TREE_PATH)
    if not artifact_path.exists():
        return {"levels": []}
    data = _read_json(artifact_path)
    if not data or data.get("schema_version") != "1":
        return {"levels": []}

    valid = set(valid_concepts) if valid_concepts is not None else None
    levels: list[dict[str, Any]] = []
    known_node_ids: set[str] = set()
    for raw_level in data.get("levels", []) or []:
        if not isinstance(raw_level, dict):
            continue
        try:
            level_idx = int(raw_level.get("level", len(levels)))
        except (TypeError, ValueError):
            level_idx = len(levels)
        nodes: list[dict[str, Any]] = []
        for raw in raw_level.get("nodes", []) or []:
            if not isinstance(raw, dict):
                continue
            node_id = raw.get("node_id")
            summary = raw.get("summary")
            if not isinstance(node_id, str) or not isinstance(summary, str) or not summary.strip():
                continue
            member_concepts = [
                c for c in (raw.get("member_concepts") or [])
                if isinstance(c, str) and (valid is None or c in valid)
            ]
            if valid is not None and not member_concepts:
                continue
            child_ids = [
                c for c in (raw.get("child_node_ids") or raw.get("member_node_ids") or [])
                if isinstance(c, str)
            ]
            nodes.append({
                "node_id": node_id,
                "summary": summary.strip(),
                "member_communities": [
                    c for c in (raw.get("member_communities") or []) if isinstance(c, str)
                ],
                "member_concepts": list(dict.fromkeys(member_concepts)),
                "member_node_ids": child_ids,
                "child_node_ids": child_ids,
                "summary_tokens": raw.get("summary_tokens"),
            })
            known_node_ids.add(node_id)
        if nodes:
            levels.append({"level": level_idx, "nodes": nodes})

    if len(levels) < 2:
        return {"levels": []}
    valid_ids = {n["node_id"] for level in levels for n in level["nodes"]}
    for level in levels:
        for node in level["nodes"]:
            node["member_node_ids"] = [n for n in node["member_node_ids"] if n in valid_ids]
            node["child_node_ids"] = [n for n in node["child_node_ids"] if n in valid_ids]

    return {
        "schema_version": "1",
        "source_seed": data.get("source_seed"),
        "model": data.get("model"),
        "levels": sorted(levels, key=lambda level: level["level"]),
        "stats": data.get("stats") or {},
    }
