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

