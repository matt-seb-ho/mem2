"""Build an LLM-extracted document-entity graph from ARC concept memory.

Inputs:
  data/arc_agi/concept_memory/compressed_v1.json

Output:
  data/arc_agi/concept_memory/concept_entity_graph_v1.json

The artifact converts each concept-memory entry into a small document with
typed entity mentions and relation edges. Graph retrievers can then consume a
document-entity substrate instead of treating concept names as the only nodes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.concepts.memory import ConceptMemory
from mem2.providers.llmplus_client import LLMPlusProviderClient


SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_entity_graph_v1.json"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28

ENTITY_TYPES = {
    "concept",
    "operation",
    "attribute",
    "target_object",
    "color",
    "shape",
    "transformation",
    "spatial_relation",
    "pattern",
    "condition",
    "parameter",
    "routine",
    "output",
    "other",
}
EDGE_TYPES = {
    "co_mentions",
    "uses",
    "transforms",
    "filters",
    "parameterizes",
    "targets",
    "has_attribute",
    "same_as",
    "related_to",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _concept_digest(name: str, raw: dict[str, Any]) -> str:
    parts = [f"- name: {name}", f"  kind: {raw.get('kind', '?')}"]
    desc = str(raw.get("description") or "").strip()
    if desc:
        parts.append(f"  description: {desc}")
    params = raw.get("parameters") or []
    if params:
        names = [str(p.get("name")) for p in params if isinstance(p, dict) and p.get("name")]
        if names:
            parts.append("  parameters: " + ", ".join(names[:10]))
    cues = [str(c).strip() for c in (raw.get("cues") or []) if str(c).strip()]
    if cues:
        parts.append("  cues: " + " | ".join(cues[:6]))
    impl = [str(c).strip() for c in (raw.get("implementation") or []) if str(c).strip()]
    if impl:
        parts.append("  implementation: " + " | ".join(impl[:8]))
    return "\n".join(parts)


def _build_prompt(concepts: dict[str, dict[str, Any]], name: str, *, max_entities: int, max_edges: int) -> str:
    return f"""You convert one ARC-AGI concept-memory document into a typed entity graph.

Return lines only. No prose, no markdown fences, no JSON wrapper.

Line formats:
ENTITY || local_id || mention_text || entity_type || attributes_json || supporting_text
EDGE || src_local_id || dst_local_id || edge_type || weight || supporting_text

Rules:
- Extract 5 to {max_entities} ENTITY lines when possible.
- local_id must be short and unique within this concept, e.g. e1, e2.
- entity_type must be one of: {", ".join(sorted(ENTITY_TYPES))}
- attributes_json must be a compact JSON object, or {{}} if none.
- EDGE endpoints must refer to extracted local_ids.
- Edge type must be one of: {", ".join(sorted(EDGE_TYPES))}
- Use {max_edges} or fewer EDGE lines.
- Favor operational ARC solver entities: operations, objects, colors, shapes, transformations, spatial relations, parameters, outputs.
- Do not invent facts outside the concept fields.

# CONCEPT DOCUMENT

{_concept_digest(name, concepts[name])}

# EXAMPLE
ENTITY || e1 || object extraction || operation || {{"role":"routine"}} || extract objects from grid
ENTITY || e2 || connected components || target_object || {{}} || connected components
EDGE || e1 || e2 || targets || 0.9 || extraction targets connected components"""


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def _entity_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _norm_mention(text: str) -> str:
    toks = sorted(_entity_tokens(text))
    return " ".join(toks)


def _parse_attributes(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:160]}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _parse_response(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    local_ids: set[str] = set()
    for line in _strip_code_fence(raw).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.lstrip("-*0123456789. ").strip()
        fields = [f.strip() for f in line.split("||")]
        if not fields:
            continue
        tag = fields[0].upper()
        if tag == "ENTITY" and len(fields) == 6:
            local_id, mention, entity_type, attrs, supporting = fields[1:]
            if not local_id or not mention:
                continue
            if entity_type not in ENTITY_TYPES:
                entity_type = "other"
            if local_id in local_ids:
                continue
            local_ids.add(local_id)
            entities.append({
                "local_id": local_id,
                "mention_text": mention[:160],
                "entity_type": entity_type,
                "attributes": _parse_attributes(attrs),
                "supporting_text": supporting[:300],
            })
        elif tag == "EDGE" and len(fields) == 6:
            src, dst, edge_type, weight, supporting = fields[1:]
            if not src or not dst or src == dst:
                continue
            if edge_type not in EDGE_TYPES:
                edge_type = "related_to"
            try:
                weight_value = max(0.0, min(1.0, float(weight)))
            except (TypeError, ValueError):
                weight_value = 1.0
            edges.append({
                "src_local": src,
                "dst_local": dst,
                "edge_type": edge_type,
                "weight": weight_value,
                "supporting_text": supporting[:300],
            })
    return entities, edges


def _fallback_entities(name: str, raw: dict[str, Any], *, max_entities: int) -> list[dict[str, Any]]:
    pieces = [name, str(raw.get("kind") or "concept")]
    for key in ("description", "routine_subtype", "output_typing"):
        value = raw.get(key)
        if value:
            pieces.append(str(value))
    for cue in raw.get("cues") or []:
        pieces.append(str(cue))
    tokens = [t for t in dict.fromkeys(WORD_RE.findall(" ".join(pieces).lower())) if len(t) > 2]
    entities = [{
        "local_id": "e0",
        "mention_text": name,
        "entity_type": "concept",
        "attributes": {"kind": raw.get("kind", "unknown")},
        "supporting_text": str(raw.get("description") or name)[:300],
    }]
    for idx, tok in enumerate(tokens[: max(0, max_entities - 1)], start=1):
        entities.append({
            "local_id": f"e{idx}",
            "mention_text": tok,
            "entity_type": "other",
            "attributes": {},
            "supporting_text": str(raw.get("description") or tok)[:300],
        })
    return entities


def _materialize_entities(
    raw_entities: list[dict[str, Any]],
    *,
    source_concept: str,
    start_idx: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    out: list[dict[str, Any]] = []
    local_to_global: dict[str, str] = {}
    seen_mentions: set[tuple[str, str]] = set()
    for raw in raw_entities:
        mention = str(raw.get("mention_text") or "").strip()
        if not mention:
            continue
        entity_type = str(raw.get("entity_type") or "other")
        key = (entity_type, _norm_mention(mention))
        if key in seen_mentions:
            continue
        seen_mentions.add(key)
        entity_id = f"ent_{start_idx + len(out):05d}"
        local_id = str(raw.get("local_id") or f"e{len(out)}")
        local_to_global[local_id] = entity_id
        out.append({
            "entity_id": entity_id,
            "mention_text": mention,
            "source_concept": source_concept,
            "entity_type": entity_type if entity_type in ENTITY_TYPES else "other",
            "attributes": raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {},
            "supporting_text": str(raw.get("supporting_text") or "")[:300],
        })
    return out, local_to_global


def _materialize_local_edges(
    raw_edges: list[dict[str, Any]],
    local_to_global: dict[str, str],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_edges:
        src = local_to_global.get(str(raw.get("src_local") or ""))
        dst = local_to_global.get(str(raw.get("dst_local") or ""))
        if not src or not dst or src == dst:
            continue
        edge_type = str(raw.get("edge_type") or "related_to")
        key = tuple(sorted((src, dst))) + (edge_type,)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "src_entity": src,
            "dst_entity": dst,
            "edge_type": edge_type if edge_type in EDGE_TYPES else "related_to",
            "weight": float(raw.get("weight") or 1.0),
            "supporting_text": str(raw.get("supporting_text") or "")[:300],
        })
    return edges


def _add_cross_concept_edges(
    entities: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    max_alias_edges_per_entity: int,
) -> None:
    seen = {
        tuple(sorted((e["src_entity"], e["dst_entity"]))) + (str(e.get("edge_type") or ""),)
        for e in edges
    }
    by_norm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ent in entities:
        norm = _norm_mention(str(ent.get("mention_text") or ""))
        if norm:
            by_norm[(str(ent.get("entity_type") or "other"), norm)].append(ent)
            by_type[str(ent.get("entity_type") or "other")].append(ent)

    per_entity: dict[str, int] = defaultdict(int)
    for (_, norm), group in by_norm.items():
        if len(group) < 2:
            continue
        for i, src in enumerate(group):
            for dst in group[i + 1:]:
                if src["source_concept"] == dst["source_concept"]:
                    continue
                if per_entity[src["entity_id"]] >= max_alias_edges_per_entity:
                    continue
                if per_entity[dst["entity_id"]] >= max_alias_edges_per_entity:
                    continue
                key = tuple(sorted((src["entity_id"], dst["entity_id"]))) + ("same_as",)
                if key in seen:
                    continue
                seen.add(key)
                per_entity[src["entity_id"]] += 1
                per_entity[dst["entity_id"]] += 1
                edges.append({
                    "src_entity": src["entity_id"],
                    "dst_entity": dst["entity_id"],
                    "edge_type": "same_as",
                    "weight": 1.0,
                    "supporting_text": f"same normalized mention: {norm}",
                })

    # Add a small number of fuzzy cross-concept edges so semantically related
    # but non-identical mentions can connect the projected concept graph.
    for entity_type, group in by_type.items():
        for i, src in enumerate(group):
            if per_entity[src["entity_id"]] >= max_alias_edges_per_entity:
                continue
            src_toks = _entity_tokens(str(src.get("mention_text") or ""))
            if not src_toks:
                continue
            candidates: list[tuple[float, dict[str, Any]]] = []
            for dst in group[i + 1:]:
                if src["source_concept"] == dst["source_concept"]:
                    continue
                if per_entity[dst["entity_id"]] >= max_alias_edges_per_entity:
                    continue
                dst_toks = _entity_tokens(str(dst.get("mention_text") or ""))
                if not dst_toks:
                    continue
                jacc = len(src_toks & dst_toks) / max(len(src_toks | dst_toks), 1)
                if jacc >= 0.5:
                    candidates.append((jacc, dst))
            candidates.sort(reverse=True, key=lambda item: item[0])
            for jacc, dst in candidates[: max_alias_edges_per_entity]:
                if per_entity[src["entity_id"]] >= max_alias_edges_per_entity:
                    break
                key = tuple(sorted((src["entity_id"], dst["entity_id"]))) + ("related_to",)
                if key in seen:
                    continue
                seen.add(key)
                per_entity[src["entity_id"]] += 1
                per_entity[dst["entity_id"]] += 1
                edges.append({
                    "src_entity": src["entity_id"],
                    "dst_entity": dst["entity_id"],
                    "edge_type": "related_to",
                    "weight": round(float(jacc), 3),
                    "supporting_text": f"shared {entity_type} mention tokens",
                })


def _estimate_cost(snapshot: dict[str, dict[str, Any]]) -> float:
    usage = snapshot.get(MODEL, {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return (input_tokens / 1_000_000.0) * INPUT_COST_PER_M + (
        output_tokens / 1_000_000.0
    ) * OUTPUT_COST_PER_M


async def main_async(args: argparse.Namespace) -> int:
    if OUT_FILE.exists() and not args.force:
        print(f"ERROR: output already exists: {OUT_FILE}", file=sys.stderr)
        return 2
    if not SEED_MEM.exists():
        print(f"ERROR: seed memory not found: {SEED_MEM}", file=sys.stderr)
        return 2

    seed_payload = json.loads(SEED_MEM.read_text())
    mem = ConceptMemory.from_payload(seed_payload)
    concepts: dict[str, dict[str, Any]] = seed_payload.get("concepts", {})
    names = sorted(mem.concepts.keys())
    if args.limit_concepts:
        names = names[: args.limit_concepts]
    if not names:
        print("ERROR: no concepts loaded", file=sys.stderr)
        return 2

    prompts = [
        _build_prompt(
            concepts,
            name,
            max_entities=args.max_entities_per_concept,
            max_edges=args.max_edges_per_concept,
        )
        for name in names
    ]
    print(f"[entity_graph] concepts={len(names)} model={MODEL}")

    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "default_max_concurrency": args.concurrency,
    })
    t0 = time.monotonic()
    results = await client.async_batch_generate(
        prompts,
        MODEL,
        {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "batch_size": args.concurrency,
            "ignore_cache": args.ignore_cache,
            "extra_kwargs": {
                "extra_body": {"reasoning": {"effort": "none", "exclude": True}},
            },
        },
        request_timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()

    entities: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    entities_per_concept: list[int] = []
    edges_per_concept: list[int] = []
    for name, completions in zip(names, results, strict=True):
        raw = completions[0] if completions else ""
        try:
            raw_entities, raw_edges = _parse_response(raw or "")
        except Exception as exc:
            failures.append({"source_concept": name, "error": str(exc)[:240]})
            raw_entities, raw_edges = [], []
        if not raw_entities:
            failures.append({"source_concept": name, "error": "no_valid_entities"})
            raw_entities = _fallback_entities(
                name,
                concepts[name],
                max_entities=args.max_entities_per_concept,
            )
            raw_edges = raw_edges or []

        materialized, local_to_global = _materialize_entities(
            raw_entities,
            source_concept=name,
            start_idx=len(entities),
        )
        local_edges = _materialize_local_edges(raw_edges, local_to_global)
        entities.extend(materialized)
        edges.extend(local_edges)
        entities_per_concept.append(len(materialized))
        edges_per_concept.append(len(local_edges))

    _add_cross_concept_edges(
        entities,
        edges,
        max_alias_edges_per_entity=args.max_alias_edges_per_entity,
    )
    source_by_entity = {e["entity_id"]: e["source_concept"] for e in entities}
    cross_edges = sum(
        1 for e in edges
        if source_by_entity.get(e["src_entity"]) != source_by_entity.get(e["dst_entity"])
    )
    cost = _estimate_cost(usage)
    out = {
        "schema_version": "1",
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entities": entities,
        "edges": edges,
        "stats": {
            "num_concepts": len(names),
            "num_entities": len(entities),
            "num_edges": len(edges),
            "num_cross_concept_edges": cross_edges,
            "entities_per_concept_min": min(entities_per_concept) if entities_per_concept else 0,
            "entities_per_concept_max": max(entities_per_concept) if entities_per_concept else 0,
            "entities_per_concept_mean": statistics.fmean(entities_per_concept) if entities_per_concept else 0.0,
            "edges_per_concept_mean": statistics.fmean(edges_per_concept) if edges_per_concept else 0.0,
            "num_failures": len(failures),
            "failures": failures[:20],
            "wall_time_s": elapsed,
            "estimated_cost_usd": cost,
            "token_usage": usage,
        },
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[entity_graph] wrote {OUT_FILE}")
    print(
        f"[entity_graph] entities={len(entities)} edges={len(edges)} "
        f"cross_edges={cross_edges} failures={len(failures)} cost=${cost:.4f}"
    )
    if cost > args.max_cost_usd:
        print(
            f"ERROR: estimated cost ${cost:.4f} exceeded limit ${args.max_cost_usd:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0 if entities and edges else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--max-entities-per-concept", type=int, default=15)
    parser.add_argument("--max-edges-per-concept", type=int, default=18)
    parser.add_argument("--max-alias-edges-per-entity", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-cost-usd", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
