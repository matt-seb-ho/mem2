"""Build LightRAG adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/shared/lightrag_embed_v1.json
        data/arc_agi/concept_memory/shared/entity_graph_v1.json
        data/arc_agi/concept_memory/shared/openie_facts_v1.json
Prompt: scripts/prereq/ports/lightrag_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/lightrag_memory_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.providers.llmplus_client import LLMPlusProviderClient


MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28
PROMPT_PATH = Path(__file__).parent / "prompt.md"
INPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
SHARED_DIR = ROOT / "data" / "arc_agi" / "concept_memory" / "shared"
LIGHTRAG_META_PATH = SHARED_DIR / "lightrag_embed_v1.json"
ENTITY_GRAPH_PATH = SHARED_DIR / "entity_graph_v1.json"
OPENIE_FACTS_PATH = SHARED_DIR / "openie_facts_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "lightrag_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/lightrag"


def _json_for_prompt(value: Any, *, limit: int = 1400) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "..."


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response was not a JSON object")
    return parsed


def _clean_text(value: Any, *, max_len: int) -> str:
    return str(value or "").strip()[:max_len]


def _clean_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [_clean_text(item, max_len=max_len) for item in value]
    return list(dict.fromkeys(item for item in cleaned if item))[:max_items]


def _clean_entity(value: Any, *, concept_id: str) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    mention = _clean_text(value.get("mention"), max_len=90)
    entity_type = _clean_text(value.get("entity_type"), max_len=60)
    entity_summary = _clean_text(value.get("entity_summary"), max_len=220)
    if not mention or not entity_summary:
        raise ValueError(f"{concept_id}: incomplete local entity")
    return {
        "mention": mention,
        "entity_type": entity_type or "concept",
        "entity_summary": entity_summary,
    }


def _clean_relationship(value: Any, *, concept_id: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    relation = _clean_text(value.get("relation"), max_len=90)
    target = _clean_text(value.get("target_concept"), max_len=100)
    summary = _clean_text(value.get("relation_summary"), max_len=240)
    try:
        strength = float(value.get("strength", 0.0))
    except (TypeError, ValueError):
        raise ValueError(f"{concept_id}: relationship strength was not numeric")
    if not relation or not target or not summary:
        raise ValueError(f"{concept_id}: incomplete global relationship")
    return {
        "relation": relation,
        "target_concept": target,
        "relation_summary": summary,
        "strength": max(0.0, min(1.0, strength)),
    }


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    entities = [
        entity
        for entity in (
            _clean_entity(item, concept_id=concept_id)
            for item in (raw.get("local_entities") or [])
        )
        if entity
    ][:8]
    relationships = [
        relationship
        for relationship in (
            _clean_relationship(item, concept_id=concept_id)
            for item in (raw.get("global_relationships") or [])
        )
        if relationship
    ][:8]
    low_level_keywords = _clean_list(raw.get("low_level_keywords"), max_items=12, max_len=80)
    high_level_keywords = _clean_list(raw.get("high_level_keywords"), max_items=12, max_len=80)
    one_hop_neighbors = _clean_list(raw.get("one_hop_neighbors"), max_items=12, max_len=100)
    entity_summary = _clean_text(raw.get("entity_value_summary"), max_len=420)
    relation_summary = _clean_text(raw.get("relation_value_summary"), max_len=420)
    chunk_reference = _clean_text(raw.get("chunk_reference"), max_len=320)
    if not entities:
        raise ValueError(f"{concept_id}: no local_entities")
    if not relationships:
        raise ValueError(f"{concept_id}: no global_relationships")
    if len(low_level_keywords) < 2:
        raise ValueError(f"{concept_id}: fewer than two low_level_keywords")
    if len(high_level_keywords) < 2:
        raise ValueError(f"{concept_id}: fewer than two high_level_keywords")
    if not entity_summary or not relation_summary:
        raise ValueError(f"{concept_id}: missing value summaries")
    if not chunk_reference:
        raise ValueError(f"{concept_id}: missing chunk_reference")
    return {
        "concept_id": concept_id,
        "local_entities": entities,
        "global_relationships": relationships,
        "low_level_keywords": low_level_keywords,
        "high_level_keywords": high_level_keywords,
        "entity_value_summary": entity_summary,
        "relation_value_summary": relation_summary,
        "one_hop_neighbors": one_hop_neighbors,
        "chunk_reference": chunk_reference,
        "retrieval_notes": _clean_text(raw.get("retrieval_notes"), max_len=280),
    }


def _estimate_cost(snapshot: dict[str, dict[str, Any]]) -> float:
    usage = snapshot.get(MODEL, {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return (input_tokens / 1_000_000.0) * INPUT_COST_PER_M + (output_tokens / 1_000_000.0) * OUTPUT_COST_PER_M


def _concept_fields(name: str, raw: dict[str, Any]) -> dict[str, str]:
    return {
        "concept_id": name,
        "name": str(raw.get("name") or name),
        "kind": str(raw.get("kind") or ""),
        "routine_subtype": str(raw.get("routine_subtype") or ""),
        "output_typing": str(raw.get("output_typing") or ""),
        "description": str(raw.get("description") or ""),
        "parameters": _json_for_prompt(raw.get("parameters") or []),
        "cues": _json_for_prompt(raw.get("cues") or []),
        "implementation": _json_for_prompt(raw.get("implementation") or []),
    }


def _load_lightrag_contexts() -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    if LIGHTRAG_META_PATH.exists():
        meta = json.loads(LIGHTRAG_META_PATH.read_text())
        for mention, source, entity_type in zip(
            meta.get("entity_mentions") or [],
            meta.get("entity_sources") or [],
            meta.get("entity_types") or [],
            strict=False,
        ):
            if not isinstance(source, str):
                continue
            ctx = contexts.setdefault(source, {"entities": [], "relationships": [], "openie_facts": []})
            ctx["entities"].append({
                "mention": str(mention),
                "entity_type": str(entity_type),
            })
    entity_by_id: dict[str, dict[str, Any]] = {}
    if ENTITY_GRAPH_PATH.exists():
        graph = json.loads(ENTITY_GRAPH_PATH.read_text())
        for entity in graph.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            entity_id = entity.get("entity_id")
            source = entity.get("source_concept")
            if isinstance(entity_id, str):
                entity_by_id[entity_id] = entity
            if isinstance(source, str):
                ctx = contexts.setdefault(source, {"entities": [], "relationships": [], "openie_facts": []})
                ctx["entities"].append({
                    "mention": str(entity.get("mention_text") or ""),
                    "entity_type": str(entity.get("entity_type") or ""),
                    "supporting_text": str(entity.get("supporting_text") or ""),
                })
        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            src_entity = entity_by_id.get(edge.get("src_entity"))
            dst_entity = entity_by_id.get(edge.get("dst_entity"))
            if not src_entity or not dst_entity:
                continue
            src_concept = src_entity.get("source_concept")
            dst_concept = dst_entity.get("source_concept")
            if not isinstance(src_concept, str):
                continue
            ctx = contexts.setdefault(src_concept, {"entities": [], "relationships": [], "openie_facts": []})
            ctx["relationships"].append({
                "relation": str(edge.get("edge_type") or ""),
                "target_concept": str(dst_concept or dst_entity.get("mention_text") or ""),
                "src_mention": str(src_entity.get("mention_text") or ""),
                "dst_mention": str(dst_entity.get("mention_text") or ""),
                "supporting_text": str(edge.get("supporting_text") or ""),
                "weight": edge.get("weight"),
            })
    if OPENIE_FACTS_PATH.exists():
        facts = json.loads(OPENIE_FACTS_PATH.read_text()).get("facts") or []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            source = fact.get("source_concept")
            if not isinstance(source, str):
                continue
            ctx = contexts.setdefault(source, {"entities": [], "relationships": [], "openie_facts": []})
            ctx["openie_facts"].append({
                "predicate": str(fact.get("predicate") or ""),
                "object": str(fact.get("object") or ""),
                "linked_concepts": fact.get("linked_concepts") or [],
                "supporting_text": str(fact.get("supporting_text") or ""),
                "confidence": fact.get("confidence"),
            })
    for ctx in contexts.values():
        ctx["entities"] = ctx.get("entities", [])[:14]
        ctx["relationships"] = ctx.get("relationships", [])[:12]
        ctx["openie_facts"] = ctx.get("openie_facts", [])[:8]
    return contexts


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    lightrag_contexts: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    fields["lightrag_context"] = _json_for_prompt(lightrag_contexts.get(name) or {}, limit=2400)
    prompt = prompt_template.format(**fields)
    last_error: Exception | None = None
    async with sem:
        for attempt in range(4):
            try:
                completions = await client.async_generate(
                    prompt,
                    MODEL,
                    {
                        "temperature": 0.0,
                        "max_tokens": args.max_tokens,
                        "ignore_cache": args.ignore_cache,
                        "extra_kwargs": {
                            "extra_body": {"reasoning": {"effort": "none", "exclude": True}},
                        },
                    },
                )
                raw_text = completions[0] if completions else ""
                return _clean_record(name, _extract_json_object(raw_text or ""))
            except Exception as exc:  # noqa: BLE001 - surface exact concept failure after retries
                last_error = exc
                if attempt < 3:
                    await asyncio.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"{name}: failed after 3 retries: {last_error}")


async def main_async(args: argparse.Namespace) -> int:
    if OUTPUT_PATH.exists() and not args.force and not args.smoke:
        print(f"ERROR: output already exists: {OUTPUT_PATH}", file=sys.stderr)
        return 2
    seed = json.loads(INPUT_PATH.read_text())
    concepts: dict[str, dict[str, Any]] = seed.get("concepts", {})
    names = sorted(concepts.keys())
    if args.smoke:
        names = names[:5]
    elif args.limit_concepts:
        names = names[: args.limit_concepts]
    if not names:
        print("ERROR: no concepts loaded", file=sys.stderr)
        return 2
    prompt_template = PROMPT_PATH.read_text()
    lightrag_contexts = _load_lightrag_contexts()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[lightrag_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    tasks = [
        _adapt_one(client, sem, name, concepts[name], prompt_template, lightrag_contexts, args)
        for name in names
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "lightrag",
        "source": "literature/2410.05779.pdf",
        "shared_substrate": [
            "data/arc_agi/concept_memory/shared/lightrag_embed_v1.json",
            "data/arc_agi/concept_memory/shared/entity_graph_v1.json",
            "data/arc_agi/concept_memory/shared/openie_facts_v1.json",
        ],
        "adapted_concepts": results,
        "stats": {
            "num_concepts": len(results),
            "llm_calls": len(names),
            "num_failures": 0,
            "failures": [],
            "wall_time_s": elapsed,
            "estimated_cost_usd": _estimate_cost(usage),
            "token_usage": usage,
        },
    }
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    if args.smoke:
        print(json.dumps(out["adapted_concepts"], indent=2, ensure_ascii=False))
        print(f"[lightrag_adapter] smoke cost=${out['stats']['estimated_cost_usd']:.4f}")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[lightrag_adapter] wrote {OUTPUT_PATH}")
    print(f"[lightrag_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
