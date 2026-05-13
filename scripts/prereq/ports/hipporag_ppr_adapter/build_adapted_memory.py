"""Build HippoRAG PPR adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/shared/entity_graph_v1.json
        data/arc_agi/concept_memory/shared/openie_facts_v1.json
Prompt: scripts/prereq/ports/hipporag_ppr_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/hipporag_ppr_memory_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
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
ENTITY_GRAPH_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "entity_graph_v1.json"
OPENIE_FACTS_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "openie_facts_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "hipporag_ppr_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/hipporag_ppr"

ENTITY_TYPES = {
    "operation",
    "object",
    "attribute",
    "parameter",
    "condition",
    "transformation",
    "spatial_relation",
    "pattern",
    "output",
    "concept",
    "other",
}
SPECIFICITY = {"high", "medium", "low"}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


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


def _clean_entity(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    text = _clean_text(raw.get("text") or raw.get("mention_text"), max_len=120)
    if not text:
        return None
    typ = _clean_text(raw.get("type") or raw.get("entity_type"), max_len=40)
    if typ not in ENTITY_TYPES:
        typ = "other"
    return {
        "text": text,
        "type": typ,
        "role": _clean_text(raw.get("role"), max_len=160),
        "supporting_text": _clean_text(raw.get("supporting_text"), max_len=240),
    }


def _clean_triple(raw: Any, entity_texts: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    subject = _clean_text(raw.get("subject"), max_len=120)
    predicate = _clean_text(raw.get("predicate"), max_len=80)
    obj = _clean_text(raw.get("object"), max_len=120)
    if not subject or not predicate or not obj:
        return None
    try:
        confidence = float(raw.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": max(0.0, min(1.0, confidence)),
        "supporting_text": _clean_text(raw.get("supporting_text"), max_len=240),
        "uses_known_entity": subject in entity_texts or obj in entity_texts,
    }


def _clean_specificity(raw: Any, entity_texts: set[str]) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    node = _clean_text(raw.get("node"), max_len=120)
    specificity = _clean_text(raw.get("specificity"), max_len=20).lower()
    if not node:
        return None
    if specificity not in SPECIFICITY:
        specificity = "medium"
    return {
        "node": node,
        "specificity": specificity,
        "reason": _clean_text(raw.get("reason"), max_len=180),
        "uses_known_entity": str(node in entity_texts).lower(),
    }


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    passage = _clean_text(raw.get("passage_text"), max_len=1200)
    if len(passage.split()) < 4:
        raise ValueError(f"{concept_id}: passage_text too short")
    entities = [_clean_entity(item) for item in raw.get("entity_mentions", [])]
    entities = [item for item in entities if item]
    if len(entities) < 2:
        raise ValueError(f"{concept_id}: fewer than two valid entity_mentions")
    entity_texts = {item["text"] for item in entities}
    triples = [_clean_triple(item, entity_texts) for item in raw.get("triples", [])]
    triples = [item for item in triples if item]
    if not triples:
        raise ValueError(f"{concept_id}: no valid triples")
    query_terms = [
        _clean_text(item, max_len=80)
        for item in raw.get("query_node_terms", [])
        if _clean_text(item, max_len=80)
    ]
    if not query_terms:
        raise ValueError(f"{concept_id}: no query_node_terms")
    specificity = [
        _clean_specificity(item, entity_texts)
        for item in raw.get("node_specificity_hints", [])
    ]
    specificity = [item for item in specificity if item]
    return {
        "concept_id": concept_id,
        "passage_text": passage,
        "entity_mentions": entities[:12],
        "triples": triples[:8],
        "query_node_terms": list(dict.fromkeys(query_terms))[:12],
        "node_specificity_hints": specificity[:8],
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


def _load_contexts() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    entity_data = json.loads(ENTITY_GRAPH_PATH.read_text()) if ENTITY_GRAPH_PATH.exists() else {}
    fact_data = json.loads(OPENIE_FACTS_PATH.read_text()) if OPENIE_FACTS_PATH.exists() else {}
    entities_by_concept: dict[str, list[dict[str, Any]]] = {}
    for raw in entity_data.get("entities", []) or []:
        if isinstance(raw, dict) and isinstance(raw.get("source_concept"), str):
            entities_by_concept.setdefault(raw["source_concept"], []).append(raw)
    facts_by_concept: dict[str, list[dict[str, Any]]] = {}
    for raw in fact_data.get("facts", []) or []:
        if isinstance(raw, dict) and isinstance(raw.get("source_concept"), str):
            facts_by_concept.setdefault(raw["source_concept"], []).append(raw)
    return entities_by_concept, facts_by_concept


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    entities_by_concept: dict[str, list[dict[str, Any]]],
    facts_by_concept: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    fields["entity_context"] = _json_for_prompt(entities_by_concept.get(name, [])[:12], limit=2200)
    fields["fact_context"] = _json_for_prompt(facts_by_concept.get(name, [])[:8], limit=1800)
    prompt = prompt_template.format(**fields)
    last_error: Exception | None = None
    async with sem:
        for attempt in range(1, 4):
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
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"{name}: failed after 3 attempts: {last_error}")


async def main_async(args: argparse.Namespace) -> int:
    if OUTPUT_PATH.exists() and not args.force:
        print(f"ERROR: output already exists: {OUTPUT_PATH}", file=sys.stderr)
        return 2
    seed = json.loads(INPUT_PATH.read_text())
    concepts: dict[str, dict[str, Any]] = seed.get("concepts", {})
    names = sorted(concepts.keys())
    if args.limit_concepts:
        names = names[: args.limit_concepts]
    if not names:
        print("ERROR: no concepts loaded", file=sys.stderr)
        return 2
    prompt_template = PROMPT_PATH.read_text()
    entities_by_concept, facts_by_concept = _load_contexts()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[hipporag_ppr_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    tasks = [
        _adapt_one(client, sem, name, concepts[name], prompt_template, entities_by_concept, facts_by_concept, args)
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
        "port": "hipporag_ppr",
        "paper": "HippoRAG, arXiv:2405.14831, Sections 2.2-2.3",
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[hipporag_ppr_adapter] wrote {OUTPUT_PATH}")
    print(f"[hipporag_ppr_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
