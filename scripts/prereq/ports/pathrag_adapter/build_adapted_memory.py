"""Build PathRAG adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/shared/entity_graph_v1.json
        data/arc_agi/concept_memory/shared/openie_facts_v1.json
Prompt: scripts/prereq/ports/pathrag_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/pathrag_memory_v1.json
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
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "pathrag_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/pathrag"

NODE_TYPES = {
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


def _clean_node(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    node_id = _clean_text(raw.get("node_id"), max_len=40)
    label = _clean_text(raw.get("label"), max_len=120)
    text_chunk = _clean_text(raw.get("text_chunk"), max_len=260)
    node_type = _clean_text(raw.get("node_type"), max_len=40)
    if not node_id or not label:
        return None
    if node_type not in NODE_TYPES:
        node_type = "other"
    return {
        "node_id": node_id,
        "label": label,
        "text_chunk": text_chunk or label,
        "node_type": node_type,
    }


def _clean_edge(raw: Any, node_ids: set[str]) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    src = _clean_text(raw.get("src"), max_len=40)
    dst = _clean_text(raw.get("dst"), max_len=40)
    relation = _clean_text(raw.get("relation"), max_len=80)
    text_chunk = _clean_text(raw.get("text_chunk"), max_len=220)
    if not src or not dst or src == dst or src not in node_ids or dst not in node_ids:
        return None
    if not relation:
        relation = "relates_to"
    return {
        "src": src,
        "dst": dst,
        "relation": relation,
        "text_chunk": text_chunk or relation,
    }


def _clean_path(raw: Any, node_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    path_id = _clean_text(raw.get("path_id"), max_len=40)
    nodes = [_clean_text(n, max_len=40) for n in raw.get("nodes", [])]
    nodes = [n for n in nodes if n in node_ids]
    edges = [_clean_edge(edge, node_ids) for edge in raw.get("edges", [])]
    edges = [edge for edge in edges if edge]
    textual_path = _clean_text(raw.get("textual_path"), max_len=1000)
    if not path_id:
        path_id = f"p{abs(hash(textual_path)) % 100000}"
    if len(nodes) < 2 or not edges or not textual_path:
        return None
    try:
        reliability = float(raw.get("reliability_hint", 0.5))
    except (TypeError, ValueError):
        reliability = 0.5
    return {
        "path_id": path_id,
        "nodes": nodes[:8],
        "edges": edges[:7],
        "textual_path": textual_path,
        "reliability_hint": max(0.0, min(1.0, reliability)),
        "pruning_rationale": _clean_text(raw.get("pruning_rationale"), max_len=260),
    }


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    keywords = [
        _clean_text(item, max_len=80)
        for item in raw.get("query_keywords", [])
        if _clean_text(item, max_len=80)
    ]
    if not keywords:
        raise ValueError(f"{concept_id}: no query_keywords")
    nodes = [_clean_node(item) for item in raw.get("path_nodes", [])]
    nodes = [node for node in nodes if node]
    node_ids = {node["node_id"] for node in nodes}
    if len(nodes) < 2:
        raise ValueError(f"{concept_id}: fewer than two path_nodes")
    paths = [_clean_path(item, node_ids) for item in raw.get("entity_paths", [])]
    paths = [path for path in paths if path]
    if not paths:
        raise ValueError(f"{concept_id}: no valid entity_paths")
    paths.sort(key=lambda p: float(p.get("reliability_hint", 0.0)))
    return {
        "concept_id": concept_id,
        "query_keywords": list(dict.fromkeys(keywords))[:12],
        "path_nodes": nodes[:14],
        "entity_paths": paths[:6],
        "answer_generation_notes": _clean_text(raw.get("answer_generation_notes"), max_len=280),
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
            except Exception as exc:  # noqa: BLE001 - raise exact concept after retries
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
    print(f"[pathrag_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _adapt_one(client, sem, name, concepts[name], prompt_template, entities_by_concept, facts_by_concept, args)
        for name in names
    ])
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "pathrag",
        "paper": "PathRAG, arXiv:2502.14902, Methodology",
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
    print(f"[pathrag_adapter] wrote {OUTPUT_PATH}")
    print(f"[pathrag_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
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
