"""Build H-MEM hierarchical adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/concept_hierarchy_v1.json
Prompt: scripts/prereq/ports/hmem_hierarchical_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/hmem_memory_v1.json
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
HIERARCHY_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_hierarchy_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "hmem_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/hmem_hierarchical"
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


def _clean_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [_clean_text(item, max_len=max_len) for item in value]
    return list(dict.fromkeys(item for item in cleaned if item))[:max_items]


def _slug(value: str) -> str:
    words = WORD_RE.findall(value.lower().replace("_", " "))
    return "-".join(words[:5]) or "unknown"


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    category = _clean_text(raw.get("category"), max_len=100)
    subcategory = _clean_text(raw.get("subcategory"), max_len=100)
    if not category:
        raise ValueError(f"{concept_id}: missing category")
    if not subcategory:
        raise ValueError(f"{concept_id}: missing subcategory")
    trace_raw = raw.get("memory_trace")
    episode_raw = raw.get("episode")
    if not isinstance(trace_raw, dict):
        raise ValueError(f"{concept_id}: memory_trace was not an object")
    if not isinstance(episode_raw, dict):
        raise ValueError(f"{concept_id}: episode was not an object")
    trace_title = _clean_text(trace_raw.get("title"), max_len=120)
    trace_summary = _clean_text(trace_raw.get("trace_summary"), max_len=320)
    trace_keywords = _clean_list(trace_raw.get("keywords"), max_items=8, max_len=80)
    episode_summary = _clean_text(episode_raw.get("summary"), max_len=520)
    grounded_operations = _clean_list(
        episode_raw.get("grounded_operations"), max_items=8, max_len=90
    )
    route_here = _clean_text(episode_raw.get("when_to_route_here"), max_len=260)
    routing_keywords = _clean_list(raw.get("routing_keywords"), max_items=12, max_len=80)
    if not trace_title or not trace_summary or len(trace_keywords) < 2:
        raise ValueError(f"{concept_id}: incomplete memory_trace")
    if len(episode_summary.split()) < 5 or not route_here:
        raise ValueError(f"{concept_id}: incomplete episode")
    if len(routing_keywords) < 2:
        raise ValueError(f"{concept_id}: fewer than two routing_keywords")
    try:
        confidence = float(raw.get("confidence_weight", 0.0))
    except (TypeError, ValueError):
        raise ValueError(f"{concept_id}: confidence_weight was not numeric")
    return {
        "concept_id": concept_id,
        "domain": "ARC-AGI",
        "category": category,
        "category_position_index": _clean_text(
            raw.get("category_position_index"), max_len=60
        ) or f"L1:{_slug(category)}",
        "subcategory": subcategory,
        "subcategory_position_index": _clean_text(
            raw.get("subcategory_position_index"), max_len=60
        ) or f"L2:{_slug(category)}:{_slug(subcategory)}",
        "memory_trace": {
            "title": trace_title,
            "keywords": trace_keywords,
            "trace_summary": trace_summary,
        },
        "episode": {
            "summary": episode_summary,
            "grounded_operations": grounded_operations,
            "when_to_route_here": route_here,
        },
        "routing_keywords": routing_keywords,
        "confidence_weight": max(0.0, min(1.0, confidence)),
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


def _load_hierarchy_contexts() -> dict[str, dict[str, str]]:
    if not HIERARCHY_PATH.exists():
        return {}
    data = json.loads(HIERARCHY_PATH.read_text())
    contexts: dict[str, dict[str, str]] = {}
    for cat_index, cat in enumerate(data.get("categories") or [], start=1):
        if not isinstance(cat, dict):
            continue
        cat_name = str(cat.get("name") or "")
        cat_desc = str(cat.get("description") or "")
        for sub_index, sub in enumerate(cat.get("subcategories") or [], start=1):
            if not isinstance(sub, dict):
                continue
            sub_name = str(sub.get("name") or "")
            sub_desc = str(sub.get("description") or "")
            for concept in sub.get("concepts") or []:
                if isinstance(concept, str):
                    contexts[concept] = {
                        "category": cat_name,
                        "category_description": cat_desc,
                        "category_position_index": f"L1:{cat_index}:{_slug(cat_name)}",
                        "subcategory": sub_name,
                        "subcategory_description": sub_desc,
                        "subcategory_position_index": f"L2:{cat_index}.{sub_index}:{_slug(sub_name)}",
                    }
    return contexts


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    hierarchy_contexts: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    fields["hierarchy_context"] = _json_for_prompt(hierarchy_contexts.get(name) or {}, limit=1200)
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
    hierarchy_contexts = _load_hierarchy_contexts()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[hmem_hierarchical_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    tasks = [
        _adapt_one(client, sem, name, concepts[name], prompt_template, hierarchy_contexts, args)
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
        "port": "hmem_hierarchical",
        "paper": "H-MEM, arXiv:2507.22925, Sections 3.1-3.2",
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
        print(f"[hmem_hierarchical_adapter] smoke cost=${out['stats']['estimated_cost_usd']:.4f}")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[hmem_hierarchical_adapter] wrote {OUTPUT_PATH}")
    print(f"[hmem_hierarchical_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
