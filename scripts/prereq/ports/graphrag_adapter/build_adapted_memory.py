"""Build GraphRAG adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/shared/community_summaries_v1.json
        data/arc_agi/concept_memory/shared/hierarchical_reports_v1.json
Prompt: scripts/prereq/ports/graphrag_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/graphrag_memory_v1.json
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
COMMUNITY_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "community_summaries_v1.json"
HIERARCHY_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "hierarchical_reports_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "graphrag_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/graphrag"
IMPORTANCE = {"high", "medium", "low"}


def _json_for_prompt(value: Any, *, limit: int = 1800) -> str:
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


def _clean_path_item(raw: Any, allowed_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    community_id = _clean_text(raw.get("community_id"), max_len=80)
    if community_id not in allowed_ids:
        return None
    try:
        level = int(raw.get("level", 0))
    except (TypeError, ValueError):
        level = 0
    return {
        "level": level,
        "community_id": community_id,
        "role_at_level": _clean_text(raw.get("role_at_level"), max_len=160),
        "report_connection": _clean_text(raw.get("report_connection"), max_len=260),
    }


def _clean_claim(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    claim = _clean_text(raw.get("claim"), max_len=260)
    if not claim:
        return None
    importance = _clean_text(raw.get("importance"), max_len=20).lower()
    if importance not in IMPORTANCE:
        importance = "medium"
    return {"claim": claim, "importance": importance}


def _clean_record(concept_id: str, raw: dict[str, Any], allowed_ids: set[str]) -> dict[str, Any]:
    primary = _clean_text(raw.get("primary_community_id"), max_len=80)
    if primary not in allowed_ids:
        raise ValueError(f"{concept_id}: invalid primary_community_id {primary!r}")
    contribution = _clean_text(raw.get("contribution_to_cluster"), max_len=900)
    card = _clean_text(raw.get("map_reduce_card"), max_len=700)
    if len(contribution.split()) < 8:
        raise ValueError(f"{concept_id}: contribution_to_cluster too short")
    if len(card.split()) < 6:
        raise ValueError(f"{concept_id}: map_reduce_card too short")
    summary_path = [
        item for item in (
            _clean_path_item(raw_item, allowed_ids)
            for raw_item in raw.get("summary_path", [])
        )
        if item
    ]
    if not summary_path:
        summary_path = [{
            "level": 0,
            "community_id": primary,
            "role_at_level": _clean_text(raw.get("community_role"), max_len=160),
            "report_connection": contribution[:260],
        }]
    claims = [
        item for item in (_clean_claim(raw_item) for raw_item in raw.get("entity_relationship_claims", []))
        if item
    ]
    keywords = [
        _clean_text(item, max_len=80)
        for item in raw.get("query_focus_keywords", [])
        if _clean_text(item, max_len=80)
    ]
    if not keywords:
        raise ValueError(f"{concept_id}: no query_focus_keywords")
    return {
        "concept_id": concept_id,
        "primary_community_id": primary,
        "community_role": _clean_text(raw.get("community_role"), max_len=200),
        "contribution_to_cluster": contribution,
        "map_reduce_card": card,
        "summary_path": summary_path[:4],
        "entity_relationship_claims": claims[:8],
        "query_focus_keywords": list(dict.fromkeys(keywords))[:12],
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


def _load_community_contexts() -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    data = json.loads(COMMUNITY_PATH.read_text())
    by_concept: dict[str, list[dict[str, Any]]] = {}
    ids: set[str] = set()
    for community in data.get("communities", []) or []:
        if not isinstance(community, dict):
            continue
        cid = str(community.get("community_id") or "")
        if not cid:
            continue
        ids.add(cid)
        compact = {
            "community_id": cid,
            "seed_concept": community.get("seed_concept"),
            "llm_summary": community.get("llm_summary"),
            "member_digest": community.get("member_digest"),
            "member_concepts": community.get("member_concepts", [])[:16],
        }
        for concept in community.get("member_concepts") or []:
            if isinstance(concept, str):
                by_concept.setdefault(concept, []).append(compact)
    return by_concept, ids


def _load_hierarchical_contexts() -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    data = json.loads(HIERARCHY_PATH.read_text())
    by_concept: dict[str, list[dict[str, Any]]] = {}
    ids: set[str] = set()
    for reports in (data.get("hierarchy") or {}).values():
        if not isinstance(reports, list):
            continue
        for report in reports:
            if not isinstance(report, dict):
                continue
            cid = str(report.get("community_id") or "")
            if not cid:
                continue
            ids.add(cid)
            compact = {
                "community_id": cid,
                "level": report.get("level", 0),
                "llm_summary": report.get("llm_summary"),
                "member_digest": report.get("member_digest"),
                "child_communities": report.get("child_communities", []),
                "source_concepts": report.get("source_concepts", [])[:18],
            }
            for concept in report.get("source_concepts") or []:
                if isinstance(concept, str):
                    by_concept.setdefault(concept, []).append(compact)
    return by_concept, ids


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    community_by_concept: dict[str, list[dict[str, Any]]],
    hierarchy_by_concept: dict[str, list[dict[str, Any]]],
    allowed_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    fields["community_context"] = _json_for_prompt(community_by_concept.get(name, [])[:4], limit=3000)
    fields["hierarchical_context"] = _json_for_prompt(hierarchy_by_concept.get(name, [])[:8], limit=3600)
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
                return _clean_record(name, _extract_json_object(raw_text or ""), allowed_ids)
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
    if args.smoke:
        names = names[:5]
    if args.limit_concepts:
        names = names[: args.limit_concepts]
    community_by_concept, community_ids = _load_community_contexts()
    hierarchy_by_concept, hierarchy_ids = _load_hierarchical_contexts()
    allowed_ids = community_ids | hierarchy_ids
    prompt_template = PROMPT_PATH.read_text()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[graphrag_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _adapt_one(
            client,
            sem,
            name,
            concepts[name],
            prompt_template,
            community_by_concept,
            hierarchy_by_concept,
            allowed_ids,
            args,
        )
        for name in names
    ])
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "graphrag",
        "paper": "GraphRAG, arXiv:2404.16130, Section 3.1",
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
    if args.smoke:
        print(json.dumps(out["adapted_concepts"][:5], indent=2, ensure_ascii=False))
        print(f"[graphrag_adapter] smoke_ok concepts={len(results)}")
        return 0
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[graphrag_adapter] wrote {OUTPUT_PATH}")
    print(f"[graphrag_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
