"""Build MemTree adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/shared/hierarchical_reports_v1.json
Prompt: scripts/prereq/ports/memtree_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/memtree_memory_v1.json
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
HIERARCHY_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "hierarchical_reports_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "memtree_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/memtree"
INSERTION_DECISIONS = {"traverse_deeper", "create_new_leaf", "expand_leaf"}


def _json_for_prompt(value: Any, *, limit: int = 3000) -> str:
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


def _clean_string_list(value: Any, *, max_len: int, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _clean_text(item, max_len=max_len)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _clean_path_item(raw: Any, concept_id: str, allowed_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    node_id = _clean_text(raw.get("node_id"), max_len=140)
    leaf_id = f"memtree::{concept_id}"
    if node_id != leaf_id and node_id not in allowed_ids:
        return None
    try:
        depth = int(raw.get("depth", 0))
    except (TypeError, ValueError):
        depth = 0
    return {
        "node_id": node_id,
        "depth": depth,
        "content_summary": _clean_text(raw.get("content_summary"), max_len=420),
        "update_role": _clean_text(raw.get("update_role"), max_len=300),
    }


def _clean_record(concept_id: str, raw: dict[str, Any], allowed_ids: set[str]) -> dict[str, Any]:
    position = raw.get("tree_position") if isinstance(raw.get("tree_position"), dict) else {}
    leaf_id = _clean_text(position.get("leaf_node_id"), max_len=140) or f"memtree::{concept_id}"
    if leaf_id != f"memtree::{concept_id}":
        leaf_id = f"memtree::{concept_id}"
    parent = _clean_text(position.get("parent_node_id"), max_len=120)
    if parent not in allowed_ids:
        raise ValueError(f"{concept_id}: invalid parent_node_id {parent!r}")
    try:
        depth = int(position.get("depth", 2))
    except (TypeError, ValueError):
        depth = 2
    decision = _clean_text(position.get("insertion_decision"), max_len=40)
    if decision not in INSERTION_DECISIONS:
        decision = "traverse_deeper"
    rationale = _clean_text(position.get("depth_threshold_rationale"), max_len=500)
    if len(rationale.split()) < 5:
        raise ValueError(f"{concept_id}: depth_threshold_rationale too short")

    node_content = raw.get("node_content") if isinstance(raw.get("node_content"), dict) else {}
    leaf_content = _clean_text(node_content.get("leaf_content"), max_len=900)
    embedding_text = _clean_text(node_content.get("embedding_text"), max_len=700)
    aggregate = _clean_text(node_content.get("aggregate_contribution"), max_len=600)
    if len(leaf_content.split()) < 6:
        raise ValueError(f"{concept_id}: leaf_content too short")
    if len(embedding_text.split()) < 5:
        raise ValueError(f"{concept_id}: embedding_text too short")

    path = [
        item for item in (
            _clean_path_item(path_item, concept_id, allowed_ids)
            for path_item in raw.get("path_to_root", [])
        )
        if item
    ]
    if not path or path[0]["node_id"] != leaf_id:
        path.insert(0, {
            "node_id": leaf_id,
            "depth": depth,
            "content_summary": leaf_content[:420],
            "update_role": aggregate[:300],
        })
    if not any(item["node_id"] == parent for item in path):
        path.append({
            "node_id": parent,
            "depth": max(0, depth - 1),
            "content_summary": aggregate[:420],
            "update_role": "parent aggregate receives this concept leaf",
        })
    card = _clean_text(raw.get("collapsed_retrieval_card"), max_len=900)
    if len(card.split()) < 8:
        raise ValueError(f"{concept_id}: collapsed_retrieval_card too short")
    keywords = _clean_string_list(raw.get("retrieval_keywords"), max_len=80, limit=14)
    if not keywords:
        raise ValueError(f"{concept_id}: retrieval_keywords empty")
    sibling_group = raw.get("sibling_group") if isinstance(raw.get("sibling_group"), dict) else {}
    return {
        "concept_id": concept_id,
        "tree_position": {
            "leaf_node_id": leaf_id,
            "parent_node_id": parent,
            "depth": max(1, depth),
            "insertion_decision": decision,
            "depth_threshold_rationale": rationale,
        },
        "node_content": {
            "leaf_content": leaf_content,
            "embedding_text": embedding_text,
            "aggregate_contribution": aggregate,
        },
        "path_to_root": path[:5],
        "collapsed_retrieval_card": card,
        "retrieval_keywords": keywords,
        "sibling_group": {
            "sibling_role": _clean_text(sibling_group.get("sibling_role"), max_len=320),
            "near_sibling_concepts": _clean_string_list(
                sibling_group.get("near_sibling_concepts"),
                max_len=120,
                limit=10,
            ),
        },
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
        "parameters": _json_for_prompt(raw.get("parameters") or [], limit=1800),
        "cues": _json_for_prompt(raw.get("cues") or [], limit=1600),
        "implementation": _json_for_prompt(raw.get("implementation") or [], limit=1600),
        "used_in_count": str(len(raw.get("used_in") or [])),
    }


def _load_report_contexts() -> tuple[dict[str, list[dict[str, Any]]], set[str], list[dict[str, Any]]]:
    data = json.loads(HIERARCHY_PATH.read_text())
    by_concept: dict[str, list[dict[str, Any]]] = {}
    ids: set[str] = set()
    default_contexts: list[dict[str, Any]] = []
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
            if int(report.get("level", 0) or 0) >= 1:
                default_contexts.append(compact)
            for concept in report.get("source_concepts") or []:
                if isinstance(concept, str):
                    by_concept.setdefault(concept, []).append(compact)
    default_contexts.sort(key=lambda item: (int(item.get("level", 0) or 0), str(item.get("community_id"))))
    return by_concept, ids, default_contexts[-6:]


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    report_contexts: dict[str, list[dict[str, Any]]],
    allowed_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    contexts = report_contexts.get(name, []) or report_contexts.get("__default__", [])
    fields["report_context"] = _json_for_prompt(contexts[:8], limit=6500)
    allowed_parent_ids = [
        str(context.get("community_id"))
        for context in contexts[:8]
        if isinstance(context, dict) and str(context.get("community_id") or "").strip()
    ]
    fields["allowed_parent_ids"] = ", ".join(allowed_parent_ids)
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
    if OUTPUT_PATH.exists() and not args.force and not args.smoke:
        print(f"ERROR: output already exists: {OUTPUT_PATH}", file=sys.stderr)
        return 2
    seed = json.loads(INPUT_PATH.read_text())
    concepts: dict[str, dict[str, Any]] = seed.get("concepts", {})
    names = sorted(concepts.keys())
    if args.smoke:
        names = names[:5]
    if args.limit_concepts:
        names = names[: args.limit_concepts]
    report_contexts, allowed_ids, default_contexts = _load_report_contexts()
    report_contexts["__default__"] = default_contexts
    prompt_template = PROMPT_PATH.read_text()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[memtree_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _adapt_one(client, sem, name, concepts[name], prompt_template, report_contexts, allowed_ids, args)
        for name in names
    ])
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "source_hierarchy": str(HIERARCHY_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "memtree",
        "paper": "MemTree, arXiv:2410.14052, Sections 3.1-3.2",
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
        print(f"[memtree_adapter] smoke_ok concepts={len(results)}")
        return 0
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[memtree_adapter] wrote {OUTPUT_PATH}")
    print(f"[memtree_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1900)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
