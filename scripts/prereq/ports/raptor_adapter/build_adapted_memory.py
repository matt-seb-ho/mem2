"""Build RAPTOR adapted concept memory."""
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
TREE_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "raptor_tree_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "raptor_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/raptor"


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


def _clean_path_item(raw: Any, allowed_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    node_id = _clean_text(raw.get("node_id"), max_len=80)
    if node_id not in allowed_ids:
        return None
    try:
        level = int(raw.get("level", 0))
    except (TypeError, ValueError):
        level = 0
    return {
        "level": level,
        "node_id": node_id,
        "summary_role": _clean_text(raw.get("summary_role"), max_len=220),
        "retrieval_text": _clean_text(raw.get("retrieval_text"), max_len=500),
    }


def _clean_record(
    concept_id: str,
    raw: dict[str, Any],
    allowed_ids: set[str],
    allowed_leaf_ids: set[str],
) -> dict[str, Any]:
    leaf = _clean_text(raw.get("leaf_node_id"), max_len=80)
    if leaf not in allowed_leaf_ids:
        raise ValueError(f"{concept_id}: invalid leaf_node_id {leaf!r}")
    leaf_text = _clean_text(raw.get("leaf_text"), max_len=900)
    if len(leaf_text.split()) < 6:
        raise ValueError(f"{concept_id}: leaf_text too short")
    path = [
        item for item in (
            _clean_path_item(path_item, allowed_ids)
            for path_item in raw.get("path_to_root", [])
        )
        if item
    ]
    if not path:
        raise ValueError(f"{concept_id}: no valid path_to_root")
    if path[0]["node_id"] != leaf:
        path.insert(0, {
            "level": 0,
            "node_id": leaf,
            "summary_role": _clean_text(raw.get("tree_membership_rationale"), max_len=220),
            "retrieval_text": leaf_text[:500],
        })
    keywords = [
        _clean_text(item, max_len=80)
        for item in raw.get("collapsed_tree_keywords", [])
        if _clean_text(item, max_len=80)
    ]
    cues = [
        _clean_text(item, max_len=120)
        for item in raw.get("tree_traversal_cues", [])
        if _clean_text(item, max_len=120)
    ]
    if not keywords:
        raise ValueError(f"{concept_id}: no collapsed_tree_keywords")
    return {
        "concept_id": concept_id,
        "leaf_node_id": leaf,
        "tree_membership_rationale": _clean_text(raw.get("tree_membership_rationale"), max_len=360),
        "leaf_text": leaf_text,
        "path_to_root": path[:5],
        "collapsed_tree_keywords": list(dict.fromkeys(keywords))[:12],
        "tree_traversal_cues": list(dict.fromkeys(cues))[:12],
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
        "parameters": _json_for_prompt(raw.get("parameters") or [], limit=1400),
        "cues": _json_for_prompt(raw.get("cues") or [], limit=1400),
        "implementation": _json_for_prompt(raw.get("implementation") or [], limit=1400),
    }


def _load_tree_contexts() -> tuple[dict[str, list[dict[str, Any]]], set[str], set[str]]:
    tree = json.loads(TREE_PATH.read_text())
    levels = tree.get("levels", []) or []
    nodes_by_id = {
        node.get("node_id"): {**node, "level": level.get("level", 0)}
        for level in levels
        for node in level.get("nodes", []) or []
        if isinstance(node, dict) and node.get("node_id")
    }
    parent_by_child: dict[str, list[str]] = {}
    for node_id, node in nodes_by_id.items():
        for child in node.get("child_node_ids") or node.get("member_node_ids") or []:
            if isinstance(child, str):
                parent_by_child.setdefault(child, []).append(node_id)
    allowed_ids = set(nodes_by_id)
    allowed_leaf_ids = {
        node_id for node_id, node in nodes_by_id.items()
        if int(node.get("level", 0) or 0) == 0
    }
    by_concept: dict[str, list[dict[str, Any]]] = {}
    for node_id, node in nodes_by_id.items():
        compact = {
            "node_id": node_id,
            "level": node.get("level", 0),
            "summary": node.get("summary"),
            "member_communities": node.get("member_communities", []),
            "member_concepts": node.get("member_concepts", [])[:18],
            "child_node_ids": node.get("child_node_ids") or node.get("member_node_ids") or [],
            "parent_node_ids": parent_by_child.get(node_id, []),
        }
        for concept in node.get("member_concepts") or []:
            if isinstance(concept, str):
                by_concept.setdefault(concept, []).append(compact)
    return by_concept, allowed_ids, allowed_leaf_ids


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    tree_by_concept: dict[str, list[dict[str, Any]]],
    allowed_ids: set[str],
    allowed_leaf_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    fields["tree_context"] = _json_for_prompt(tree_by_concept.get(name, [])[:8], limit=5200)
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
                return _clean_record(
                    name,
                    _extract_json_object(raw_text or ""),
                    allowed_ids,
                    allowed_leaf_ids,
                )
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
    tree_by_concept, allowed_ids, allowed_leaf_ids = _load_tree_contexts()
    prompt_template = PROMPT_PATH.read_text()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[raptor_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _adapt_one(client, sem, name, concepts[name], prompt_template, tree_by_concept, allowed_ids, allowed_leaf_ids, args)
        for name in names
    ])
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "source_tree": str(TREE_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "raptor",
        "paper": "RAPTOR, arXiv:2401.18059, Sections 3-4",
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
        print(f"[raptor_adapter] smoke_ok concepts={len(results)}")
        return 0
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[raptor_adapter] wrote {OUTPUT_PATH}")
    print(f"[raptor_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
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
