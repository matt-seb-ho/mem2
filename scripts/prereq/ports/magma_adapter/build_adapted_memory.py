"""Build MAGMA adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
        data/arc_agi/concept_memory/shared/magma_typed_views_v1.json
Prompt: scripts/prereq/ports/magma_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/magma_memory_v1.json
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
TYPED_VIEWS_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "magma_typed_views_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "magma_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/magma"
VIEWS = {"semantic", "temporal", "causal", "entity", "structural"}
INTENTS = {"WHY", "WHEN", "ENTITY", "SEMANTIC", "STRUCTURAL"}


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


def _clean_event_node(raw: Any, concept_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{concept_id}: event_node must be an object")
    content = _clean_text(raw.get("content"), max_len=900)
    if len(content.split()) < 6:
        raise ValueError(f"{concept_id}: event_node.content too short")
    attrs = _clean_string_list(raw.get("attributes"), max_len=120, limit=12)
    if not attrs:
        raise ValueError(f"{concept_id}: event_node.attributes empty")
    return {
        "content": content,
        "timestamp_hint": _clean_text(raw.get("timestamp_hint"), max_len=180),
        "attributes": attrs,
    }


def _clean_membership(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    view = _clean_text(raw.get("view"), max_len=40).lower()
    if view not in VIEWS:
        return None
    node_refs = _clean_string_list(raw.get("node_refs"), max_len=160, limit=12)
    edge_refs = _clean_string_list(raw.get("edge_refs"), max_len=180, limit=12)
    role = _clean_text(raw.get("role"), max_len=360)
    traversal_value = _clean_text(raw.get("traversal_value"), max_len=420)
    intents = [
        item.upper()
        for item in _clean_string_list(raw.get("query_intents"), max_len=40, limit=8)
        if item.upper() in INTENTS
    ]
    if not role or not traversal_value:
        return None
    return {
        "view": view,
        "node_refs": node_refs,
        "edge_refs": edge_refs,
        "role": role,
        "traversal_value": traversal_value,
        "query_intents": list(dict.fromkeys(intents)) or ["SEMANTIC"],
    }


def _clean_policy(raw: Any, memberships: list[dict[str, Any]]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    preferred = [
        view
        for view in _clean_string_list(raw.get("preferred_views"), max_len=40, limit=5)
        if view.lower() in VIEWS
    ]
    if not preferred:
        preferred = [m["view"] for m in memberships]
    return {
        "preferred_views": list(dict.fromkeys(view.lower() for view in preferred))[:5],
        "why_signal": _clean_text(raw.get("why_signal"), max_len=280),
        "when_signal": _clean_text(raw.get("when_signal"), max_len=220),
        "entity_signal": _clean_text(raw.get("entity_signal"), max_len=280),
    }


def _clean_salience(raw: Any) -> dict[str, list[str]]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "keep_full": _clean_string_list(raw.get("keep_full"), max_len=160, limit=8),
        "summarize_if_needed": _clean_string_list(raw.get("summarize_if_needed"), max_len=160, limit=8),
    }


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    event_node = _clean_event_node(raw.get("event_node"), concept_id)
    memberships = [
        item for item in (_clean_membership(raw_item) for raw_item in raw.get("view_memberships", []))
        if item
    ]
    if len(memberships) < 2:
        raise ValueError(f"{concept_id}: expected at least two valid view_memberships")
    anchors = _clean_string_list(raw.get("anchor_keywords"), max_len=80, limit=14)
    if not anchors:
        raise ValueError(f"{concept_id}: anchor_keywords empty")
    card = _clean_text(raw.get("graph_linearization_card"), max_len=900)
    if len(card.split()) < 8:
        raise ValueError(f"{concept_id}: graph_linearization_card too short")
    return {
        "concept_id": concept_id,
        "event_node": event_node,
        "view_memberships": memberships[:6],
        "anchor_keywords": anchors,
        "policy_hints": _clean_policy(raw.get("policy_hints"), memberships),
        "graph_linearization_card": card,
        "salience_budget": _clean_salience(raw.get("salience_budget")),
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


def _load_typed_view_contexts() -> dict[str, list[dict[str, Any]]]:
    data = json.loads(TYPED_VIEWS_PATH.read_text())
    views = data.get("views") or {}
    by_concept: dict[str, list[dict[str, Any]]] = {}
    for view_name, view in views.items():
        if not isinstance(view, dict) or view_name not in VIEWS:
            continue
        node_labels: dict[str, str] = {}
        for node in view.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id") or "")
            if not node_id:
                continue
            node_labels[node_id] = " ".join(
                str(node.get(key) or "")
                for key in ("label", "node_type", "kind", "source_concept")
            )
        for edge in view.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src") or "")
            dst = str(edge.get("dst") or "")
            concepts = [
                node.split("concept::", 1)[1]
                for node in (src, dst)
                if node.startswith("concept::")
            ]
            if not concepts:
                continue
            compact = {
                "view": view_name,
                "src": src,
                "dst": dst,
                "src_label": node_labels.get(src, ""),
                "dst_label": node_labels.get(dst, ""),
                "edge_type": edge.get("edge_type"),
                "weight": edge.get("weight"),
                "supporting_text": edge.get("supporting_text"),
                "predicates": edge.get("predicates", []),
                "relation_types": edge.get("relation_types", []),
            }
            for concept in concepts:
                by_concept.setdefault(concept, []).append(compact)
    return by_concept


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    typed_contexts: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = _concept_fields(name, raw)
    fields["typed_view_context"] = _json_for_prompt(typed_contexts.get(name, [])[:14], limit=6500)
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
    typed_contexts = _load_typed_view_contexts()
    prompt_template = PROMPT_PATH.read_text()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[magma_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _adapt_one(client, sem, name, concepts[name], prompt_template, typed_contexts, args)
        for name in names
    ])
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "source_typed_views": str(TYPED_VIEWS_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "magma",
        "paper": "MAGMA, arXiv:2601.03236, Sections 3.2-3.3",
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
        print(f"[magma_adapter] smoke_ok concepts={len(results)}")
        return 0
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[magma_adapter] wrote {OUTPUT_PATH}")
    print(f"[magma_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
