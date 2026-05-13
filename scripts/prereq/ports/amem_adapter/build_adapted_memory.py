"""Build A-Mem adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
Prompt: scripts/prereq/ports/amem_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/amem_memory_v1.json
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
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "amem_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/amem"
LINK_TYPES = {
    "generalizes",
    "specializes",
    "prerequisite_of",
    "contrast_with",
    "applied_with",
    "similar_to",
    "updates_context_of",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


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


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _concept_text(name: str, raw: dict[str, Any]) -> str:
    parts = [
        name,
        str(raw.get("name") or ""),
        str(raw.get("kind") or ""),
        str(raw.get("routine_subtype") or ""),
        str(raw.get("output_typing") or ""),
        str(raw.get("description") or ""),
        _json_for_prompt(raw.get("parameters") or [], limit=1200),
        _json_for_prompt(raw.get("cues") or [], limit=1200),
        _json_for_prompt(raw.get("implementation") or [], limit=1200),
    ]
    return "\n".join(part for part in parts if part)


def _candidate_neighbors(
    name: str,
    raw: dict[str, Any],
    concepts: dict[str, dict[str, Any]],
    *,
    limit: int = 18,
) -> list[dict[str, Any]]:
    source_toks = _tokens(_concept_text(name, raw))
    source_used = set(raw.get("used_in") or [])
    scored: list[tuple[float, str]] = []
    for other_name, other_raw in concepts.items():
        if other_name == name:
            continue
        overlap_used = len(source_used & set(other_raw.get("used_in") or []))
        overlap_text = len(source_toks & _tokens(_concept_text(other_name, other_raw)))
        same_kind = 1 if raw.get("kind") and raw.get("kind") == other_raw.get("kind") else 0
        score = overlap_used * 2.0 + overlap_text * 0.2 + same_kind * 0.5
        scored.append((score, other_name))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    neighbors = []
    for _score, other_name in scored[:limit]:
        other = concepts[other_name]
        neighbors.append({
            "concept_id": other_name,
            "kind": other.get("kind"),
            "routine_subtype": other.get("routine_subtype"),
            "description": str(other.get("description") or "")[:240],
            "parameters": other.get("parameters", [])[:4],
            "shared_used_in_count": len(source_used & set(other.get("used_in") or [])),
        })
    return neighbors


def _clean_note(raw: Any, concept_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{concept_id}: note must be an object")
    content = _clean_text(raw.get("content"), max_len=900)
    context = _clean_text(raw.get("contextual_description"), max_len=900)
    if len(content.split()) < 6:
        raise ValueError(f"{concept_id}: note.content too short")
    if len(context.split()) < 8:
        raise ValueError(f"{concept_id}: note.contextual_description too short")
    keywords = _clean_string_list(raw.get("keywords"), max_len=80, limit=14)
    tags = _clean_string_list(raw.get("tags"), max_len=64, limit=12)
    if not keywords:
        raise ValueError(f"{concept_id}: note.keywords empty")
    if not tags:
        raise ValueError(f"{concept_id}: note.tags empty")
    return {
        "content": content,
        "timestamp": _clean_text(raw.get("timestamp"), max_len=120),
        "keywords": keywords,
        "tags": tags,
        "contextual_description": context,
    }


def _clean_link(raw: Any, allowed_targets: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    target = _clean_text(raw.get("target_concept"), max_len=120)
    if target not in allowed_targets:
        return None
    link_type = _clean_text(raw.get("link_type"), max_len=60)
    if link_type not in LINK_TYPES:
        link_type = "similar_to"
    rationale = _clean_text(raw.get("rationale"), max_len=320)
    if len(rationale.split()) < 4:
        return None
    try:
        confidence = float(raw.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75
    confidence = max(0.0, min(1.0, confidence))
    return {
        "target_concept": target,
        "link_type": link_type,
        "rationale": rationale,
        "confidence": confidence,
    }


def _clean_evolution(raw: Any, allowed_targets: set[str]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    suggestions: list[dict[str, str]] = []
    for item in raw.get("neighbor_update_suggestions") or []:
        if not isinstance(item, dict):
            continue
        target = _clean_text(item.get("target_concept"), max_len=120)
        update = _clean_text(item.get("suggested_update"), max_len=260)
        if target in allowed_targets and update:
            suggestions.append({"target_concept": target, "suggested_update": update})
        if len(suggestions) >= 6:
            break
    return {
        "context_update": _clean_text(raw.get("context_update"), max_len=500),
        "tag_updates": _clean_string_list(raw.get("tag_updates"), max_len=64, limit=10),
        "neighbor_update_suggestions": suggestions,
    }


def _clean_record(concept_id: str, raw: dict[str, Any], allowed_targets: set[str]) -> dict[str, Any]:
    note = _clean_note(raw.get("note"), concept_id)
    links = [
        link for link in (
            _clean_link(item, allowed_targets)
            for item in raw.get("zettel_links", [])
        )
        if link
    ]
    deduped_links: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for link in links:
        key = (link["target_concept"], link["link_type"])
        if key in seen:
            continue
        seen.add(key)
        deduped_links.append(link)
    if not deduped_links:
        raise ValueError(f"{concept_id}: no valid zettel_links")
    retrieval_text = _clean_text(raw.get("retrieval_text"), max_len=900)
    if len(retrieval_text.split()) < 6:
        raise ValueError(f"{concept_id}: retrieval_text too short")
    return {
        "concept_id": concept_id,
        "note": note,
        "zettel_links": deduped_links[:6],
        "memory_evolution": _clean_evolution(raw.get("memory_evolution"), allowed_targets),
        "retrieval_text": retrieval_text,
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


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    concepts: dict[str, dict[str, Any]],
    prompt_template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = _candidate_neighbors(name, raw, concepts)
    allowed_targets = {item["concept_id"] for item in candidates}
    fields = _concept_fields(name, raw)
    fields["candidate_neighbors"] = _json_for_prompt(candidates, limit=6500)
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
                return _clean_record(name, _extract_json_object(raw_text or ""), allowed_targets)
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
    prompt_template = PROMPT_PATH.read_text()
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[amem_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _adapt_one(client, sem, name, concepts[name], concepts, prompt_template, args)
        for name in names
    ])
    elapsed = time.monotonic() - t0
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": "amem",
        "paper": "A-Mem, arXiv:2502.12110, Sections 3.1-3.4",
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
        print(f"[amem_adapter] smoke_ok concepts={len(results)}")
        return 0
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[amem_adapter] wrote {OUTPUT_PATH}")
    print(f"[amem_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
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
