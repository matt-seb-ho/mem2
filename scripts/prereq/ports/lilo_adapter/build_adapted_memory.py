"""Build LILO best-effort adapted concept memory."""
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


PORT = "lilo"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28
PROMPT_PATH = Path(__file__).parent / "prompt.md"
INPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "lilo_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/lilo"
SMOKE_PATH = Path("/private/tmp/mem2_per_port_adapters/lilo_smoke.json")
SUBSTRATE_GAP = (
    "Best-effort partial: ARC concept records are not executable LILO DSL "
    "programs or task frontiers; no DreamCoder/Stitch compression, AutoDoc, "
    "or full dual-system library-learning loop is run."
)


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


def _clean_text_list(raw: Any, *, max_len: int, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = [_clean_text(item, max_len=max_len) for item in raw]
    return list(dict.fromkeys(item for item in out if item))[:limit]


def _clean_grounding(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    phrase = _clean_text(raw.get("phrase"), max_len=120)
    grounding = _clean_text(raw.get("grounding"), max_len=240)
    if not phrase or not grounding:
        return None
    return {"phrase": phrase, "grounding": grounding}


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    profile = _clean_text(raw.get("library_profile"), max_len=1000)
    proposal = raw.get("abstraction_proposal")
    if not isinstance(proposal, dict):
        proposal = {}
    cleaned_proposal = {
        "readable_name_hint": _clean_text(proposal.get("readable_name_hint") or f"lilo_{concept_id}", max_len=120),
        "members_or_roles": _clean_text_list(proposal.get("members_or_roles"), max_len=120, limit=10),
        "function_expression_hint": _clean_text(proposal.get("function_expression_hint"), max_len=360),
        "description": _clean_text(proposal.get("description"), max_len=700),
    }
    grounding = [_clean_grounding(item) for item in raw.get("language_grounding", [])]
    grounding = [item for item in grounding if item]
    terms = _clean_text_list(raw.get("abstraction_terms"), max_len=80, limit=12)
    if len(profile.split()) < 5:
        raise ValueError(f"{concept_id}: library_profile too short")
    if len(cleaned_proposal["description"].split()) < 5:
        raise ValueError(f"{concept_id}: abstraction description too short")
    if not grounding:
        raise ValueError(f"{concept_id}: no language_grounding")
    if not terms:
        raise ValueError(f"{concept_id}: no abstraction_terms")
    return {
        "concept_id": concept_id,
        "library_profile": profile,
        "abstraction_proposal": cleaned_proposal,
        "language_grounding": grounding[:8],
        "abstraction_terms": terms,
        "iterative_growth_notes": _clean_text(raw.get("iterative_growth_notes"), max_len=360),
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


async def _adapt_one(
    client: LLMPlusProviderClient,
    sem: asyncio.Semaphore,
    name: str,
    raw: dict[str, Any],
    prompt_template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = prompt_template.format(**_concept_fields(name, raw))
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
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"{name}: failed after 3 attempts: {last_error}")


async def main_async(args: argparse.Namespace) -> int:
    output_path = SMOKE_PATH if args.smoke else OUTPUT_PATH
    if output_path.exists() and not args.force:
        print(f"ERROR: output already exists: {output_path}", file=sys.stderr)
        return 2
    seed = json.loads(INPUT_PATH.read_text())
    concepts: dict[str, dict[str, Any]] = seed.get("concepts", {})
    names = sorted(concepts.keys())
    if args.smoke:
        names = names[:5]
    elif args.limit_concepts:
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
    print(f"[{PORT}_adapter] concepts={len(names)} model={MODEL} concurrency={args.concurrency}")
    t0 = time.monotonic()
    raw_results = await asyncio.gather(*[
        _adapt_one(client, sem, name, concepts[name], prompt_template, args)
        for name in names
    ], return_exceptions=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name, result in zip(names, raw_results, strict=True):
        if isinstance(result, Exception):
            failures.append({"concept_id": name, "error": str(result)[:500]})
        else:
            results.append(result)
    if failures and args.smoke:
        raise RuntimeError(f"smoke failed on {len(failures)} concepts: {failures[:3]}")
    if not results:
        raise RuntimeError(f"all concepts failed: {failures[:3]}")
    usage = client.get_usage_snapshot()
    out = {
        "schema_version": "1",
        "source_seed": str(INPUT_PATH.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": PORT,
        "paper": "LILO, arXiv:2310.19791",
        "substrate_gap": SUBSTRATE_GAP,
        "adapted_concepts": results,
        "stats": {
            "num_concepts": len(results),
            "llm_calls": len(names),
            "num_failures": len(failures),
            "failures": failures,
            "wall_time_s": time.monotonic() - t0,
            "estimated_cost_usd": _estimate_cost(usage),
            "token_usage": usage,
            "smoke": bool(args.smoke),
        },
    }
    if out["stats"]["estimated_cost_usd"] > args.max_cost_usd:
        raise RuntimeError(
            f"estimated cost ${out['stats']['estimated_cost_usd']:.4f} exceeded limit ${args.max_cost_usd:.2f}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{PORT}_adapter] wrote {output_path}")
    print(f"[{PORT}_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
