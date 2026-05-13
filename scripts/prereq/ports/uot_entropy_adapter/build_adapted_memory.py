"""Build UoT entropy adapted concept memory.

Reads:  data/arc_agi/concept_memory/compressed_v1.json
Prompt: scripts/prereq/ports/uot_entropy_adapter/prompt.md
Writes: data/arc_agi/concept_memory/ports/uot_memory_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
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
OUTPUT_PATH = ROOT / "data" / "arc_agi" / "concept_memory" / "ports" / "uot_memory_v1.json"
CACHE_DIR = "/private/tmp/mem2_per_port_adapters/uot_entropy"
SIMULATION_TREE_ROLES = {
    "root_candidate",
    "answer_branch",
    "questioner_branch",
    "terminal_commit",
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


def _clean_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [_clean_text(item, max_len=max_len) for item in value]
    return list(dict.fromkeys(item for item in cleaned if item))[:max_items]


def _bounded_float(value: Any, *, field: str, concept_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{concept_id}: {field} was not numeric")
    return max(0.0, min(1.0, parsed))


def _entropy_reward(x: float, lamb: float = 0.4) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    h = -x * math.log2(x) - (1 - x) * math.log2(1 - x)
    return h / (1.0 + abs(2 * x - 1) / lamb)


def _clean_record(concept_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    uncertainty_state = _clean_text(raw.get("uncertainty_state"), max_len=260)
    candidate_question = _clean_text(raw.get("candidate_question"), max_len=200)
    yes_partition_hint = _clean_list(raw.get("yes_partition_hint"), max_items=6, max_len=120)
    no_partition_hint = _clean_list(raw.get("no_partition_hint"), max_items=6, max_len=120)
    information_gain_target = _clean_text(raw.get("information_gain_target"), max_len=260)
    role = _clean_text(raw.get("simulation_tree_role"), max_len=40)
    routing_keywords = _clean_list(raw.get("routing_keywords"), max_items=12, max_len=80)
    if role not in SIMULATION_TREE_ROLES:
        role = "other"
    if not uncertainty_state:
        raise ValueError(f"{concept_id}: missing uncertainty_state")
    if not candidate_question or "?" not in candidate_question:
        raise ValueError(f"{concept_id}: candidate_question was not a question")
    if not yes_partition_hint or not no_partition_hint:
        raise ValueError(f"{concept_id}: missing partition hints")
    if not information_gain_target:
        raise ValueError(f"{concept_id}: missing information_gain_target")
    if len(routing_keywords) < 2:
        raise ValueError(f"{concept_id}: fewer than two routing_keywords")
    expected_yes_ratio = _bounded_float(
        raw.get("expected_yes_ratio"), field="expected_yes_ratio", concept_id=concept_id
    )
    raw_reward = raw.get("entropy_reward")
    try:
        entropy_reward = float(raw_reward)
    except (TypeError, ValueError):
        entropy_reward = _entropy_reward(expected_yes_ratio)
    return {
        "concept_id": concept_id,
        "uncertainty_state": uncertainty_state,
        "candidate_question": candidate_question,
        "yes_partition_hint": yes_partition_hint,
        "no_partition_hint": no_partition_hint,
        "expected_yes_ratio": expected_yes_ratio,
        "entropy_reward": max(0.0, min(1.0, entropy_reward)),
        "information_gain_target": information_gain_target,
        "simulation_tree_role": role,
        "reward_propagation_notes": _clean_text(
            raw.get("reward_propagation_notes"), max_len=280
        ),
        "routing_keywords": routing_keywords,
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
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": CACHE_DIR,
        "default_max_concurrency": args.concurrency,
        "retry_attempts": 1,
    })
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[uot_entropy_adapter] concepts={len(names)} model={MODEL}")
    t0 = time.monotonic()
    tasks = [
        _adapt_one(client, sem, name, concepts[name], prompt_template, args)
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
        "port": "uot_entropy",
        "source": "literature/2402.03271.pdf",
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
        print(f"[uot_entropy_adapter] smoke cost=${out['stats']['estimated_cost_usd']:.4f}")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[uot_entropy_adapter] wrote {OUTPUT_PATH}")
    print(f"[uot_entropy_adapter] cost=${out['stats']['estimated_cost_usd']:.4f}")
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
