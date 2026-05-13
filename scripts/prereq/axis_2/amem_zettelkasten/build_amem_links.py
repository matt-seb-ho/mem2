"""Build an A-Mem Zettelkasten link graph over ARC concepts.

Input:
  data/arc_agi/concept_memory/compressed_v1.json

Output:
  data/arc_agi/concept_memory/amem_link_graph_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mem2.concepts.memory import ConceptMemory
from mem2.providers.llmplus_client import LLMPlusProviderClient


SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "amem_link_graph_v1.json"
MODEL = "deepseek/deepseek-v4-flash"
INPUT_COST_PER_M = 0.14
OUTPUT_COST_PER_M = 0.28
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
LINK_TYPES = {
    "generalizes",
    "specializes",
    "prerequisite_of",
    "contrast_with",
    "applied_with",
    "related_to",
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _concept_digest(name: str, raw: dict[str, Any], *, limit: int = 220) -> str:
    parts = [name, str(raw.get("kind") or "")]
    for key in ("description", "routine_subtype", "output_typing"):
        if raw.get(key):
            parts.append(str(raw[key]))
    cues = [str(c) for c in (raw.get("cues") or [])[:3]]
    impl = [str(c) for c in (raw.get("implementation") or [])[:3]]
    if cues:
        parts.append("cues: " + "; ".join(cues))
    if impl:
        parts.append("impl: " + "; ".join(impl))
    return " | ".join(parts)[:limit]


def _candidate_names(
    source: str,
    concepts: dict[str, dict[str, Any]],
    *,
    max_candidates: int,
) -> list[str]:
    raw = concepts[source]
    source_text = _concept_digest(source, raw, limit=1000)
    source_toks = _tokens(source_text)
    source_used = set(raw.get("used_in") or [])
    scored: list[tuple[float, str]] = []
    for name, cand in concepts.items():
        if name == source:
            continue
        cand_toks = _tokens(_concept_digest(name, cand, limit=1000))
        overlap = len(source_toks & cand_toks) / max(len(source_toks | cand_toks), 1)
        shared_used = len(source_used & set(cand.get("used_in") or []))
        kind_bonus = 0.05 if raw.get("kind") == cand.get("kind") else 0.0
        score = overlap + 0.08 * shared_used + kind_bonus
        if score > 0:
            scored.append((score, name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:max_candidates]]


def _build_prompt(
    source: str,
    concepts: dict[str, dict[str, Any]],
    candidates: list[str],
    *,
    max_links: int,
) -> str:
    candidate_lines = "\n".join(
        f"- {name}: {_concept_digest(name, concepts[name])}"
        for name in candidates
    )
    return f"""You build A-Mem Zettelkasten links between ARC concept notes.

Return lines only. No prose, no JSON, no markdown fences.

Line format:
target_concept || link_type || confidence || rationale

Rules:
- Select 3 to {max_links} targets from ALLOWED TARGETS only.
- link_type must be one of: {", ".join(sorted(LINK_TYPES))}
- confidence is a number from 0.0 to 1.0.
- rationale should be one short retrieval-useful phrase.
- Prefer conceptual links a solver should remember, not task IDs.

# SOURCE NOTE
{source}: {_concept_digest(source, concepts[source], limit=700)}

# ALLOWED TARGETS
{candidate_lines}
"""


def _parse_links(raw: str, source: str, allowed: set[str], max_links: int) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("||")]
        if len(fields) != 4:
            continue
        target, link_type, confidence, rationale = fields
        if target not in allowed or target == source:
            continue
        if link_type not in LINK_TYPES:
            link_type = "related_to"
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            conf = 1.0
        key = (target, link_type)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source_concept": source,
            "target_concept": target,
            "link_type": link_type,
            "rationale": rationale[:240],
            "confidence": conf,
        })
        if len(out) >= max_links:
            break
    return out


def _fallback_links(
    source: str,
    concepts: dict[str, dict[str, Any]],
    candidates: list[str],
    *,
    max_links: int,
) -> list[dict[str, Any]]:
    source_kind = concepts[source].get("kind")
    out: list[dict[str, Any]] = []
    for target in candidates[:max_links]:
        target_kind = concepts[target].get("kind")
        link_type = "applied_with" if source_kind == target_kind else "related_to"
        out.append({
            "source_concept": source,
            "target_concept": target,
            "link_type": link_type,
            "rationale": "deterministic lexical and co-use fallback",
            "confidence": 0.5,
        })
    return out


def _estimate_cost(snapshot: dict[str, dict[str, Any]]) -> float:
    usage = snapshot.get(MODEL, {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return (input_tokens / 1_000_000.0) * INPUT_COST_PER_M + (
        output_tokens / 1_000_000.0
    ) * OUTPUT_COST_PER_M


async def main_async(args: argparse.Namespace) -> int:
    if OUT_FILE.exists() and not args.force:
        print(f"ERROR: output already exists: {OUT_FILE}", file=sys.stderr)
        return 2
    seed_payload = json.loads(SEED_MEM.read_text())
    mem = ConceptMemory.from_payload(seed_payload)
    concepts: dict[str, dict[str, Any]] = seed_payload.get("concepts", {})
    names = sorted(mem.concepts.keys())
    if args.limit_concepts:
        names = names[: args.limit_concepts]
    candidates_by_source = {
        name: _candidate_names(name, concepts, max_candidates=args.max_candidates)
        for name in names
    }
    prompts = [
        _build_prompt(
            name,
            concepts,
            candidates_by_source[name],
            max_links=args.max_links_per_concept,
        )
        for name in names
    ]
    print(f"[amem_links] concepts={len(names)} model={MODEL}")
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "dotenv_path": str(ROOT / ".env"),
        "cache_dir": "/private/tmp/mem2_phase3b_llm_cache/amem_links",
        "default_max_concurrency": args.concurrency,
    })
    t0 = time.monotonic()
    results = await client.async_batch_generate(
        prompts,
        MODEL,
        {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "batch_size": args.concurrency,
            "ignore_cache": args.ignore_cache,
            "extra_kwargs": {
                "extra_body": {"reasoning": {"effort": "none", "exclude": True}},
            },
        },
        request_timeout=args.timeout,
    )
    links: list[dict[str, Any]] = []
    failures: list[str] = []
    for source, completions in zip(names, results, strict=True):
        allowed = set(candidates_by_source[source])
        raw = completions[0] if completions else ""
        parsed = _parse_links(raw or "", source, allowed, args.max_links_per_concept)
        if not parsed:
            failures.append(source)
            parsed = _fallback_links(
                source,
                concepts,
                candidates_by_source[source],
                max_links=args.max_links_per_concept,
            )
        links.extend(parsed)
    usage = client.get_usage_snapshot()
    cost = _estimate_cost(usage)
    counts: dict[str, int] = {}
    for link in links:
        counts[link["source_concept"]] = counts.get(link["source_concept"], 0) + 1
    out = {
        "schema_version": "1",
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "model": MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "links": links,
        "stats": {
            "num_concepts": len(names),
            "num_links": len(links),
            "links_per_concept_mean": statistics.fmean(counts.values()) if counts else 0.0,
            "llm_calls": len(names),
            "num_failures": len(failures),
            "failures": failures[:20],
            "estimated_cost_usd": cost,
            "wall_time_s": time.monotonic() - t0,
            "token_usage": usage,
        },
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[amem_links] wrote {OUT_FILE}")
    print(
        f"[amem_links] links={len(links)} failures={len(failures)} "
        f"cost=${cost:.4f}"
    )
    if cost > args.max_cost_usd:
        print(
            f"ERROR: estimated cost ${cost:.4f} exceeded limit ${args.max_cost_usd:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0 if links else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--max-links-per-concept", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-cost-usd", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
