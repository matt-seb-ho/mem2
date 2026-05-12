"""Build a typed concept-relation graph for HippoRAG-PPR (axis 1).

Why this exists
---------------
HippoRAG-PPR (axis B.4) needs a knowledge graph with typed entity-relation
edges. The shipped seed memory `compressed_v1.json` only carries
co-activation overlap (`used_in`), which is too thin: PPR collapses to
"top-K by frequency" — same as the baseline.

This script asks an LLM, per concept, "which OTHER concepts does this
relate to, and how?" and saves the resulting typed edge list as a graph
file that hipporag_ppr (and any future relation-aware retriever) can
consume.

Inputs
------
- mem2/data/arc_agi/concept_memory/compressed_v1.json (seed memory)

Outputs
-------
- mem2/data/arc_agi/concept_memory/concept_graph_v1.json
  Schema:
    {
      "schema_version": "1",
      "model": "deepseek/deepseek-v4-flash",
      "built_at": "<ISO timestamp>",
      "edges": [
        {"src": "<concept_name>", "tgt": "<concept_name>",
         "relation": "uses|is_a|specializes|opposite_of|composed_of",
         "weight": 1.0}
      ],
      "stats": {"num_concepts": 270, "num_edges": <int>,
                "edges_per_concept": {"mean": float, "max": int, "min": int}}
    }

Usage
-----
  cd mem2
  source .env
  .venv/bin/python scripts/prereq/axis_1/hipporag_ppr/build_concept_graph.py \
      [--limit N]   # optional: only process first N concepts (for smoketest)
      [--max-targets-per-concept K]  # cap edges per concept (default 15)

Cost
----
DeepSeek V4 Flash via OpenRouter, $0.14/M input + $0.28/M output.
Estimated per-call: ~6K input + ~600 output tokens → ~$0.001/call.
270 concepts → ~$0.30 total. Wall: ~5-10 min.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[4]  # mem2/
SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_graph_v1.json"

VALID_RELATIONS = {"uses", "is_a", "specializes", "opposite_of", "composed_of"}

SYSTEM_PROMPT = """You are extracting concept relationships for a knowledge graph.

Given a focal concept and a catalog of all other concepts, identify which
others the focal concept has a meaningful semantic relationship with, and
the type of each relationship.

Relationship types (use exactly these strings):
- "uses": the focal concept invokes / depends on / is implemented in terms
  of the target concept
- "is_a": the focal concept is a kind of / subtype of the target concept
- "specializes": the focal concept is a more specific version of a more
  general target (inverse of "is_a" loosely; use when the target is a
  generic and the focal is the specialization)
- "opposite_of": the focal concept is semantically opposite or dual to
  the target (e.g., "rotate clockwise" vs "rotate counterclockwise")
- "composed_of": the focal concept is built from / composed of the
  target concept(s) as parts

Be conservative. Only emit a relation when it is strongly suggested by
the descriptions. Do NOT emit "co-occurs" or "related"-style weak edges.
Skip the focal concept itself. If no clear relationships exist, return
an empty list.

Output format (strict JSON, no prose):
[
  {"target": "<exact target concept name>", "relation": "uses"},
  {"target": "<exact target concept name>", "relation": "is_a"}
]

The "target" must be an EXACT match to a concept name in the catalog."""


def build_user_prompt(focal: dict, catalog_lines: list[str], max_targets: int) -> str:
    """Construct the user prompt for one focal concept."""
    name = focal["name"]
    kind = focal.get("kind", "")
    desc = focal.get("description") or ""
    cues = focal.get("cues") or []
    impl = focal.get("implementation") or []

    cues_block = "\n".join(f"  - {c}" for c in cues[:5]) if cues else "  (none)"
    impl_block = "\n".join(f"  - {i}" for i in impl[:5]) if impl else "  (none)"

    catalog_block = "\n".join(catalog_lines)

    return f"""# FOCAL CONCEPT

name: {name}
kind: {kind}
description: {desc}
cues:
{cues_block}
implementation (truncated):
{impl_block}

# CATALOG (all 270 concepts; pick targets from this list ONLY)

{catalog_block}

# TASK

For the focal concept "{name}", identify up to {max_targets} OTHER
concepts in the catalog that have a clear semantic relationship to it.
Use only relations from the allowed set: uses, is_a, specializes,
opposite_of, composed_of. Be conservative.

Output strict JSON list of {{"target": str, "relation": str}}. No prose.
"""


def make_catalog_lines(concepts: dict[str, dict]) -> list[str]:
    """Compact one-line summary per concept for the catalog block."""
    lines = []
    for name in sorted(concepts.keys()):
        c = concepts[name]
        kind = c.get("kind", "?")
        desc = (c.get("description") or "").replace("\n", " ").strip()[:120]
        lines.append(f"- {name} [{kind}]: {desc}")
    return lines


async def call_llm(client: httpx.AsyncClient, api_key: str, system: str, user: str) -> str:
    """Single OpenRouter call. Returns raw assistant text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    resp = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def parse_edges(raw: str, focal_name: str, valid_targets: set[str], max_targets: int) -> list[dict]:
    """Parse the LLM's JSON output, validate and normalize."""
    raw = raw.strip()
    # Strip code-fence wrappers if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to find a JSON array substring
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            items = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(items, list):
        return []

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        tgt = it.get("target")
        rel = it.get("relation")
        if not isinstance(tgt, str) or not isinstance(rel, str):
            continue
        if rel not in VALID_RELATIONS:
            continue
        if tgt == focal_name:
            continue
        if tgt not in valid_targets:
            continue
        key = (tgt, rel)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": focal_name, "tgt": tgt, "relation": rel, "weight": 1.0})
        if len(edges) >= max_targets:
            break
    return edges


async def process_concept(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    api_key: str,
    focal: dict,
    catalog_lines: list[str],
    valid_targets: set[str],
    max_targets: int,
) -> tuple[str, list[dict], dict]:
    """Process one concept: build prompt, call LLM, parse edges."""
    async with sem:
        user = build_user_prompt(focal, catalog_lines, max_targets)
        t0 = time.monotonic()
        try:
            raw = await call_llm(client, api_key, SYSTEM_PROMPT, user)
            edges = parse_edges(raw, focal["name"], valid_targets, max_targets)
            elapsed = time.monotonic() - t0
            return focal["name"], edges, {"ok": True, "elapsed_s": elapsed, "raw_len": len(raw)}
        except Exception as e:
            elapsed = time.monotonic() - t0
            return focal["name"], [], {"ok": False, "elapsed_s": elapsed, "error": str(e)[:200]}


async def main_async(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment.", file=sys.stderr)
        return 2

    if not SEED_MEM.exists():
        print(f"ERROR: seed memory not found at {SEED_MEM}", file=sys.stderr)
        return 2

    seed = json.loads(SEED_MEM.read_text())
    concepts = seed.get("concepts", {})
    if not concepts:
        print("ERROR: no concepts in seed memory", file=sys.stderr)
        return 2

    print(f"[build_graph] loaded {len(concepts)} concepts from {SEED_MEM.name}")

    catalog_lines = make_catalog_lines(concepts)
    valid_targets = set(concepts.keys())

    items = [concepts[n] for n in sorted(concepts.keys())]
    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]
        print(f"[build_graph] LIMIT mode: processing first {len(items)} concepts only")

    sem = asyncio.Semaphore(args.concurrency)
    all_edges: list[dict] = []
    failures: list[str] = []
    timings: list[float] = []

    async with httpx.AsyncClient() as client:
        tasks = [
            process_concept(
                sem, client, api_key, c, catalog_lines, valid_targets, args.max_targets_per_concept
            )
            for c in items
        ]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            name, edges, meta = await coro
            completed += 1
            timings.append(meta.get("elapsed_s", 0.0))
            if not meta.get("ok"):
                failures.append(name)
                print(f"  [{completed}/{len(items)}] {name}: FAIL ({meta.get('error', '?')[:80]})")
            else:
                all_edges.extend(edges)
                print(f"  [{completed}/{len(items)}] {name}: {len(edges)} edges ({meta['elapsed_s']:.1f}s)")

    edges_per = {}
    for e in all_edges:
        edges_per.setdefault(e["src"], 0)
        edges_per[e["src"]] += 1

    stats = {
        "num_concepts": len(items),
        "num_edges": len(all_edges),
        "num_concepts_with_zero_edges": sum(1 for n in (c["name"] for c in items) if edges_per.get(n, 0) == 0),
        "edges_per_concept_mean": statistics.mean(edges_per.values()) if edges_per else 0.0,
        "edges_per_concept_median": statistics.median(edges_per.values()) if edges_per else 0.0,
        "edges_per_concept_max": max(edges_per.values()) if edges_per else 0,
        "edges_per_concept_min": min(edges_per.values()) if edges_per else 0,
        "wall_time_total_s": sum(timings),
        "wall_time_per_call_mean_s": statistics.mean(timings) if timings else 0.0,
        "num_failures": len(failures),
        "failures": failures[:20],
    }
    relation_counts = {r: 0 for r in VALID_RELATIONS}
    for e in all_edges:
        relation_counts[e["relation"]] += 1
    stats["relation_counts"] = relation_counts

    out = {
        "schema_version": "1",
        "model": "deepseek/deepseek-v4-flash",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "edges": all_edges,
        "stats": stats,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\n[build_graph] wrote {OUT_FILE.name}")
    print(f"[build_graph] {len(all_edges)} edges across {len(items)} concepts")
    print(f"[build_graph] mean edges/concept: {stats['edges_per_concept_mean']:.1f}")
    print(f"[build_graph] relation breakdown: {relation_counts}")
    if failures:
        print(f"[build_graph] WARN: {len(failures)} concepts failed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build typed concept-relation graph for HippoRAG-PPR")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N concepts (smoketest)")
    ap.add_argument("--max-targets-per-concept", type=int, default=15, help="Cap edges per concept")
    ap.add_argument("--concurrency", type=int, default=8, help="Concurrent LLM calls")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
