"""Build a multi-level concept hierarchy for H-MEM hierarchical retrieval (axis 1).

Why this exists
---------------
H-MEM (axis 1.9) needs a multi-level memory hierarchy: Domain → Category
→ Trace → Episode. With our flat 270-concept ArcMemoPS memory grouped
only by `kind`, layer routing collapses to single-level grouping +
token overlap — H-MEM's distinctive layer-by-layer descent never happens.

This script asks the LLM, in a SINGLE call, to organize all 270 concepts
into a 3-level hierarchy:
    Domain (root, single)
      └── Category (4-7 broad themes)
            └── Sub-category (2-5 per category)
                  └── concepts (the existing 270)

The output is consumed by `hmem_hierarchical` to do real layer routing.

Inputs
------
- mem2/data/arc_agi/concept_memory/compressed_v1.json (seed memory)

Outputs
-------
- mem2/data/arc_agi/concept_memory/concept_hierarchy_v1.json
  Schema:
    {
      "schema_version": "1",
      "model": "...",
      "built_at": "...",
      "domain": "ARC-AGI",
      "categories": [
        {
          "name": "<broad theme>",
          "description": "<one line>",
          "subcategories": [
            {
              "name": "<sub-theme>",
              "description": "<one line>",
              "concepts": ["<concept_name>", ...]
            }
          ]
        }
      ],
      "stats": {"num_concepts_assigned": int, "num_unassigned": int,
                "num_categories": int, "num_subcategories": int}
    }

Cost / runtime
--------------
DeepSeek V4 Flash via OpenRouter. Single call with full concept catalog
in context (~5K input tokens, ~5K output tokens) → ~$0.05. Wall: ~30-90s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[4]
SEED_MEM = ROOT / "data" / "arc_agi" / "concept_memory" / "compressed_v1.json"
OUT_FILE = ROOT / "data" / "arc_agi" / "concept_memory" / "concept_hierarchy_v1.json"

SYSTEM_PROMPT = """You are organizing a flat list of 270 ARC-AGI reasoning
concepts into a 3-level hierarchy for memory routing.

Levels:
  Domain (root, single, "ARC-AGI")
    └── Category (4-7 broad themes; e.g. "spatial transformations",
                  "object detection", "color/pattern manipulation",
                  "counting/arithmetic", "shape composition")
          └── Sub-category (2-5 per category; e.g. under "spatial
                            transformations": "rotation", "reflection",
                            "translation", "scaling")
                └── Concepts (the existing 270, distributed)

Constraints:
- Every concept must be assigned to EXACTLY ONE sub-category. No
  duplicates. No leftovers.
- Category and sub-category names should be 2-4 words, descriptive.
- Each category and sub-category needs a one-line description.
- The hierarchy should be SEMANTIC, not by `kind` field — group concepts
  that solve similar problem patterns or operate on similar objects.

Output format (strict JSON, no prose):
{
  "categories": [
    {
      "name": "spatial transformations",
      "description": "operations that move/orient objects in the grid",
      "subcategories": [
        {
          "name": "rotation",
          "description": "rotating shapes 90/180/270 degrees",
          "concepts": ["rotate_90", "rotate_180", ...]
        }
      ]
    }
  ]
}

Each "concepts" list contains EXACT concept names from the catalog."""


def make_catalog_block(concepts: dict[str, dict]) -> str:
    """Render the full concept catalog as a compact prompt block."""
    lines = []
    for name in sorted(concepts.keys()):
        c = concepts[name]
        kind = c.get("kind", "?")
        desc = (c.get("description") or "").replace("\n", " ").strip()[:100]
        lines.append(f"- {name} [{kind}]: {desc}")
    return "\n".join(lines)


async def call_llm(api_key: str, system: str, user: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 8000,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=240.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


def parse_hierarchy(raw: str, all_names: set[str]) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise
        data = json.loads(raw[start : end + 1])

    cats = data.get("categories", [])
    seen_concepts: set[str] = set()
    cleaned_cats = []
    duplicates: list[tuple[str, str]] = []
    for cat in cats:
        cname = cat.get("name", "").strip()
        cdesc = cat.get("description", "").strip()
        subs = []
        for sub in cat.get("subcategories", []):
            sname = sub.get("name", "").strip()
            sdesc = sub.get("description", "").strip()
            members = []
            for n in sub.get("concepts", []):
                if not isinstance(n, str):
                    continue
                if n not in all_names:
                    continue
                if n in seen_concepts:
                    duplicates.append((n, f"{cname}/{sname}"))
                    continue
                seen_concepts.add(n)
                members.append(n)
            subs.append({"name": sname, "description": sdesc, "concepts": members})
        cleaned_cats.append({"name": cname, "description": cdesc, "subcategories": subs})

    unassigned = sorted(all_names - seen_concepts)
    return {
        "categories": cleaned_cats,
        "unassigned": unassigned,
        "duplicates": duplicates,
    }


async def main_async() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    seed = json.loads(SEED_MEM.read_text())
    concepts = seed.get("concepts", {})
    print(f"[build_hierarchy] {len(concepts)} concepts loaded")

    catalog = make_catalog_block(concepts)
    user = (
        f"# CATALOG ({len(concepts)} concepts)\n\n{catalog}\n\n"
        "# TASK\n\n"
        "Organize all concepts above into a 3-level hierarchy. Output "
        "strict JSON per the system prompt schema. Every concept must be "
        "assigned to exactly one sub-category."
    )

    print(f"[build_hierarchy] catalog block: {len(catalog)} chars")
    t0 = time.monotonic()
    raw = await call_llm(api_key, SYSTEM_PROMPT, user)
    elapsed = time.monotonic() - t0
    print(f"[build_hierarchy] LLM responded in {elapsed:.1f}s, {len(raw)} chars")

    parsed = parse_hierarchy(raw, set(concepts.keys()))
    cats = parsed["categories"]

    n_subs = sum(len(c["subcategories"]) for c in cats)
    n_assigned = sum(
        len(s["concepts"]) for c in cats for s in c["subcategories"]
    )

    out = {
        "schema_version": "1",
        "model": "deepseek/deepseek-v4-flash",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_seed": str(SEED_MEM.relative_to(ROOT)),
        "domain": "ARC-AGI",
        "categories": cats,
        "stats": {
            "num_concepts_total": len(concepts),
            "num_categories": len(cats),
            "num_subcategories": n_subs,
            "num_concepts_assigned": n_assigned,
            "num_unassigned": len(parsed["unassigned"]),
            "unassigned_concepts": parsed["unassigned"][:20],
            "num_duplicates_dropped": len(parsed["duplicates"]),
            "wall_time_s": elapsed,
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[build_hierarchy] wrote {OUT_FILE.name}")
    print(f"[build_hierarchy] {len(cats)} categories / {n_subs} sub-categories")
    print(f"[build_hierarchy] {n_assigned}/{len(concepts)} concepts assigned ({len(parsed['unassigned'])} unassigned)")
    if parsed["unassigned"]:
        print(f"[build_hierarchy] first 5 unassigned: {parsed['unassigned'][:5]}")
    return 0


def main() -> int:
    import asyncio
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
