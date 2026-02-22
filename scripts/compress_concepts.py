#!/usr/bin/env python3
"""Compress concept memory by deduplicating cues and implementation notes.

Follows arc_memo's memory_compression.ipynb pattern:
  1. Find concepts with redundant cues/implementation (used_in > 1)
  2. Send each to LLM for deduplication/synthesis
  3. Write compressed annotations back to concept memory
  4. Save as compressed JSON

Usage:
    python scripts/compress_concepts.py \
        --input data/competition_math_nt_cp_l5/concept_memory/extracted_v2.json \
        --output data/competition_math_nt_cp_l5/concept_memory/compressed_v1.json \
        --model qwen/qwen3.5-397b-a17b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mem2.concepts.memory import ConceptMemory
from mem2.providers.llmplus_client import LLMPlusProviderClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Compression prompt — adapted from arc_memo's memory_compression.ipynb
# Domain-agnostic: works for ARC, math, code concepts.
# ---------------------------------------------------------------------------
COMPRESSION_PROMPT_TEMPLATE = """\
# Introduction
You are helping maintain a concept memory system that records reusable ideas from \
previously solved problems. Each concept has:
- **cues**: signals that suggest this concept is relevant to a new problem
- **implementation**: notes on how to apply this concept in code

Over multiple problems, these lists accumulate redundant or near-duplicate entries.

# Task
Remove redundancy from the cues and implementation lists below. Rules:
- Keep separate ideas in separate entries
- Remove exact duplicates
- If entries are very similar (only subtly different), synthesize into a single entry
- Preserve the most informative version of each idea
- Do NOT add new entries — only deduplicate and synthesize existing ones

Output a fenced yaml block with the compressed lists:
```yaml
cues:
  - first cue
  - second cue
implementation:
  - first implementation note
  - second implementation note
```

If either list is empty after deduplication, output an empty list: `cues: []`

# Concept to Compress
```
{concept}
```"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress concept memory")
    p.add_argument("--input", type=Path, required=True,
                    help="Path to concept memory JSON (from extract_concepts.py)")
    p.add_argument("--output", type=Path, required=True,
                    help="Path to write compressed concept memory JSON")
    p.add_argument("--model", type=str, default="qwen/qwen3.5-397b-a17b",
                    help="LLM model for compression")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--min-entries", type=int, default=2,
                    help="Only compress concepts with >= this many cues or impl entries")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def needs_compression(concept, min_entries: int = 2) -> bool:
    """Check if a concept has redundant entries worth compressing."""
    return (len(concept.cues) >= min_entries or
            len(concept.implementation) >= min_entries)


async def main() -> None:
    args = parse_args()

    # ── Load concept memory ──────────────────────────────────────────
    mem = ConceptMemory()
    mem.load_from_file(args.input)
    logger.info(f"Loaded {len(mem.concepts)} concepts from {args.input}")

    # ── Find concepts needing compression ─────────────────────────────
    to_compress = []
    for name, concept in mem.concepts.items():
        if needs_compression(concept, args.min_entries):
            to_compress.append(concept)

    logger.info(f"{len(to_compress)} of {len(mem.concepts)} concepts need compression")

    if not to_compress:
        logger.info("Nothing to compress. Copying input to output.")
        mem.save_to_file(args.output)
        return

    # ── Stats before compression ──────────────────────────────────────
    before_cues = [len(c.cues) for c in to_compress]
    before_impl = [len(c.implementation) for c in to_compress]
    logger.info(f"Before compression:")
    logger.info(f"  Cues:   min={min(before_cues)}, max={max(before_cues)}, "
                f"mean={sum(before_cues)/len(before_cues):.1f}")
    logger.info(f"  Impl:   min={min(before_impl)}, max={max(before_impl)}, "
                f"mean={sum(before_impl)/len(before_impl):.1f}")

    # ── Build prompts ─────────────────────────────────────────────────
    prompts = []
    for concept in to_compress:
        concept_str = concept.to_string()
        prompts.append(COMPRESSION_PROMPT_TEMPLATE.format(concept=concept_str))

    logger.info(f"Built {len(prompts)} compression prompts "
                f"(avg {sum(len(p) for p in prompts) // len(prompts)} chars)")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — first prompt preview:")
        print(f"{'='*60}")
        print(prompts[0][:2000])
        print(f"... ({len(prompts[0])} chars total)")
        return

    # ── Run LLM compression ──────────────────────────────────────────
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "default_max_concurrency": args.concurrency,
    })
    gen_cfg = {"n": 1, "temperature": 0.1, "max_tokens": args.max_tokens}

    logger.info(f"Sending {len(prompts)} compression prompts to {args.model}...")
    results = await client.async_batch_generate(
        prompts=prompts, model=args.model, gen_cfg=gen_cfg,
    )

    # ── Parse and apply compression ───────────────────────────────────
    n_ok = 0
    n_fail = 0
    parse_errors: list[str] = []

    for concept, result_list in zip(to_compress, results):
        completion = str(result_list[0]) if result_list else ""
        if not completion.strip():
            n_fail += 1
            parse_errors.append(f"{concept.name}: empty_completion")
            continue

        m = _YAML_BLOCK_RE.search(completion)
        if not m:
            n_fail += 1
            parse_errors.append(f"{concept.name}: no_yaml_block")
            continue

        try:
            parsed = yaml.safe_load(m.group(1))
        except Exception as exc:
            n_fail += 1
            parse_errors.append(f"{concept.name}: yaml_parse_error: {exc}")
            continue

        if not isinstance(parsed, dict):
            n_fail += 1
            parse_errors.append(f"{concept.name}: not_a_dict: {type(parsed).__name__}")
            continue

        # Apply compressed annotations
        new_cues = parsed.get("cues", concept.cues)
        new_impl = parsed.get("implementation", concept.implementation)

        if isinstance(new_cues, list):
            concept.cues = [str(c).strip() for c in new_cues if c]
        if isinstance(new_impl, list):
            concept.implementation = [str(i).strip() for i in new_impl if i]

        n_ok += 1

    logger.info(f"Compression: {n_ok} ok, {n_fail} failed")
    if parse_errors:
        for err in parse_errors:
            logger.warning(f"  {err}")

    # ── Stats after compression ───────────────────────────────────────
    after_cues = [len(c.cues) for c in to_compress]
    after_impl = [len(c.implementation) for c in to_compress]
    logger.info(f"After compression:")
    logger.info(f"  Cues:   min={min(after_cues)}, max={max(after_cues)}, "
                f"mean={sum(after_cues)/len(after_cues):.1f}")
    logger.info(f"  Impl:   min={min(after_impl)}, max={max(after_impl)}, "
                f"mean={sum(after_impl)/len(after_impl):.1f}")

    # Total reduction
    total_before = sum(before_cues) + sum(before_impl)
    total_after = sum(after_cues) + sum(after_impl)
    logger.info(f"Total entries: {total_before} → {total_after} "
                f"({(1 - total_after/total_before)*100:.0f}% reduction)")

    # ── Save compressed memory ────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mem.save_to_file(args.output)
    logger.info(f"Saved compressed memory to {args.output}")

    # ── Summary ──────────────────────────────────────────────────────
    all_sizes = [len(c.to_string()) for c in mem.concepts.values()]
    print(f"\n{'='*60}")
    print("Compression Summary")
    print(f"{'='*60}")
    print(f"Concepts:           {len(mem.concepts)}")
    print(f"Compressed:         {n_ok} (of {len(to_compress)} needing compression)")
    print(f"Failed:             {n_fail}")
    print(f"Concept sizes:      min={min(all_sizes)}, max={max(all_sizes)}, "
          f"mean={sum(all_sizes)//len(all_sizes)}")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
