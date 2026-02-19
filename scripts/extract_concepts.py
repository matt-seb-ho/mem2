#!/usr/bin/env python3
"""Extract PS-format concepts from solved math/code problems.

Usage:
    python scripts/extract_concepts.py \
        --run-dir outputs/_runs/build_math/151900440f88 \
        --domain math \
        --model qwen/qwen3-coder-30b-a3b-instruct \
        --output data/math_concepts/extracted_v1.json

    python scripts/extract_concepts.py \
        --run-dir outputs/_runs/build_lcb/5b254edab37a \
        --domain code \
        --model qwen/qwen3-coder-30b-a3b-instruct \
        --output data/lcb_concepts/extracted_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project src is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mem2.concepts.extraction import (
    assemble_concept_memory,
    build_extraction_prompt,
    load_solved_problems,
    parse_extraction_response,
)
from mem2.providers.llmplus_client import LLMPlusProviderClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract concepts from solved problems")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to build run directory (e.g. outputs/_runs/build_math/151900440f88/)",
    )
    p.add_argument(
        "--domain",
        choices=["math", "code"],
        required=True,
        help="Domain: math or code",
    )
    p.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-coder-30b-a3b-instruct",
        help="LLM model for extraction (default: qwen/qwen3-coder-30b-a3b-instruct)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/{domain}_concepts/extracted_v1.json)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for extraction LLM (default: 4096)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent LLM calls (default: 16)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling LLM",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Resolve output path
    output_path = args.output
    if output_path is None:
        domain_dir = "math_concepts" if args.domain == "math" else "lcb_concepts"
        output_path = _REPO_ROOT / "data" / domain_dir / "extracted_v1.json"

    # Step 1: Load solved problems
    logger.info(f"Loading solved problems from {args.run_dir}")
    solved = load_solved_problems(args.run_dir, args.domain)
    logger.info(f"Found {len(solved)} solved problems")

    if not solved:
        logger.error("No solved problems found. Exiting.")
        sys.exit(1)

    # Step 2: Build extraction prompts
    prompts = [build_extraction_prompt(p, args.domain) for p in solved]
    uids = [p.uid for p in solved]

    if args.dry_run:
        for uid, prompt in zip(uids, prompts):
            print(f"\n{'='*60}")
            print(f"Problem: {uid}")
            print(f"{'='*60}")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print(f"... ({len(prompt)} chars total)")
        print(f"\nTotal prompts: {len(prompts)}")
        return

    # Step 3: Initialize LLM client
    logger.info(f"Initializing LLM client (model={args.model}, concurrency={args.concurrency})")
    client = LLMPlusProviderClient(profile_cfg={
        "profile_name": "llmplus_openrouter",
        "default_max_concurrency": args.concurrency,
    })

    # Step 4: Call LLM
    gen_cfg = {
        "max_tokens": args.max_tokens,
        "temperature": 0.3,
        "n": 1,
    }
    logger.info(f"Sending {len(prompts)} prompts to LLM...")
    results = await client.async_batch_generate(
        prompts=prompts,
        model=args.model,
        gen_cfg=gen_cfg,
    )

    # Step 5: Parse responses
    extractions: list[tuple[str, dict]] = []
    parse_failures = 0
    for uid, result_list in zip(uids, results):
        if not result_list or result_list[0] is None:
            logger.warning(f"[{uid}] No response from LLM")
            parse_failures += 1
            continue

        response = result_list[0]
        parsed = parse_extraction_response(response)
        if parsed is None:
            logger.warning(f"[{uid}] Failed to parse extraction response")
            parse_failures += 1
            continue

        extractions.append((uid, parsed))

    logger.info(
        f"Parsed {len(extractions)}/{len(solved)} responses "
        f"({parse_failures} failures)"
    )

    if not extractions:
        logger.error("No successful extractions. Exiting.")
        sys.exit(1)

    # Step 6: Assemble ConceptMemory
    mem = assemble_concept_memory(extractions)

    # Step 7: Save
    output_path = Path(output_path)
    mem.save_to_file(output_path)
    logger.info(f"Saved to {output_path}")

    # Step 8: Print summary
    print(f"\n{'='*60}")
    print(f"Extraction Summary")
    print(f"{'='*60}")
    print(f"Domain:            {args.domain}")
    print(f"Solved problems:   {len(solved)}")
    print(f"Successful parses: {len(extractions)}")
    print(f"Parse failures:    {parse_failures}")
    print(f"Total concepts:    {len(mem.concepts)}")
    print(f"Total solutions:   {len(mem.solutions)}")
    print(f"Concept kinds:     {dict(mem.categories)}")
    print(f"Output:            {output_path}")

    # Print usage stats if available
    try:
        usage = client.get_usage_snapshot()
        if usage:
            print(f"LLM usage:         {usage}")
    except Exception:
        pass

    # Spot-check: print a few concepts
    print(f"\n--- Sample concepts ---")
    for i, (name, concept) in enumerate(mem.concepts.items()):
        if i >= 3:
            break
        print(f"\n{concept.to_string()}")
        print(f"  used_in: {concept.used_in}")


if __name__ == "__main__":
    asyncio.run(main())
