#!/usr/bin/env python3
"""Two-stage concept extraction from solved math/code problems.

Follows the ARC pipeline architecture:
  Stage 1: solution code → pseudocode + summary  (all problems, one LLM call each)
  Stage 2: pseudocode → concepts                 (batched, concept repo in prompt)

Stage 2 passes the current concept repository into each prompt so the LLM can
reuse existing concept names.  After each batch the memory grows, so later
batches see a richer repository.

Usage:
    python scripts/extract_concepts.py \
        --run-dir outputs/_runs/build_math/151900440f88 \
        --domain math \
        --model qwen/qwen3-coder-30b-a3b-instruct \
        --output data/competition_math_nt_cp_l5/concept_memory/extracted_v1.json

    python scripts/extract_concepts.py \
        --run-dir outputs/_runs/build_lcb/5b254edab37a \
        --domain code \
        --model qwen/qwen3-coder-30b-a3b-instruct \
        --output data/livecodebench_v56/concept_memory/extracted_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import orjson

# Ensure project src is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mem2.concepts.extraction import (
    build_concept_prompt,
    build_pseudocode_prompt,
    load_solved_problems,
    parse_concept_response,
    parse_pseudocode_response,
    render_concept_repo,
    write_concept,
)
from mem2.concepts.memory import ConceptMemory, ProblemSolution
from mem2.providers.llmplus_client import LLMPlusProviderClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-stage concept extraction from solved problems"
    )
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
        default="qwen/qwen3.5-397b-a17b",
        help="LLM model for extraction (default: qwen/qwen3.5-397b-a17b)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/{dataset}/concept_memory/extracted_v1.json)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for LLM calls (default: 4096)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent LLM calls (default: 16)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Stage 2 batch size — memory grows between batches (default: 10)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling LLM",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Resolve output path — defaults mirror data/{dataset}/concept_memory/ layout
    output_path = args.output
    if output_path is None:
        dataset_dir = (
            "competition_math_nt_cp_l5" if args.domain == "math"
            else "livecodebench_v56"
        )
        output_path = _REPO_ROOT / "data" / dataset_dir / "concept_memory" / "extracted_v1.json"

    # ---------------------------------------------------------------
    # Load solved problems
    # ---------------------------------------------------------------
    logger.info(f"Loading solved problems from {args.run_dir}")
    solved = load_solved_problems(args.run_dir, args.domain)
    logger.info(f"Found {len(solved)} solved problems")

    if not solved:
        logger.error("No solved problems found. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Stage 1: Solution → Pseudocode + Summary
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Solution → Pseudocode + Summary")
    logger.info("=" * 60)

    s1_prompts = [build_pseudocode_prompt(p, args.domain) for p in solved]
    uids = [p.uid for p in solved]

    if args.dry_run:
        print(f"\n{'='*60}")
        print("STAGE 1 PROMPTS (first 2)")
        print(f"{'='*60}")
        for uid, prompt in list(zip(uids, s1_prompts))[:2]:
            print(f"\n--- {uid} ---")
            print(prompt[:1500])
            if len(prompt) > 1500:
                print(f"... ({len(prompt)} chars total)")
    else:
        client = LLMPlusProviderClient(profile_cfg={
            "profile_name": "llmplus_openrouter",
            "default_max_concurrency": args.concurrency,
        })
        gen_cfg = {"max_tokens": args.max_tokens, "temperature": 0.3, "n": 1}

        logger.info(f"Sending {len(s1_prompts)} Stage 1 prompts...")
        s1_results = await client.async_batch_generate(
            prompts=s1_prompts, model=args.model, gen_cfg=gen_cfg,
        )

    # Parse Stage 1 results
    # pseudocode_map: uid → pseudocode string
    # summary_map: uid → summary string
    pseudocode_map: dict[str, str] = {}
    summary_map: dict[str, str] = {}
    s1_failures = 0

    if args.dry_run:
        # In dry-run, use solution code as pseudocode stand-in
        for sp in solved:
            pseudocode_map[sp.uid] = f"(pseudocode for {sp.uid})"
            summary_map[sp.uid] = f"(summary for {sp.uid})"
    else:
        for uid, result_list in zip(uids, s1_results):
            if not result_list or result_list[0] is None:
                logger.warning(f"[{uid}] Stage 1: no response")
                s1_failures += 1
                continue
            pseudocode, summary = parse_pseudocode_response(result_list[0])
            if not pseudocode:
                logger.warning(f"[{uid}] Stage 1: empty pseudocode")
                s1_failures += 1
                continue
            pseudocode_map[uid] = pseudocode
            summary_map[uid] = summary

    logger.info(
        f"Stage 1: {len(pseudocode_map)}/{len(solved)} succeeded "
        f"({s1_failures} failures)"
    )

    if not pseudocode_map:
        logger.error("No Stage 1 results. Exiting.")
        sys.exit(1)

    # Save Stage 1 output as initial_analysis.json (matches ARC pipeline)
    analysis = {
        uid: {
            "pseudocode": pseudocode_map[uid],
            "summary": summary_map.get(uid, ""),
        }
        for uid in sorted(pseudocode_map.keys())
    }
    analysis_path = output_path.parent / "initial_analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_bytes(orjson.dumps(analysis, option=orjson.OPT_INDENT_2))
    logger.info(f"Stage 1 output saved to {analysis_path}")

    # ---------------------------------------------------------------
    # Stage 2: Pseudocode → Concepts  (batched, with concept repo)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Pseudocode → Concepts (batched)")
    logger.info("=" * 60)

    mem = ConceptMemory()
    s2_uids = sorted(pseudocode_map.keys())
    batch_size = args.batch_size
    s2_failures = 0
    total_concepts_added = 0

    for batch_start in range(0, len(s2_uids), batch_size):
        batch_uids = s2_uids[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(s2_uids) + batch_size - 1) // batch_size
        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"{len(batch_uids)} problems, "
            f"{len(mem.concepts)} concepts in repo"
        )

        # Build Stage 2 prompts for this batch (concept repo is current snapshot)
        s2_prompts = [
            build_concept_prompt(pseudocode_map[uid], args.domain, mem)
            for uid in batch_uids
        ]

        if args.dry_run:
            if batch_num <= 2:
                print(f"\n{'='*60}")
                print(f"STAGE 2 BATCH {batch_num} PROMPTS (first 1)")
                print(f"{'='*60}")
                print(f"\n--- {batch_uids[0]} ---")
                print(s2_prompts[0][:2000])
                if len(s2_prompts[0]) > 2000:
                    print(f"... ({len(s2_prompts[0])} chars total)")
            # Simulate: no concepts added in dry-run
            continue

        # Call LLM
        s2_results = await client.async_batch_generate(
            prompts=s2_prompts, model=args.model, gen_cfg=gen_cfg,
        )

        # Parse and write into memory
        for uid, result_list in zip(batch_uids, s2_results):
            if not result_list or result_list[0] is None:
                logger.warning(f"[{uid}] Stage 2: no response")
                s2_failures += 1
                continue

            concept_list = parse_concept_response(result_list[0])
            if not concept_list:
                logger.warning(f"[{uid}] Stage 2: no concepts parsed")
                s2_failures += 1
                continue

            # Record solution
            mem.solutions[uid] = ProblemSolution(
                problem_id=uid,
                solution=None,
                summary=summary_map.get(uid),
                pseudocode=pseudocode_map.get(uid),
            )

            # Write concepts into memory (grows repo for next batch)
            before = len(mem.concepts)
            for ann in concept_list:
                write_concept(mem, uid, ann)
            added = len(mem.concepts) - before
            total_concepts_added += added

    if args.dry_run:
        print(f"\nTotal Stage 1 prompts: {len(s1_prompts)}")
        print(f"Total Stage 2 batches: {(len(s2_uids) + batch_size - 1) // batch_size}")
        print(f"Batch size: {batch_size}")
        return

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    output_path = Path(output_path)
    mem.save_to_file(output_path)
    logger.info(f"Saved to {output_path}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Extraction Summary")
    print(f"{'='*60}")
    print(f"Domain:                {args.domain}")
    print(f"Solved problems:       {len(solved)}")
    print(f"Stage 1 succeeded:     {len(pseudocode_map)}")
    print(f"Stage 1 failures:      {s1_failures}")
    print(f"Stage 2 failures:      {s2_failures}")
    print(f"Solutions recorded:    {len(mem.solutions)}")
    print(f"Total concepts:        {len(mem.concepts)}")
    print(f"Concept kinds:         {dict(mem.categories)}")
    print(f"Batch size:            {batch_size}")
    print(f"Output:                {output_path}")

    try:
        usage = client.get_usage_snapshot()
        if usage:
            print(f"LLM usage:             {usage}")
    except Exception:
        pass

    # Spot-check: print concept repo
    print(f"\n--- Concept Repository ({len(mem.concepts)} concepts) ---")
    repo_text = render_concept_repo(mem)
    # Print first 2000 chars
    if len(repo_text) > 2000:
        print(repo_text[:2000])
        print(f"... ({len(repo_text)} chars total)")
    else:
        print(repo_text)


if __name__ == "__main__":
    asyncio.run(main())
