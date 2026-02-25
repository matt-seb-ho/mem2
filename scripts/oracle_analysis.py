#!/usr/bin/env python3
"""Oracle best-of analysis: what if we could pick the better answer per problem?

Compares baseline and concept runs problem-by-problem to compute:
- Oracle best-of: solved in baseline OR concept → count as solved
- Concept-only wins: solved in concept but not baseline
- Concept-only losses: solved in baseline but not concept
- Both solved / neither solved

Uses eval_records.jsonl from each run. Supports filtering by pass_idx
to isolate pass 1 (clean) from pass 2 (tainted by feedback bug).
"""

import argparse
import json
from pathlib import Path


def load_solved_set(run_dir: Path, max_pass: int | None = None) -> set[str]:
    """Load set of solved problem UIDs from a run directory.

    Args:
        run_dir: Path to run directory containing eval_records.jsonl
        max_pass: If set, only consider attempts up to this pass index.
                  0 = pass 1 only, 1 = pass 1+2, None = all.
    """
    eval_path = run_dir / "eval_records.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"No eval_records.jsonl in {run_dir}")

    solved = set()
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            pass_idx = rec.get("metadata", {}).get("pass_idx", rec.get("attempt_idx", 0))
            if max_pass is not None and pass_idx > max_pass:
                continue
            if rec.get("is_correct", False):
                solved.add(rec["problem_uid"])
    return solved


def oracle_analysis(
    baseline_dir: Path,
    concept_dir: Path,
    max_pass: int | None = None,
) -> dict:
    baseline_solved = load_solved_set(baseline_dir, max_pass)
    concept_solved = load_solved_set(concept_dir, max_pass)

    # Get all problem UIDs from both runs
    all_baseline = set()
    all_concept = set()
    for path, uid_set in [(baseline_dir, all_baseline), (concept_dir, all_concept)]:
        with open(path / "eval_records.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                uid_set.add(rec["problem_uid"])

    all_problems = all_baseline | all_concept

    oracle_solved = baseline_solved | concept_solved
    both_solved = baseline_solved & concept_solved
    baseline_only = baseline_solved - concept_solved
    concept_only = concept_solved - baseline_solved
    neither = all_problems - oracle_solved

    return {
        "total_problems": len(all_problems),
        "baseline_solved": len(baseline_solved),
        "concept_solved": len(concept_solved),
        "oracle_solved": len(oracle_solved),
        "both_solved": len(both_solved),
        "baseline_only": len(baseline_only),
        "concept_only": len(concept_only),
        "neither_solved": len(neither),
        "oracle_gain_over_baseline": len(oracle_solved) - len(baseline_solved),
        "oracle_gain_over_concept": len(oracle_solved) - len(concept_solved),
        "concept_only_problems": sorted(concept_only),
        "baseline_only_problems": sorted(baseline_only),
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle best-of analysis")
    parser.add_argument("--baseline", required=True, help="Path to baseline run dir")
    parser.add_argument("--concept", required=True, help="Path to concept run dir")
    parser.add_argument(
        "--max-pass", type=int, default=None,
        help="Max pass index to consider (0=pass1 only, 1=pass1+2, None=all)",
    )
    args = parser.parse_args()

    result = oracle_analysis(
        Path(args.baseline), Path(args.concept), args.max_pass
    )

    print(f"\n{'='*50}")
    print(f"Oracle Best-Of Analysis")
    print(f"{'='*50}")
    print(f"Baseline: {args.baseline}")
    print(f"Concept:  {args.concept}")
    if args.max_pass is not None:
        print(f"Max pass: {args.max_pass} (pass {args.max_pass + 1})")
    print(f"{'='*50}")
    print(f"Total problems:         {result['total_problems']}")
    print(f"Baseline solved:        {result['baseline_solved']}")
    print(f"Concept solved:         {result['concept_solved']}")
    print(f"Oracle solved:          {result['oracle_solved']}")
    print(f"  Both solved:          {result['both_solved']}")
    print(f"  Baseline only:        {result['baseline_only']}")
    print(f"  Concept only:         {result['concept_only']}")
    print(f"  Neither:              {result['neither_solved']}")
    print(f"{'='*50}")
    print(f"Oracle gain over baseline: +{result['oracle_gain_over_baseline']}")
    print(f"Oracle gain over concept:  +{result['oracle_gain_over_concept']}")
    print()

    if result["concept_only_problems"]:
        print(f"Concept-only wins ({len(result['concept_only_problems'])}):")
        for uid in result["concept_only_problems"]:
            print(f"  {uid}")

    if result["baseline_only_problems"]:
        print(f"\nBaseline-only wins ({len(result['baseline_only_problems'])}):")
        for uid in result["baseline_only_problems"]:
            print(f"  {uid}")


if __name__ == "__main__":
    main()
