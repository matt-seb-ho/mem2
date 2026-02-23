#!/usr/bin/env python3
"""Compute per-concept selection frequencies from selected_concepts.json.

Input:  selected_concepts.json  (from scripts/select_concepts.py)
        Format: {problem_id: [concept_name, ...], ...}

Output: concept_frequencies.json
        Format: {concept_name: fraction, ...}
        where fraction = (# problems selecting this concept) / (# total problems)

Usage:
    python scripts/compute_concept_frequencies.py \
        --input data/.../selection_v1/selected_concepts.json \
        --output data/.../selection_v1/concept_frequencies.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute concept selection frequencies")
    p.add_argument("--input", type=Path, required=True,
                    help="Path to selected_concepts.json")
    p.add_argument("--output", type=Path, required=True,
                    help="Output path for concept_frequencies.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    selected = json.loads(args.input.read_text())
    total_problems = len(selected)
    if total_problems == 0:
        print("No problems found in input file.")
        return

    counter: Counter[str] = Counter()
    for concepts in selected.values():
        for name in concepts:
            counter[name] += 1

    frequencies = {
        name: count / total_problems
        for name, count in sorted(counter.items(), key=lambda x: -x[1])
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frequencies, indent=2) + "\n")

    print(f"Computed frequencies for {len(frequencies)} concepts "
          f"across {total_problems} problems")
    print(f"Output: {args.output}")

    # Summary stats
    if frequencies:
        vals = list(frequencies.values())
        print(f"Frequency range: {min(vals):.3f} - {max(vals):.3f}")
        print(f"Mean: {sum(vals)/len(vals):.3f}")
        high = [n for n, f in frequencies.items() if f > 0.5]
        if high:
            print(f"High-frequency (>0.5): {len(high)} concepts: {high[:10]}")


if __name__ == "__main__":
    main()
