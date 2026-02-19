# Competition Math — Number Theory + Counting & Probability, Level 5

**Source:** [qwedsacf/competition_math](https://huggingface.co/datasets/qwedsacf/competition_math) (mirror of hendrycks/competition_math, MATH benchmark)

Subset of the MATH competition dataset filtered to:
- **Types:** Number Theory, Counting & Probability
- **Level:** 5 (hardest)
- **Answer format:** Integer answers only (extracted via `\boxed{}` from solutions)

679 eligible problems out of 12,500 total in the dataset.

## Contents

| File | Description |
|------|-------------|
| `problems.jsonl` | 679 problems in JSONL format (one JSON object per line) |
| `splits.json` | Train/eval split definitions (seed=42) |

## Splits (`splits.json`)

| Split | Count | Purpose |
|-------|-------|---------|
| `build` | 200 | Concept accumulation (memory building) |
| `eval` | 100 | Inference testing (baseline vs concept-augmented) |
| `held_out` | 379 | Reserved for future experiments |

Splits are random (seed=42) and disjoint. The `build` set is used to solve problems and extract concepts; the `eval` set measures whether those concepts improve a weaker solver.

## Problem JSONL Format

```json
{
  "uid": "cmath_1745",
  "dataset_idx": 1745,
  "problem": "Find the remainder when ...",
  "level": "Level 5",
  "type": "Number Theory",
  "solution": "We can write ... The answer is $\\boxed{7}$."
}
```

All original fields from the HuggingFace dataset are preserved, plus `uid` and `dataset_idx` for cross-referencing.
