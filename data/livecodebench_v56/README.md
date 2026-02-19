# LiveCodeBench v5 + v6

**Source:** [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) (Jain et al., 2024)

Combined problems from release_v5 and release_v6 of LiveCodeBench, a benchmark of competitive programming problems sourced from recent contests (Codeforces, AtCoder, LeetCode).

## Contents

| File | Description |
|------|-------------|
| `problems.jsonl` | 342 problems in JSONL format (v6 first, then v5) |
| `splits.json` | Train/eval split definitions (seed=42) |

## Problem Counts

| Version | Easy | Medium | Hard | Total | Date Range |
|---------|------|--------|------|-------|------------|
| v6 | 43 | 52 | 80 | 175 | Late 2024+ |
| v5 | 41 | 52 | 74 | 167 | Oct 2024+ |
| **Combined** | **84** | **104** | **154** | **342** | |

No overlapping problem IDs between versions.

## Splits (`splits.json`)

| Split | Count | Purpose |
|-------|-------|---------|
| `build` | 200 | Concept accumulation (memory building) |
| `eval` | 100 | Inference testing (baseline vs concept-augmented) |
| `held_out` | 42 | Reserved for future experiments |

Splits are random (seed=42) and disjoint.

## Problem JSONL Format

```json
{
  "question_id": "3336_B",
  "question_title": "Coin Games",
  "question_content": "There are n coins ...",
  "platform": "codeforces",
  "contest_id": "3336",
  "contest_date": "2025-01-18T18:35:00",
  "difficulty": "easy",
  "starter_code": "",
  "public_test_cases": "[{\"input\": \"...\", \"output\": \"...\"}]",
  "metadata": "{}",
  "_version": "v6"
}
```

All original fields from the HuggingFace dataset are preserved except `private_test_cases` (base64+zlib+pickle encoded, 660 MB uncompressed, excluded for storage). Evaluation uses public test cases only.
