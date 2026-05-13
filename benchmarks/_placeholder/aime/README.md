# AIME Benchmark Placeholder

## Motivation

AIME tests whether memory can store reusable olympiad-style tricks and avoid overfitting to ARC visual routines.

## Data + Schema

- Data path: `data/aime_1983_2025/problems.jsonl`.
- Current footprint: 1 file.
- Schema: JSONL math problems with final-answer scoring still to be normalized.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

A math task adapter, exact-answer evaluator, flat baseline, arcmemo-style builder, and at least one retrieval condition run with traces. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
