# GPQA Diamond Benchmark Placeholder

## Motivation

GPQA tests whether memory helps with domain knowledge selection and distractor-resistant reasoning.

## Data + Schema

- Data path: `data/gpqa_diamond/gpqa_diamond.csv`.
- Current footprint: 1 file.
- Schema: CSV science questions with multiple-choice answers; exact scoring adapter not wired yet.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

Multiple-choice adapter, label-normalized evaluator, and retrieval traces that show which concepts influenced answer choice. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
