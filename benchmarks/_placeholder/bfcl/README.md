# BFCL v4 Benchmark Placeholder

## Motivation

BFCL tests memory for tool schemas, parameter conventions, and multi-turn function-call repair.

## Data + Schema

- Data path: `data/bfcl_v4/`.
- Current footprint: 49 files.
- Schema: JSON function-call tasks, possible answers, and multi-turn function documentation.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

Function-call task adapter, structured-call evaluator, and retrieval traces showing schema and repair memory usage. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
