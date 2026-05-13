# Omni-Math Benchmark Placeholder

## Motivation

Omni-Math broadens math coverage and tests whether concept banks transfer beyond curated competition subsets.

## Data + Schema

- Data path: `data/omni_math/problems.jsonl`.
- Current footprint: 6 files.
- Schema: JSONL math problems plus retrieval audit and concept-memory artifacts.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

Math adapter reuse, an Omni-Math split config, and comparable retrieval telemetry against competition math. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
