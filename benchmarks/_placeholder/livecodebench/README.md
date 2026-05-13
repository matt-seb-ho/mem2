# LiveCodeBench Benchmark Placeholder

## Motivation

LiveCodeBench tests procedural memory for programming patterns, failure repair, and retrieval under executable feedback.

## Data + Schema

- Data path: `data/livecodebench_v56/`.
- Current footprint: 27 files.
- Schema: Problems, splits, and concept-memory artifacts for code generation and execution-based scoring.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

A code task adapter, sandboxed evaluator, and memory ports that retrieve reusable algorithmic and API concepts. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
