# Continual Learning Benchmark Placeholder

## Motivation

Classic continual-learning evals test whether mem2 ports preserve earlier skills while acquiring later ones.

## Data + Schema

- Data path: `future generated or imported CL suites under data/continual_learning/`.
- Current footprint: not created yet.
- Schema: Task sequence with phase labels, replay rules, and held-out regression checks for forgetting.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

Forward-transfer and backward-forgetting metrics plus provenance records for concepts introduced per task phase. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
