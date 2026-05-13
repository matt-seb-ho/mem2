# Competition Math Benchmark Placeholder

## Motivation

Competition math tests reusable theorem, transformation, and strategy memory outside ARC grid programs.

## Data + Schema

- Data path: `data/competition_math_all_l5/ and data/competition_math_nt_cp_l5/`.
- Current footprint: 102 files across the two math roots.
- Schema: JSONL problems, splits, and existing concept-memory artifacts for level-5 math subsets.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

A unified math benchmark adapter that can run both all_l5 and nt_cp_l5 subsets with exact or judge-free answer scoring. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
