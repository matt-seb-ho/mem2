# ARC-AGI Benchmark

## Motivation

ARC-AGI-1 is the active benchmark for mem2 because it stresses reusable transformation concepts, brittle prompt grounding, and retrieval quality under small visual programs. It is the current end-to-end testbed for memory-builder and retriever ports.

## Data + Schema

- Data path: `data/arc_agi/`.
- File count observed on 2026-05-13: 985 files.
- Splits: `training/` and `evaluation/` JSON tasks, plus concept-memory substrates under `data/arc_agi/concept_memory/`.
- Schema: each task has train and test grid pairs; scoring checks generated Python `transform` output against expected grids.

## What Success Looks Like

ARC Phase G means every current memory-builder and retriever axis can run with durable traces in `case_studies/runs/`, benchmark summaries in `outputs/`, and analysis modules able to compute failure type, retrieval usage, memory growth, and provenance for each case-study run.
