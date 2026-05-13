# Episodic Streams Benchmark Placeholder

## Motivation

Episodic streams make memory growth, delayed reuse, and task-order sensitivity first-class instead of incidental.

## Data + Schema

- Data path: `future generated data under data/episodic_streams/`.
- Current footprint: not created yet.
- Schema: Ordered episodes with task id, timestamp, observation, response, feedback, and memory snapshot pointer.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

A stream runner that appends memory after each episode and emits case-study traces plus analysis-ready growth records. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
