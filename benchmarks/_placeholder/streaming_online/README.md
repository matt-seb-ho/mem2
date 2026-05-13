# Streaming Online Benchmark Placeholder

## Motivation

Streaming online mode tests memory under wall-clock pressure where retrieval cost and update cost matter.

## Data + Schema

- Data path: `future streaming data under data/streaming_online/`.
- Current footprint: not created yet.
- Schema: Online query stream with per-query time budget, incremental context, and optional delayed feedback.
- Scoring: not wired in the active mem2 pipeline yet; the adapter must define exact, executable, or structured scoring before live runs.

## What Success Looks Like

A time-budgeted runner, telemetry for latency and retrieval depth, and policies that degrade gracefully under load. Expected substrate dependencies include task-specific concept-memory seeds, retrieval bundle metadata compatible with `case_studies/`, and analysis outputs readable by `analysis/` modules.
