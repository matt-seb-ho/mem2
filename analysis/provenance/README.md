# Provenance

## Motivation

Provenance records which task, episode, or external source introduced each concept. Without it, memory growth cannot distinguish useful lifelong learning from uncontrolled accumulation.

## Current Stub

`tracker.py` reads a case-study run and writes a placeholder lineage payload. It does not mutate memory or infer real concept origins yet.

## Future Work

Future provenance should attach source task IDs to memory updates, track supersession and pruning events, and render per-concept lineage reports.
