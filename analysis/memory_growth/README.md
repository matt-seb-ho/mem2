# Memory Growth

## Motivation

Memory growth tracks bank size, replacement events, and consolidation pressure over an episodic run. It is the analysis layer for lifelong-learning questions such as when memory helps, saturates, or forgets.

## Current Stub

`extract.py` reads a case-study run directory and writes a placeholder JSON payload with trace count and run metadata. It does not infer real memory size yet.

## Future Work

Future extraction should read memory snapshots, per-episode updates, and provenance hooks, then render growth curves and forgetting event timelines.
