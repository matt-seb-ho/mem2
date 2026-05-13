# Retrieval Telemetry

## Motivation

Retrieval telemetry asks which concepts are retrieved, how often, by which port, and under which failure or success outcomes. This is the utilization layer for deciding what gathers dust and what drives performance.

## Current Stub

`extract.py` reads retrieval bundles from a case-study run and writes trace-level counts only. It does not yet normalize concept IDs across all bundle shapes.

## Future Work

Future extraction should count per-concept hits, render utilization heatmaps, and join to failure taxonomy labels for method comparison.
