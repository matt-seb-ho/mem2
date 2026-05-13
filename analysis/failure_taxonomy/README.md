# Failure Taxonomy

## Motivation

Failure taxonomy turns raw incorrect attempts into comparable categories. The initial LN-018 classes are single-cell discriminative, relative-offset, small-mask, color-role, and region-boundary.

## Current Stub

`classify.py` is deliberately a no-LLM placeholder. It reads a case-study run directory, counts trace records, and writes a JSON file that marks classification as pending.

## Future Work

The future classifier should support an explicit LLM-as-judge mode, cite prompt and response files for each label, and render `matrix.md` as a failure-class by method comparison table.
