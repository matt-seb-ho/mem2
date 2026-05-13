# Variant Parse Case Studies

## Paper

- Title: PARSE
- Citation: arxiv:2510.08623
- Mechanism: paper ARCHITECT + SCOPE mechanisms - iterative schema refinement based on extraction-error reflection. Per-kind render-flag overrides substitute for per-field JSON schema edits; heuristic performance-based flag flipping replaces ARCHITECT prompt when no LLM provider wired.

## Parity Grade

- Current: partial-with-disclosed-gap
- Source: configs/axes/4.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: variant_parse
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
