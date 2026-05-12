# axis_1 / magma_multigraph — prerequisite builder

## What this folder builds

Four orthogonal graph views (semantic, temporal, causal, entity) for the
MAGMA multi-graph retriever (axis 1.11).

This script DERIVES the views from already-available data — no extra LLM
calls. The `temporal` and `causal` views are approximations of the
paper's intent (we don't have true authorship-time history or
dependency traces); the `semantic` and `entity` views are direct.

## Files

- `derive_views.py` — splits the hipporag-built graph + seed memory into
  4 views.

## Inputs / Outputs

- In: `concept_memory/compressed_v1.json` + `concept_graph_v1.json`
- Out: `concept_memory/concept_views_v1.json`

## Cost / runtime

$0, <1s. Pure Python derivation.

## Prerequisite

Must run AFTER the hipporag_ppr graph build completes (depends on
`concept_graph_v1.json`).

## Usage

```bash
cd mem2
.venv/bin/python scripts/prereq/axis_1/magma_multigraph/derive_views.py
```
