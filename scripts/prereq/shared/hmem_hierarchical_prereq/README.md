# axis_1 / hmem_hierarchical — prerequisite builder

## What this folder builds

A 3-level concept hierarchy (Domain → Category → Sub-category → concepts)
for the H-MEM hierarchical retriever (axis 1.9).

Without it, layer-by-layer routing collapses to single-level kind grouping
+ token overlap.

## Files

- `build_hierarchy.py` — single LLM call grouping all 270 concepts into a
  3-level tree.

## Inputs / Outputs

- In: `mem2/data/arc_agi/concept_memory/compressed_v1.json`
- Out: `mem2/data/arc_agi/concept_memory/concept_hierarchy_v1.json`

## Cost / runtime

DeepSeek V4 Flash, single call, ~$0.05, ~30-90s wall.

## Usage

```bash
cd mem2 && source .env
.venv/bin/python scripts/prereq/shared/hmem_hierarchical_prereq/build_hierarchy.py
```
