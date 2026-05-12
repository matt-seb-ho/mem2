# axis_1 / hipporag_ppr — prerequisite builder

## What this folder builds

A typed concept-relation graph for the HippoRAG-PPR retriever (axis 1.4).

Without it, `hipporag_ppr` runs PPR on a thin co-activation graph (built
from `used_in` overlap only) and converges to top-K-by-frequency — same
as the baseline.

With it, `hipporag_ppr` traverses real semantic edges (`uses`, `is_a`,
`specializes`, `opposite_of`, `composed_of`) and the comparison vs
baseline becomes meaningful.

## Files

- `build_concept_graph.py` — extraction script. Per-concept LLM call;
  asks "which other concepts does this relate to?" Aggregates into a
  typed edge list.

## Inputs

- `mem2/data/arc_agi/concept_memory/compressed_v1.json` (seed memory)

## Outputs

- `mem2/data/arc_agi/concept_memory/concept_graph_v1.json`
  (typed edge list + metadata stats)

## Cost / runtime

- DeepSeek V4 Flash via OpenRouter, ~$0.30 total for 270 concepts.
- ~5-10 min wall with `--concurrency 8`.

## Usage

```bash
cd mem2
source .env
.venv/bin/python scripts/prereq/axis_1/hipporag_ppr/build_concept_graph.py
```

Smoketest first 3 concepts:

```bash
.venv/bin/python scripts/prereq/axis_1/hipporag_ppr/build_concept_graph.py --limit 3
```

## Once it lands

The `hipporag_ppr` retriever needs a one-time wiring change to load
`concept_graph_v1.json` instead of synthesizing from `used_in`. Tracked
as a follow-up; see doc 52 for the broader axis-1 audit.
