# sweeps / two_round_axis_2 — reorganization sweep with proper timing

## Why this exists

The standard mem2 runner calls `consolidate()` ONCE at `runner.py:714`,
AFTER all 3 scoring passes. For axis-2 reorg builders that do their work
in `consolidate()` (LRLL, MemP, DreamCoder, LILO, A-MEM, EvolveR, Stitch,
SleepGate, MemTree, the two MDL variants, accretive_prune, plus ALMA/ADAS
on axis 5), this means the reorganization NEVER affects the run that's
being scored.

This sweep solves it with a two-round design:

```
Round 1 (warmup):  condition's builder + ps_topk retriever, 3-iter
                   → solve, accumulate per-concept outcomes,
                     consolidate fires, save reorganized memory

Round 2 (eval):    passive arcmemo_ps builder (loads round-1 memory) +
                   chosen retriever, 1-iter
                   → solve, score = official accuracy
```

## Two retrievers per condition

For each axis-2 condition, we run round 2 twice:
- `ps_topk` (baseline) → lower bound ("does this reorg help any retriever?")
- The condition's natural matching retriever → upper bound ("does this
  reorg work with its paper-paired retrieval?")

Pairings (8 of 12 conditions naturally match `ps_topk`, so only 4 need an
alternate retriever):

| Reorg method | Matching retriever |
|---|---|
| accretive_prune, reorg_dreamcoder, reorg_stitch, reorg_lilo, reorg_lrll, reorg_memp, reorg_sleepgate | ps_topk |
| reorg_amem, reorg_evolver | colbert_rerank |
| reorg_memtree | hmem_hierarchical |
| reorg_on_graph_mdl_global_plateau, reorg_on_trace_mdl_accretive_everyk | graph_traversal |

## Cost / runtime

Per condition × seed: round 1 (3 iters) + round 2 (1 iter) = 4 LLM passes
× 50 problems ≈ ~$0.20-0.30 per cell. 12 conditions + 4 extra
upper-bound cells = 16 cells × $0.25 ≈ ~$4 per seed.

## Prerequisites

The 4 matching retrievers need their data files built first:
- `colbert_rerank` → `concept_embeddings_v1.npz` (built by
  `scripts/prereq/axis_1/colbert_rerank/build_concept_embeddings.py`)
- `hmem_hierarchical` → `concept_hierarchy_v1.json` (built by
  `scripts/prereq/axis_1/hmem_hierarchical/build_hierarchy.py`)
- `graph_traversal` → reads from `compressed_v1.json` directly + lineage
  from prior reorg (which round 1 provides!)
- (also requires retriever wiring to LOAD the prereq files — currently
  the retrievers synthesize their data on the fly, will need a follow-up
  patch to switch over.)

## Usage

```bash
cd mem2 && source .env

# All 12 conditions, baseline retriever only (lower-bound first pass)
.venv/bin/python scripts/sweeps/two_round_axis_2/run.py --seeds 42 --limit 50

# All 16 cells (lower + upper bound)
.venv/bin/python scripts/sweeps/two_round_axis_2/run.py \
    --seeds 42 --limit 50 --also-baseline

# Subset of conditions
.venv/bin/python scripts/sweeps/two_round_axis_2/run.py \
    --variants reorg_lrll,reorg_memp --seeds 42 --limit 50
```

## Outputs

```
outputs/two_round_axis_2/
  s42/
    accretive_prune__ps_topk/
      round1/<run-hash>/{config.yaml,memory/final.json,result.json,...}
      round2/<run-hash>/{config.yaml,result.json,...}
      round1_memory_seed.json   # the seed_memory_file fed into round 2
      summary.json              # round1_score, round2_score, deltas
    reorg_lrll__ps_topk/...
    reorg_memtree__hmem_hierarchical/...
    _aggregate.json             # cross-condition summary for this seed
```

## Reading the result

`summary.json` per cell:
- `round1_score`: official accuracy from the warmup pass
- `round2_score`: official accuracy AFTER reorganization → **this is what
  we compare across conditions**
- The reference cell is `arcmemo_ps_no_reorg__ps_topk` (manually run as a
  control — same setup but builder has no consolidate effect)

## Caveats

- 1-iter round 2: clean signal but smaller sample-size. Single-seed at
  n=50 has ~6-7pp single-condition noise floor; pairwise differences
  below 8-10pp should be considered noise.
- Round 1's ps_topk retrieval is held constant across all conditions to
  isolate the reorg effect on the SAME accumulated outcome state.
- For the 4 conditions where round 2 uses a non-baseline retriever, the
  retriever must already be wired to consume prereq data files (see
  Prerequisites section).
