# Phase-1 Ablation Scaffold — Setup Complete

**Status:** scaffolding complete, live runs not yet executed
**Date:** 2026-04-19
**Source of truth:** `../../raw_ideas/context/surveys/03_mem2/06_ablation_plan.md`

This report documents the code infrastructure landed for Phase-1 axis
validation (A-F). Results reports (`phase1_axis_<A-F>_report.md`) will be
written after live runs on Qwen.

---

## Net-new code

| # | Path | Purpose | Axis |
|---|------|---------|------|
| 1 | `src/mem2/concepts/graph.py` | `ConceptGraph` — co-activation / embedding-sim / authorship-lineage edges | A, B, C |
| 2 | `src/mem2/scoring/mdl.py` | MDL/parsimony scorer (rendered-char length as token proxy) | A |
| 3 | `src/mem2/branches/feedback_engine/plateau_trigger.py` | Rolling-window plateau detection + every-k variant | A |
| 4 | `src/mem2/branches/memory_builder/arcmemo_reorg.py` | Reorg builder with all 4 sub-axes (A1-A4) wired | A |
| 5 | `src/mem2/branches/memory_retriever/graph_traversal.py` | BFS hierarchical retrieval over ConceptGraph | B |
| 6 | `src/mem2/branches/memory_retriever/rrmc_interactive.py` | Multi-round coverage-gated probing (simplified RRMC port) | C |
| 7 | `src/mem2/branches/memory_builder/variant_formats.py` | 5 format variants (minimal / typed_only / cue_heavy / free_text / structured_routine) | D |
| 8 | `src/mem2/branches/memory_builder/barc_ingest.py` | BARC seed ingestion (163 seeds → 146 unique concepts) | E |
| 9 | `src/mem2/branches/memory_builder/alma_style_metaedit.py` | ALMA-style LLM-proposed meta-edit with MDL gate | F |
| 10 | `src/mem2/branches/task_adapter/arc3.py` | ARC-3 task adapter + benchmark **stub** (SDK not wired) | primary eval |
| 11 | `scripts/sweeps/ablation_matrix.py` | Phase-1 sweep driver (all axes) | infra |

All 11 pieces registered in the corresponding `registry/*.py` and verified
to resolve + run end-to-end against the mock provider.

## Strict-parity status

`scripts/parity/run_arc_default_parity_lock.py` → `offline parity reproducible: True` after every addition. New builders/retrievers are behind registry keys that the strict config does not reference, so `run.strict_arcmemo_compat=true` runs are unaffected.

## Axis conditions shipped

| Axis | Conditions |
|------|------------|
| A    | `reorg_off`, `reorg_on_graph_mdl_global_plateau`, `reorg_on_trace_mdl_accretive_everyk` |
| B    | `flat_topk` (ps_selector, top_k=10), `graph_traversal` (bfs_depth=3, prefer_aggregates=True) |
| C    | `one_shot` (ps_selector), `rrmc_multi_round` (max_rounds=5, per_round_k=3, patience=2) |
| D    | `arcmemo_oe`, `arcmemo_ps`, `variant_{minimal,typed_only,cue_heavy,free_text,structured_routine}` (7 total) |
| E    | `empty_start`, `barc_seeded` (BARC dir: `../arc_memo/data/dataset/src/BARC/seeds`) |
| F    | `hand_coded_reorg`, `alma_style_metaedit` (LLM-proposed edits behind a `_meta_edit_provider` hook) |

All conditions dry-run and live-run (mock provider) cleanly.

## Known deviations from the plan

1. **Axis D format variants.** The plan lists `variant_*.py (5 new)` as five
   files. Shipped as **one parameterized file** (`variant_formats.py`) with
   five registry aliases via `VARIANTS` and a `variant=...` constructor arg.
   Same behavior, 1 file vs 5. Rationale: render-flag selection is the only
   differentiator across variants; five near-identical subclasses would be
   pure ceremony. The five variant names are preserved exactly.

2. **Axis C RRMC port is a faithful simplification.** The full RRMC codebase
   (`workstation_00_RRMC/RRMC`, ~3k lines across `rrmc/methods/*`) includes
   knowledge-graph priors, MI estimator, structured probing, and multiple
   stopping-rule families. The shipped retriever captures the core axis-C
   signal — multi-round iteration with a Coverage gate — without the full
   machinery. If Phase-1 shows rrmc_multi_round >= one_shot, a larger port
   is warranted; if null, we save the work.

3. **LLM-proposed reorg (axis F) is LLM-hookable but not LLM-driven.**
   `alma_style_metaedit.py` reads a provider from `ctx.config["_meta_edit_provider"]`
   if present; otherwise falls back to the hand-coded reorg op so the run
   completes. Enables the F-axis toggle without coupling Phase-1 infra to
   an LLM wiring change in `orchestrator/runner.py`.

## Blockers / open items

- **ARC-3 SDK not integrated.** No installable package found; open question #1
  in the plan. `branches/task_adapter/arc3.py` ships the TaskSpec + benchmark
  stub but `Arc3SdkBenchmark.load` raises `NotImplementedError`. Phase-1 runs
  default to `--benchmark arc_agi` (ARC-1/2 data, already working) until the
  SDK lands. Pivot path: once SDK is wired, rerun with `--benchmark arc3_sdk`.

- **Qwen endpoint not yet provisioned.** All Phase-1 runs need Qwen-3-4B
  (or Qwen-2.5-7B) inference. `gpu-request.md` will be written to trigger
  the `_runpod` agent. Configs to derive once endpoint URL is known:
  - `configs/experiments/phase1_qwen3_4b_base.yaml` (strict-ARC shape + Qwen provider)
  - per-axis overlays generated by the sweep driver.

- **Ablation runtime estimates** not yet produced — waiting on Qwen endpoint
  for timing smoketest. Per experiment-guardrails, runtime estimation needs
  a smoketest + 1.3-1.5× buffer.

## How to run (once Qwen endpoint is live)

```bash
# Per-axis, 3 seeds, 100 problems
python scripts/sweeps/ablation_matrix.py --axis A --seeds 42,43,44 \
  --limit 100 --benchmark arc_agi \
  --base-config configs/experiments/phase1_qwen3_4b_base.yaml \
  --output-dir outputs/phase1/axis_A
```

Axes are independent; parallelize by launching separate processes per axis.
