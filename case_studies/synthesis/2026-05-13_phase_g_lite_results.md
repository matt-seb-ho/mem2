# Phase G-Lite Results - 2026-05-13

## Configuration
- Conditions: 45
- Seeds: 42, 43
- Problems per seed: 50
- Iters: 1
- Cache: disabled
- Max workers: 512
- Model: deepseek/deepseek-v4-flash
- Tracer: enabled
- Started UTC: 2026-05-13T23:24:13+00:00
- Wall time: 101.58 minutes
- Total LLM calls: 4365
- Total spend: unknown

## Per-condition results

| Axis | Condition | Parity grade | n per seed | seed 42 | seed 43 | Mean | Std | LLM calls | Cost | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | graphrag | reduced-but-honest | 50 | 0.260 | 0.340 | 0.300 | 0.057 | 90 | unknown | OK |
| 1 | lightrag | reduced-but-honest | 50 | 0.280 | 0.320 | 0.300 | 0.028 | 93 | unknown | OK |
| 1 | pathrag | surface-port-only-disclosed | 50 | 0.320 | 0.280 | 0.300 | 0.028 | 91 | unknown | OK |
| 1 | flat_topk | baseline | 50 | 0.180 | 0.300 | 0.240 | 0.085 | 87 | unknown | OK |
| 1 | hipporag_ppr | reduced-but-honest | 50 | 0.260 | 0.160 | 0.210 | 0.071 | 86 | unknown | OK |
| 1 | graph_traversal | unknown | 50 | 0.240 | 0.160 | 0.200 | 0.057 | 91 | unknown | OK |
| 1 | colbert_rerank | reduced-but-honest | 50 | 0.200 | 0.160 | 0.180 | 0.028 | 93 | unknown | OK |
| 1 | raptor | faithful | 50 | 0.140 | 0.180 | 0.160 | 0.028 | 89 | unknown | OK |
| 1 | hmem_hierarchical | reduced-but-honest | 50 | 0.120 | 0.180 | 0.150 | 0.042 | 89 | unknown | OK |
| 1 | hipporag2_filter | surface-port-only-disclosed | 50 | 0.120 | 0.080 | 0.100 | 0.028 | 174 | unknown | OK |
| 1 | magma_multigraph | reduced-but-honest | 50 | 0.020 | 0.040 | 0.030 | 0.014 | 140 | unknown | OK |
| 2 | accretive_prune | unknown | 50 | 0.220 | 0.300 | 0.260 | 0.057 | 87 | unknown | OK |
| 2 | reorg_memp | reduced-but-honest | 50 | 0.200 | 0.220 | 0.210 | 0.014 | 90 | unknown | OK |
| 2 | reorg_memtree | partial-with-disclosed-gap | 50 | 0.240 | 0.140 | 0.190 | 0.071 | 99 | unknown | OK |
| 2 | reorg_amem | partial-with-disclosed-gap | 50 | 0.120 | 0.220 | 0.170 | 0.071 | 91 | unknown | OK |
| 2 | reorg_dreamcoder | faithful | 50 | 0.120 | 0.220 | 0.170 | 0.071 | 92 | unknown | OK |
| 2 | reorg_sleepgate | wrong-disclosed | 50 | 0.160 | 0.160 | 0.160 | 0.000 | 98 | unknown | OK |
| 2 | reorg_lilo | partial-with-disclosed-gap | 50 | 0.120 | 0.160 | 0.140 | 0.028 | 94 | unknown | OK |
| 2 | reorg_lrll | reduced-but-honest | 50 | 0.060 | 0.200 | 0.130 | 0.099 | 93 | unknown | OK |
| 2 | reorg_off | baseline | 50 | 0.100 | 0.160 | 0.130 | 0.042 | 97 | unknown | OK |
| 2 | reorg_evolver | reduced-but-honest | 50 | 0.200 | 0.040 | 0.120 | 0.113 | 98 | unknown | OK |
| 2 | reorg_on_graph_mdl_global_plateau | unknown | 50 | 0.140 | 0.100 | 0.120 | 0.028 | 98 | unknown | OK |
| 2 | reorg_on_trace_mdl_accretive_everyk | unknown | 50 | 0.100 | 0.120 | 0.110 | 0.014 | 97 | unknown | OK |
| 2 | reorg_stitch | reduced-but-honest | 50 | 0.080 | 0.080 | 0.080 | 0.000 | 98 | unknown | OK |
| 3 | uot_entropy | reduced-but-honest | 50 | 0.140 | 0.220 | 0.180 | 0.057 | 92 | unknown | OK |
| 3 | rrmc_multi_round | reduced-but-honest | 50 | 0.140 | 0.100 | 0.120 | 0.028 | 99 | unknown | OK |
| 3 | mediq_policy | reduced-but-honest | 50 | 0.100 | 0.100 | 0.100 | 0.000 | 95 | unknown | OK |
| 3 | one_shot | baseline | 50 | 0.060 | 0.040 | 0.050 | 0.014 | 96 | unknown | OK |
| 4 | variant_cue_heavy | unknown | 50 | 0.300 | 0.140 | 0.220 | 0.113 | 89 | unknown | OK |
| 4 | arcmemo_oe | unknown | 50 | 0.100 | 0.240 | 0.170 | 0.099 | 91 | unknown | OK |
| 4 | variant_minimal | unknown | 50 | 0.160 | 0.120 | 0.140 | 0.028 | 96 | unknown | OK |
| 4 | arcmemo_ps | baseline | 50 | 0.100 | 0.120 | 0.110 | 0.014 | 98 | unknown | OK |
| 4 | variant_free_text | partial-with-disclosed-gap | 50 | 0.100 | 0.100 | 0.100 | 0.000 | 98 | unknown | OK |
| 4 | variant_structured_routine | unknown | 50 | 0.120 | 0.080 | 0.100 | 0.028 | 98 | unknown | OK |
| 4 | variant_typed_only | unknown | 50 | 0.080 | 0.080 | 0.080 | 0.000 | 96 | unknown | OK |
| 4 | variant_gepa | reduced-but-honest | 50 | 0.040 | 0.060 | 0.050 | 0.014 | 96 | unknown | OK |
| 4 | variant_parse | partial-with-disclosed-gap | 50 | 0.040 | 0.060 | 0.050 | 0.014 | 97 | unknown | OK |
| 4 | variant_dspy_opt | faithful | 50 | 0.060 | 0.000 | 0.030 | 0.042 | 100 | unknown | OK |
| 5 | adas_style_search | surface-port-only-disclosed | 50 | 0.140 | 0.120 | 0.130 | 0.014 | 95 | unknown | OK |
| 5 | hand_coded_reorg | baseline | 50 | 0.140 | 0.100 | 0.120 | 0.028 | 97 | unknown | OK |
| 5 | alma_style_metaedit | surface-port-only-disclosed | 50 | 0.120 | 0.120 | 0.120 | 0.000 | 96 | unknown | OK |
| 6 | empty_start | baseline | 50 | 0.040 | 0.180 | 0.110 | 0.099 | 96 | unknown | OK |
| 6 | barc_seeded | reduced-but-honest | 50 | 0.100 | 0.080 | 0.090 | 0.014 | 99 | unknown | OK |
| 6 | corpus_hipporag_init | reduced-but-honest | 50 | 0.040 | 0.080 | 0.060 | 0.028 | 88 | unknown | OK |
| 6 | barc_synthetic | reduced-but-honest | 50 | 0.020 | 0.040 | 0.030 | 0.014 | 98 | unknown | OK |

## Findings to inspect

- No failed condition-seed runs.
- ANOMALY `magma_multigraph` seed 43: score=0.040
- ANOMALY `magma_multigraph` seed 42: score=0.020
- ANOMALY `reorg_evolver` seed 43: score=0.040
- ANOMALY `one_shot` seed 43: score=0.040
- ANOMALY `variant_dspy_opt` seed 43: score=0.000
- ANOMALY `variant_parse` seed 42: score=0.040
- ANOMALY `variant_gepa` seed 42: score=0.040
- ANOMALY `empty_start` seed 42: score=0.040
- ANOMALY `barc_synthetic` seed 42: score=0.020
- ANOMALY `barc_synthetic` seed 43: score=0.040
- ANOMALY `corpus_hipporag_init` seed 42: score=0.040

## Surface-tier footnotes

- `pathrag`: RN-007 downgrade: adapted entity paths are persisted and rendered as seed/context text, but runtime path selection still enumerates ConceptGraph paths and does not use artifact paths as the primary reliability-scored path source.
- `hipporag2_filter`: RN-007 downgrade: adapted passages and entity hints engage at runtime, but default second-stage filtering is template token overlap, not the paper's BERT-token-level relevance filter or DSPy/LLM-generated filter prompt loop.
- `adas_style_search`: ADAS's actual algorithm: meta agent iteratively PROGRAMS new agent designs in code, evaluated by RUNNING the generated agent on benchmark tasks (see third_party/adas/_arc/search.py). Our port: iterative reflexion (max 3 rounds) over the ...
- `alma_style_metaedit`: ALMA's actual algorithm: meta-learning over SHA-indexed memory-design ARCHIVE with MCTS-style selection, LLM-generated EXECUTABLE CODE for new memory designs, end-to-end agentic-eval reward signal (see third_party/alma/core/{meta_agent.p...

## Per-run trace links

- Axis 1 `colbert_rerank` seed 42: [2026-05-13T23-46-05Z_colbert_rerank_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-46-05Z_colbert_rerank_n50_seed42_phase-g-lite-2026-05-13/), score=0.200
- Axis 1 `colbert_rerank` seed 43: [2026-05-13T23-48-50Z_colbert_rerank_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-48-50Z_colbert_rerank_n50_seed43_phase-g-lite-2026-05-13/), score=0.160
- Axis 1 `flat_topk` seed 42: [2026-05-13T23-24-13Z_flat_topk_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-24-13Z_flat_topk_n50_seed42_phase-g-lite-2026-05-13/), score=0.180
- Axis 1 `flat_topk` seed 43: [2026-05-13T23-24-13Z_flat_topk_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-24-13Z_flat_topk_n50_seed43_phase-g-lite-2026-05-13/), score=0.300
- Axis 1 `graph_traversal` seed 42: [2026-05-13T23-24-13Z_graph_traversal_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-24-13Z_graph_traversal_n50_seed42_phase-g-lite-2026-05-13/), score=0.240
- Axis 1 `graph_traversal` seed 43: [2026-05-13T23-24-13Z_graph_traversal_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-24-13Z_graph_traversal_n50_seed43_phase-g-lite-2026-05-13/), score=0.160
- Axis 1 `graphrag` seed 42: [2026-05-13T23-34-51Z_graphrag_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-34-51Z_graphrag_n50_seed42_phase-g-lite-2026-05-13/), score=0.260
- Axis 1 `graphrag` seed 43: [2026-05-13T23-34-59Z_graphrag_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-34-59Z_graphrag_n50_seed43_phase-g-lite-2026-05-13/), score=0.340
- Axis 1 `hipporag2_filter` seed 42: [2026-05-13T23-29-30Z_hipporag2_filter_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-29-30Z_hipporag2_filter_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 1 `hipporag2_filter` seed 43: [2026-05-13T23-29-32Z_hipporag2_filter_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-29-32Z_hipporag2_filter_n50_seed43_phase-g-lite-2026-05-13/), score=0.080
- Axis 1 `hipporag_ppr` seed 42: [2026-05-13T23-24-13Z_hipporag_ppr_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-24-13Z_hipporag_ppr_n50_seed42_phase-g-lite-2026-05-13/), score=0.260
- Axis 1 `hipporag_ppr` seed 43: [2026-05-13T23-29-28Z_hipporag_ppr_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-29-28Z_hipporag_ppr_n50_seed43_phase-g-lite-2026-05-13/), score=0.160
- Axis 1 `hmem_hierarchical` seed 42: [2026-05-13T23-43-29Z_hmem_hierarchical_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-43-29Z_hmem_hierarchical_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 1 `hmem_hierarchical` seed 43: [2026-05-13T23-45-55Z_hmem_hierarchical_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-45-55Z_hmem_hierarchical_n50_seed43_phase-g-lite-2026-05-13/), score=0.180
- Axis 1 `lightrag` seed 42: [2026-05-13T23-35-02Z_lightrag_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-35-02Z_lightrag_n50_seed42_phase-g-lite-2026-05-13/), score=0.280
- Axis 1 `lightrag` seed 43: [2026-05-13T23-37-58Z_lightrag_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-37-58Z_lightrag_n50_seed43_phase-g-lite-2026-05-13/), score=0.320
- Axis 1 `magma_multigraph` seed 42: [2026-05-13T23-40-12Z_magma_multigraph_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-40-12Z_magma_multigraph_n50_seed42_phase-g-lite-2026-05-13/), score=0.020
- Axis 1 `magma_multigraph` seed 43: [2026-05-13T23-40-14Z_magma_multigraph_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-40-14Z_magma_multigraph_n50_seed43_phase-g-lite-2026-05-13/), score=0.040
- Axis 1 `pathrag` seed 42: [2026-05-13T23-40-28Z_pathrag_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-40-28Z_pathrag_n50_seed42_phase-g-lite-2026-05-13/), score=0.320
- Axis 1 `pathrag` seed 43: [2026-05-13T23-40-38Z_pathrag_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-40-38Z_pathrag_n50_seed43_phase-g-lite-2026-05-13/), score=0.280
- Axis 1 `raptor` seed 42: [2026-05-13T23-29-36Z_raptor_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-29-36Z_raptor_n50_seed42_phase-g-lite-2026-05-13/), score=0.140
- Axis 1 `raptor` seed 43: [2026-05-13T23-29-37Z_raptor_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-29-37Z_raptor_n50_seed43_phase-g-lite-2026-05-13/), score=0.180
- Axis 2 `accretive_prune` seed 42: [2026-05-13T23-52-43Z_accretive_prune_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-52-43Z_accretive_prune_n50_seed42_phase-g-lite-2026-05-13/), score=0.220
- Axis 2 `accretive_prune` seed 43: [2026-05-13T23-54-19Z_accretive_prune_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-54-19Z_accretive_prune_n50_seed43_phase-g-lite-2026-05-13/), score=0.300
- Axis 2 `reorg_amem` seed 42: [2026-05-14T00-13-41Z_reorg_amem_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-13-41Z_reorg_amem_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 2 `reorg_amem` seed 43: [2026-05-14T00-14-07Z_reorg_amem_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-14-07Z_reorg_amem_n50_seed43_phase-g-lite-2026-05-13/), score=0.220
- Axis 2 `reorg_dreamcoder` seed 42: [2026-05-13T23-59-38Z_reorg_dreamcoder_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-59-38Z_reorg_dreamcoder_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 2 `reorg_dreamcoder` seed 43: [2026-05-13T23-59-42Z_reorg_dreamcoder_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-59-42Z_reorg_dreamcoder_n50_seed43_phase-g-lite-2026-05-13/), score=0.220
- Axis 2 `reorg_evolver` seed 42: [2026-05-14T00-05-02Z_reorg_evolver_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-05-02Z_reorg_evolver_n50_seed42_phase-g-lite-2026-05-13/), score=0.200
- Axis 2 `reorg_evolver` seed 43: [2026-05-14T00-05-17Z_reorg_evolver_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-05-17Z_reorg_evolver_n50_seed43_phase-g-lite-2026-05-13/), score=0.040
- Axis 2 `reorg_lilo` seed 42: [2026-05-14T00-14-20Z_reorg_lilo_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-14-20Z_reorg_lilo_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 2 `reorg_lilo` seed 43: [2026-05-14T00-15-26Z_reorg_lilo_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-15-26Z_reorg_lilo_n50_seed43_phase-g-lite-2026-05-13/), score=0.160
- Axis 2 `reorg_lrll` seed 42: [2026-05-14T00-03-15Z_reorg_lrll_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-03-15Z_reorg_lrll_n50_seed42_phase-g-lite-2026-05-13/), score=0.060
- Axis 2 `reorg_lrll` seed 43: [2026-05-14T00-04-50Z_reorg_lrll_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-04-50Z_reorg_lrll_n50_seed43_phase-g-lite-2026-05-13/), score=0.200
- Axis 2 `reorg_memp` seed 42: [2026-05-14T00-10-10Z_reorg_memp_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-10-10Z_reorg_memp_n50_seed42_phase-g-lite-2026-05-13/), score=0.200
- Axis 2 `reorg_memp` seed 43: [2026-05-14T00-10-20Z_reorg_memp_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-10-20Z_reorg_memp_n50_seed43_phase-g-lite-2026-05-13/), score=0.220
- Axis 2 `reorg_memtree` seed 42: [2026-05-14T00-08-50Z_reorg_memtree_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-08-50Z_reorg_memtree_n50_seed42_phase-g-lite-2026-05-13/), score=0.240
- Axis 2 `reorg_memtree` seed 43: [2026-05-14T00-10-08Z_reorg_memtree_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-10-08Z_reorg_memtree_n50_seed43_phase-g-lite-2026-05-13/), score=0.140
- Axis 2 `reorg_off` seed 42: [2026-05-13T23-51-16Z_reorg_off_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-51-16Z_reorg_off_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 2 `reorg_off` seed 43: [2026-05-13T23-51-34Z_reorg_off_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-51-34Z_reorg_off_n50_seed43_phase-g-lite-2026-05-13/), score=0.160
- Axis 2 `reorg_on_graph_mdl_global_plateau` seed 42: [2026-05-13T23-54-41Z_reorg_on_graph_mdl_global_plateau_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-54-41Z_reorg_on_graph_mdl_global_plateau_n50_seed42_phase-g-lite-2026-05-13/), score=0.140
- Axis 2 `reorg_on_graph_mdl_global_plateau` seed 43: [2026-05-13T23-56-25Z_reorg_on_graph_mdl_global_plateau_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-56-25Z_reorg_on_graph_mdl_global_plateau_n50_seed43_phase-g-lite-2026-05-13/), score=0.100
- Axis 2 `reorg_on_trace_mdl_accretive_everyk` seed 42: [2026-05-13T23-56-49Z_reorg_on_trace_mdl_accretive_everyk_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-56-49Z_reorg_on_trace_mdl_accretive_everyk_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 2 `reorg_on_trace_mdl_accretive_everyk` seed 43: [2026-05-13T23-57-59Z_reorg_on_trace_mdl_accretive_everyk_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-13T23-57-59Z_reorg_on_trace_mdl_accretive_everyk_n50_seed43_phase-g-lite-2026-05-13/), score=0.120
- Axis 2 `reorg_sleepgate` seed 42: [2026-05-14T00-06-51Z_reorg_sleepgate_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-06-51Z_reorg_sleepgate_n50_seed42_phase-g-lite-2026-05-13/), score=0.160
- Axis 2 `reorg_sleepgate` seed 43: [2026-05-14T00-08-25Z_reorg_sleepgate_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-08-25Z_reorg_sleepgate_n50_seed43_phase-g-lite-2026-05-13/), score=0.160
- Axis 2 `reorg_stitch` seed 42: [2026-05-14T00-00-01Z_reorg_stitch_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-00-01Z_reorg_stitch_n50_seed42_phase-g-lite-2026-05-13/), score=0.080
- Axis 2 `reorg_stitch` seed 43: [2026-05-14T00-02-03Z_reorg_stitch_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-02-03Z_reorg_stitch_n50_seed43_phase-g-lite-2026-05-13/), score=0.080
- Axis 3 `mediq_policy` seed 42: [2026-05-14T00-20-40Z_mediq_policy_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-20-40Z_mediq_policy_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 3 `mediq_policy` seed 43: [2026-05-14T00-20-47Z_mediq_policy_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-20-47Z_mediq_policy_n50_seed43_phase-g-lite-2026-05-13/), score=0.100
- Axis 3 `one_shot` seed 42: [2026-05-14T00-15-37Z_one_shot_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-15-37Z_one_shot_n50_seed42_phase-g-lite-2026-05-13/), score=0.060
- Axis 3 `one_shot` seed 43: [2026-05-14T00-18-52Z_one_shot_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-18-52Z_one_shot_n50_seed43_phase-g-lite-2026-05-13/), score=0.040
- Axis 3 `rrmc_multi_round` seed 42: [2026-05-14T00-19-23Z_rrmc_multi_round_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-19-23Z_rrmc_multi_round_n50_seed42_phase-g-lite-2026-05-13/), score=0.140
- Axis 3 `rrmc_multi_round` seed 43: [2026-05-14T00-19-34Z_rrmc_multi_round_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-19-34Z_rrmc_multi_round_n50_seed43_phase-g-lite-2026-05-13/), score=0.100
- Axis 3 `uot_entropy` seed 42: [2026-05-14T00-23-34Z_uot_entropy_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-23-34Z_uot_entropy_n50_seed42_phase-g-lite-2026-05-13/), score=0.140
- Axis 3 `uot_entropy` seed 43: [2026-05-14T00-24-01Z_uot_entropy_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-24-01Z_uot_entropy_n50_seed43_phase-g-lite-2026-05-13/), score=0.220
- Axis 4 `arcmemo_oe` seed 42: [2026-05-14T00-24-53Z_arcmemo_oe_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-24-53Z_arcmemo_oe_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 4 `arcmemo_oe` seed 43: [2026-05-14T00-25-58Z_arcmemo_oe_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-25-58Z_arcmemo_oe_n50_seed43_phase-g-lite-2026-05-13/), score=0.240
- Axis 4 `arcmemo_ps` seed 42: [2026-05-14T00-26-10Z_arcmemo_ps_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-26-10Z_arcmemo_ps_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 4 `arcmemo_ps` seed 43: [2026-05-14T00-28-48Z_arcmemo_ps_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-28-48Z_arcmemo_ps_n50_seed43_phase-g-lite-2026-05-13/), score=0.120
- Axis 4 `variant_cue_heavy` seed 42: [2026-05-14T00-34-03Z_variant_cue_heavy_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-34-03Z_variant_cue_heavy_n50_seed42_phase-g-lite-2026-05-13/), score=0.300
- Axis 4 `variant_cue_heavy` seed 43: [2026-05-14T00-34-31Z_variant_cue_heavy_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-34-31Z_variant_cue_heavy_n50_seed43_phase-g-lite-2026-05-13/), score=0.140
- Axis 4 `variant_dspy_opt` seed 42: [2026-05-14T00-39-44Z_variant_dspy_opt_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-39-44Z_variant_dspy_opt_n50_seed42_phase-g-lite-2026-05-13/), score=0.060
- Axis 4 `variant_dspy_opt` seed 43: [2026-05-14T00-39-46Z_variant_dspy_opt_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-39-46Z_variant_dspy_opt_n50_seed43_phase-g-lite-2026-05-13/), score=0.000
- Axis 4 `variant_free_text` seed 42: [2026-05-14T00-34-35Z_variant_free_text_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-34-35Z_variant_free_text_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 4 `variant_free_text` seed 43: [2026-05-14T00-36-22Z_variant_free_text_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-36-22Z_variant_free_text_n50_seed43_phase-g-lite-2026-05-13/), score=0.100
- Axis 4 `variant_gepa` seed 42: [2026-05-14T00-40-59Z_variant_gepa_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-40-59Z_variant_gepa_n50_seed42_phase-g-lite-2026-05-13/), score=0.040
- Axis 4 `variant_gepa` seed 43: [2026-05-14T00-41-47Z_variant_gepa_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-41-47Z_variant_gepa_n50_seed43_phase-g-lite-2026-05-13/), score=0.060
- Axis 4 `variant_minimal` seed 42: [2026-05-14T00-29-18Z_variant_minimal_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-29-18Z_variant_minimal_n50_seed42_phase-g-lite-2026-05-13/), score=0.160
- Axis 4 `variant_minimal` seed 43: [2026-05-14T00-30-04Z_variant_minimal_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-30-04Z_variant_minimal_n50_seed43_phase-g-lite-2026-05-13/), score=0.120
- Axis 4 `variant_parse` seed 42: [2026-05-14T00-42-32Z_variant_parse_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-42-32Z_variant_parse_n50_seed42_phase-g-lite-2026-05-13/), score=0.040
- Axis 4 `variant_parse` seed 43: [2026-05-14T00-46-42Z_variant_parse_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-46-42Z_variant_parse_n50_seed43_phase-g-lite-2026-05-13/), score=0.060
- Axis 4 `variant_structured_routine` seed 42: [2026-05-14T00-36-33Z_variant_structured_routine_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-36-33Z_variant_structured_routine_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 4 `variant_structured_routine` seed 43: [2026-05-14T00-39-18Z_variant_structured_routine_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-39-18Z_variant_structured_routine_n50_seed43_phase-g-lite-2026-05-13/), score=0.080
- Axis 4 `variant_typed_only` seed 42: [2026-05-14T00-31-11Z_variant_typed_only_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-31-11Z_variant_typed_only_n50_seed42_phase-g-lite-2026-05-13/), score=0.080
- Axis 4 `variant_typed_only` seed 43: [2026-05-14T00-31-26Z_variant_typed_only_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-31-26Z_variant_typed_only_n50_seed43_phase-g-lite-2026-05-13/), score=0.080
- Axis 5 `adas_style_search` seed 42: [2026-05-14T00-52-22Z_adas_style_search_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-52-22Z_adas_style_search_n50_seed42_phase-g-lite-2026-05-13/), score=0.140
- Axis 5 `adas_style_search` seed 43: [2026-05-14T00-53-04Z_adas_style_search_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-53-04Z_adas_style_search_n50_seed43_phase-g-lite-2026-05-13/), score=0.120
- Axis 5 `alma_style_metaedit` seed 42: [2026-05-14T00-50-13Z_alma_style_metaedit_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-50-13Z_alma_style_metaedit_n50_seed42_phase-g-lite-2026-05-13/), score=0.120
- Axis 5 `alma_style_metaedit` seed 43: [2026-05-14T00-52-04Z_alma_style_metaedit_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-52-04Z_alma_style_metaedit_n50_seed43_phase-g-lite-2026-05-13/), score=0.120
- Axis 5 `hand_coded_reorg` seed 42: [2026-05-14T00-47-51Z_hand_coded_reorg_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-47-51Z_hand_coded_reorg_n50_seed42_phase-g-lite-2026-05-13/), score=0.140
- Axis 5 `hand_coded_reorg` seed 43: [2026-05-14T00-48-22Z_hand_coded_reorg_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-48-22Z_hand_coded_reorg_n50_seed43_phase-g-lite-2026-05-13/), score=0.100
- Axis 6 `barc_seeded` seed 42: [2026-05-14T00-57-19Z_barc_seeded_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-57-19Z_barc_seeded_n50_seed42_phase-g-lite-2026-05-13/), score=0.100
- Axis 6 `barc_seeded` seed 43: [2026-05-14T00-57-36Z_barc_seeded_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-57-36Z_barc_seeded_n50_seed43_phase-g-lite-2026-05-13/), score=0.080
- Axis 6 `barc_synthetic` seed 42: [2026-05-14T00-58-20Z_barc_synthetic_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-58-20Z_barc_synthetic_n50_seed42_phase-g-lite-2026-05-13/), score=0.020
- Axis 6 `barc_synthetic` seed 43: [2026-05-14T00-59-16Z_barc_synthetic_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-59-16Z_barc_synthetic_n50_seed43_phase-g-lite-2026-05-13/), score=0.040
- Axis 6 `corpus_hipporag_init` seed 42: [2026-05-14T01-00-41Z_corpus_hipporag_init_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T01-00-41Z_corpus_hipporag_init_n50_seed42_phase-g-lite-2026-05-13/), score=0.040
- Axis 6 `corpus_hipporag_init` seed 43: [2026-05-14T01-00-42Z_corpus_hipporag_init_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T01-00-42Z_corpus_hipporag_init_n50_seed43_phase-g-lite-2026-05-13/), score=0.080
- Axis 6 `empty_start` seed 42: [2026-05-14T00-54-06Z_empty_start_n50_seed42_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-54-06Z_empty_start_n50_seed42_phase-g-lite-2026-05-13/), score=0.040
- Axis 6 `empty_start` seed 43: [2026-05-14T00-55-29Z_empty_start_n50_seed43_phase-g-lite-2026-05-13](case_studies/runs/2026-05-14T00-55-29Z_empty_start_n50_seed43_phase-g-lite-2026-05-13/), score=0.180
