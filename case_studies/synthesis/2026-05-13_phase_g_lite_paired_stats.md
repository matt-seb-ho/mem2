# Paired Statistical Comparison - Phase G-Lite - 2026-05-13

## Methodology
- McNemar's test uses paired correct/wrong asymmetry with Yates continuity correction.
- Bootstrap 95% CI uses 1000 resamples of paired accuracy differences.
- Bonferroni correction across 990 condition pairs: alpha_corrected=5.05051e-05.
- Pairing key: seed plus ARC problem UID.

## Significantly different pairs

| Condition A | Condition B | A acc | B acc | Gap | McNemar p | CI 95 | n |
|---|---|---:|---:|---:|---:|---|---:|
| lightrag | magma_multigraph | 0.300 | 0.030 | 0.270 | 5.62e-07 | [0.180, 0.360] | 100 |
| barc_synthetic | lightrag | 0.030 | 0.300 | -0.270 | 1.38e-06 | [-0.360, -0.180] | 100 |
| graphrag | magma_multigraph | 0.300 | 0.030 | 0.270 | 1.38e-06 | [0.170, 0.370] | 100 |
| magma_multigraph | pathrag | 0.030 | 0.300 | -0.270 | 1.38e-06 | [-0.360, -0.180] | 100 |
| barc_synthetic | graphrag | 0.030 | 0.300 | -0.270 | 3.02e-06 | [-0.370, -0.180] | 100 |
| barc_synthetic | pathrag | 0.030 | 0.300 | -0.270 | 3.02e-06 | [-0.370, -0.180] | 100 |
| lightrag | variant_dspy_opt | 0.300 | 0.030 | 0.270 | 3.02e-06 | [0.170, 0.370] | 100 |
| lightrag | variant_gepa | 0.300 | 0.050 | 0.250 | 3.86e-06 | [0.160, 0.340] | 100 |
| graphrag | variant_dspy_opt | 0.300 | 0.030 | 0.270 | 6.01e-06 | [0.170, 0.370] | 100 |
| pathrag | variant_dspy_opt | 0.300 | 0.030 | 0.270 | 6.01e-06 | [0.170, 0.370] | 100 |
| lightrag | one_shot | 0.300 | 0.050 | 0.250 | 8.32e-06 | [0.160, 0.340] | 100 |
| lightrag | variant_parse | 0.300 | 0.050 | 0.250 | 8.32e-06 | [0.150, 0.340] | 100 |
| one_shot | pathrag | 0.050 | 0.300 | -0.250 | 8.32e-06 | [-0.340, -0.160] | 100 |
| pathrag | variant_parse | 0.300 | 0.050 | 0.250 | 8.32e-06 | [0.160, 0.350] | 100 |
| accretive_prune | barc_synthetic | 0.260 | 0.030 | 0.230 | 1.08e-05 | [0.150, 0.310] | 100 |
| accretive_prune | magma_multigraph | 0.260 | 0.030 | 0.230 | 1.08e-05 | [0.140, 0.320] | 100 |
| graphrag | variant_parse | 0.300 | 0.050 | 0.250 | 1.63e-05 | [0.140, 0.350] | 100 |
| pathrag | variant_gepa | 0.300 | 0.050 | 0.250 | 1.63e-05 | [0.150, 0.350] | 100 |
| pathrag | reorg_stitch | 0.300 | 0.080 | 0.220 | 1.81e-05 | [0.140, 0.310] | 100 |
| accretive_prune | variant_dspy_opt | 0.260 | 0.030 | 0.230 | 2.30e-05 | [0.140, 0.320] | 100 |
| corpus_hipporag_init | pathrag | 0.060 | 0.300 | -0.240 | 2.68e-05 | [-0.340, -0.150] | 100 |
| graphrag | one_shot | 0.300 | 0.050 | 0.250 | 2.94e-05 | [0.140, 0.350] | 100 |
| graphrag | variant_gepa | 0.300 | 0.050 | 0.250 | 2.94e-05 | [0.150, 0.350] | 100 |
| flat_topk | magma_multigraph | 0.240 | 0.030 | 0.210 | 3.04e-05 | [0.130, 0.300] | 100 |
| lightrag | reorg_stitch | 0.300 | 0.080 | 0.220 | 3.81e-05 | [0.130, 0.310] | 100 |
| lightrag | variant_typed_only | 0.300 | 0.080 | 0.220 | 3.81e-05 | [0.130, 0.310] | 100 |
| corpus_hipporag_init | lightrag | 0.060 | 0.300 | -0.240 | 4.79e-05 | [-0.340, -0.140] | 100 |

## Per-axis ranking

### Axis 1

| Rank | Condition | Mean acc | vs flat_topk gap CI | n |
|---:|---|---:|---|---:|
| 1 | graphrag | 0.300 | [-0.050, 0.160] | 100 |
| 2 | lightrag | 0.300 | [-0.040, 0.160] | 100 |
| 3 | pathrag | 0.300 | [-0.050, 0.180] | 100 |
| 4 | flat_topk | 0.240 | [0.000, 0.000] | 100 |
| 5 | hipporag_ppr | 0.210 | [-0.130, 0.070] | 100 |
| 6 | graph_traversal | 0.200 | [-0.140, 0.070] | 100 |
| 7 | colbert_rerank | 0.180 | [-0.170, 0.040] | 100 |
| 8 | raptor | 0.160 | [-0.180, 0.020] | 100 |
| 9 | hmem_hierarchical | 0.150 | [-0.190, 0.020] | 100 |
| 10 | hipporag2_filter | 0.100 | [-0.250, -0.040] | 100 |
| 11 | magma_multigraph | 0.030 | [-0.300, -0.130] | 100 |

### Axis 2

| Rank | Condition | Mean acc | vs flat_topk gap CI | n |
|---:|---|---:|---|---:|
| 1 | accretive_prune | 0.260 | [-0.090, 0.130] | 100 |
| 2 | reorg_memp | 0.210 | [-0.140, 0.080] | 100 |
| 3 | reorg_memtree | 0.190 | [-0.160, 0.070] | 100 |
| 4 | reorg_dreamcoder | 0.170 | [-0.170, 0.050] | 100 |
| 5 | reorg_amem | 0.170 | [-0.170, 0.020] | 100 |
| 6 | reorg_sleepgate | 0.160 | [-0.190, 0.030] | 100 |
| 7 | reorg_lilo | 0.140 | [-0.210, 0.010] | 100 |
| 8 | reorg_off | 0.130 | [-0.210, -0.010] | 100 |
| 9 | reorg_lrll | 0.130 | [-0.200, -0.020] | 100 |
| 10 | reorg_on_graph_mdl_global_plateau | 0.120 | [-0.220, -0.030] | 100 |
| 11 | reorg_evolver | 0.120 | [-0.220, -0.020] | 100 |
| 12 | reorg_on_trace_mdl_accretive_everyk | 0.110 | [-0.230, -0.030] | 100 |
| 13 | reorg_stitch | 0.080 | [-0.260, -0.060] | 100 |

### Axis 3

| Rank | Condition | Mean acc | vs flat_topk gap CI | n |
|---:|---|---:|---|---:|
| 1 | uot_entropy | 0.180 | [-0.160, 0.040] | 100 |
| 2 | rrmc_multi_round | 0.120 | [-0.220, -0.020] | 100 |
| 3 | mediq_policy | 0.100 | [-0.230, -0.050] | 100 |
| 4 | one_shot | 0.050 | [-0.280, -0.100] | 100 |

### Axis 4

| Rank | Condition | Mean acc | vs flat_topk gap CI | n |
|---:|---|---:|---|---:|
| 1 | variant_cue_heavy | 0.220 | [-0.130, 0.100] | 100 |
| 2 | arcmemo_oe | 0.170 | [-0.170, 0.030] | 100 |
| 3 | variant_minimal | 0.140 | [-0.200, -0.000] | 100 |
| 4 | arcmemo_ps | 0.110 | [-0.230, -0.030] | 100 |
| 5 | variant_free_text | 0.100 | [-0.240, -0.040] | 100 |
| 6 | variant_structured_routine | 0.100 | [-0.230, -0.040] | 100 |
| 7 | variant_typed_only | 0.080 | [-0.250, -0.070] | 100 |
| 8 | variant_parse | 0.050 | [-0.280, -0.100] | 100 |
| 9 | variant_gepa | 0.050 | [-0.280, -0.100] | 100 |
| 10 | variant_dspy_opt | 0.030 | [-0.300, -0.120] | 100 |

### Axis 5

| Rank | Condition | Mean acc | vs flat_topk gap CI | n |
|---:|---|---:|---|---:|
| 1 | adas_style_search | 0.130 | [-0.210, 0.000] | 100 |
| 2 | hand_coded_reorg | 0.120 | [-0.220, -0.010] | 100 |
| 3 | alma_style_metaedit | 0.120 | [-0.230, -0.010] | 100 |

### Axis 6

| Rank | Condition | Mean acc | vs flat_topk gap CI | n |
|---:|---|---:|---|---:|
| 1 | empty_start | 0.110 | [-0.230, -0.030] | 100 |
| 2 | barc_seeded | 0.090 | [-0.260, -0.040] | 100 |
| 3 | corpus_hipporag_init | 0.060 | [-0.280, -0.080] | 100 |
| 4 | barc_synthetic | 0.030 | [-0.300, -0.120] | 100 |

## All pairwise comparisons

| Condition A | Condition B | A acc | B acc | Gap | McNemar p | CI 95 | b | c | n |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| colbert_rerank | corpus_hipporag_init | 0.180 | 0.060 | 0.120 | 0.0060 | [0.050, 0.200] | 14 | 2 | 100 |
| colbert_rerank | empty_start | 0.180 | 0.110 | 0.070 | 0.2109 | [-0.020, 0.170] | 15 | 8 | 100 |
| colbert_rerank | flat_topk | 0.180 | 0.240 | -0.060 | 0.3768 | [-0.170, 0.040] | 13 | 19 | 100 |
| colbert_rerank | graph_traversal | 0.180 | 0.200 | -0.020 | 0.8383 | [-0.110, 0.070] | 11 | 13 | 100 |
| colbert_rerank | graphrag | 0.180 | 0.300 | -0.120 | 0.0668 | [-0.240, 0.000] | 12 | 24 | 100 |
| colbert_rerank | hand_coded_reorg | 0.180 | 0.120 | 0.060 | 0.1814 | [-0.020, 0.130] | 10 | 4 | 100 |
| colbert_rerank | hipporag2_filter | 0.180 | 0.100 | 0.080 | 0.1530 | [-0.010, 0.180] | 16 | 8 | 100 |
| colbert_rerank | hipporag_ppr | 0.180 | 0.210 | -0.030 | 0.6625 | [-0.110, 0.060] | 9 | 12 | 100 |
| colbert_rerank | hmem_hierarchical | 0.180 | 0.150 | 0.030 | 0.6625 | [-0.060, 0.120] | 12 | 9 | 100 |
| colbert_rerank | lightrag | 0.180 | 0.300 | -0.120 | 0.0446 | [-0.220, -0.020] | 9 | 21 | 100 |
| colbert_rerank | magma_multigraph | 0.180 | 0.030 | 0.150 | 0.0013 | [0.070, 0.240] | 17 | 2 | 100 |
| colbert_rerank | mediq_policy | 0.180 | 0.100 | 0.080 | 0.0990 | [0.000, 0.160] | 13 | 5 | 100 |
| colbert_rerank | one_shot | 0.180 | 0.050 | 0.130 | 0.0019 | [0.060, 0.200] | 14 | 1 | 100 |
| colbert_rerank | pathrag | 0.180 | 0.300 | -0.120 | 0.0592 | [-0.230, -0.010] | 11 | 23 | 100 |
| colbert_rerank | raptor | 0.180 | 0.160 | 0.020 | 0.8231 | [-0.060, 0.100] | 11 | 9 | 100 |
| colbert_rerank | reorg_amem | 0.180 | 0.170 | 0.010 | 1.0000 | [-0.090, 0.110] | 13 | 12 | 100 |
| colbert_rerank | reorg_dreamcoder | 0.180 | 0.170 | 0.010 | 1.0000 | [-0.080, 0.100] | 11 | 10 | 100 |
| colbert_rerank | reorg_evolver | 0.180 | 0.120 | 0.060 | 0.2864 | [-0.030, 0.150] | 14 | 8 | 100 |
| colbert_rerank | reorg_lilo | 0.180 | 0.140 | 0.040 | 0.5403 | [-0.060, 0.130] | 14 | 10 | 100 |
| colbert_rerank | reorg_lrll | 0.180 | 0.130 | 0.050 | 0.4042 | [-0.040, 0.140] | 14 | 9 | 100 |
| colbert_rerank | reorg_memp | 0.180 | 0.210 | -0.030 | 0.7103 | [-0.140, 0.080] | 13 | 16 | 100 |
| colbert_rerank | reorg_memtree | 0.180 | 0.190 | -0.010 | 1.0000 | [-0.110, 0.080] | 11 | 12 | 100 |
| colbert_rerank | reorg_off | 0.180 | 0.130 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| colbert_rerank | reorg_on_graph_mdl_global_plateau | 0.180 | 0.120 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| colbert_rerank | reorg_on_trace_mdl_accretive_everyk | 0.180 | 0.110 | 0.070 | 0.1213 | [0.000, 0.150] | 11 | 4 | 100 |
| colbert_rerank | reorg_sleepgate | 0.180 | 0.160 | 0.020 | 0.8445 | [-0.080, 0.120] | 14 | 12 | 100 |
| colbert_rerank | reorg_stitch | 0.180 | 0.080 | 0.100 | 0.0162 | [0.040, 0.170] | 12 | 2 | 100 |
| colbert_rerank | rrmc_multi_round | 0.180 | 0.120 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| colbert_rerank | uot_entropy | 0.180 | 0.180 | 0.000 | 1.0000 | [-0.100, 0.100] | 13 | 13 | 100 |
| colbert_rerank | variant_cue_heavy | 0.180 | 0.220 | -0.040 | 0.5839 | [-0.150, 0.070] | 13 | 17 | 100 |
| colbert_rerank | variant_dspy_opt | 0.180 | 0.030 | 0.150 | 6.85e-04 | [0.070, 0.230] | 16 | 1 | 100 |
| colbert_rerank | variant_free_text | 0.180 | 0.100 | 0.080 | 0.1175 | [0.000, 0.170] | 14 | 6 | 100 |
| colbert_rerank | variant_gepa | 0.180 | 0.050 | 0.130 | 0.0059 | [0.050, 0.220] | 16 | 3 | 100 |
| colbert_rerank | variant_minimal | 0.180 | 0.140 | 0.040 | 0.5023 | [-0.040, 0.130] | 12 | 8 | 100 |
| colbert_rerank | variant_parse | 0.180 | 0.050 | 0.130 | 0.0088 | [0.050, 0.220] | 17 | 4 | 100 |
| colbert_rerank | variant_structured_routine | 0.180 | 0.100 | 0.080 | 0.0801 | [0.000, 0.160] | 12 | 4 | 100 |
| colbert_rerank | variant_typed_only | 0.180 | 0.080 | 0.100 | 0.0244 | [0.030, 0.170] | 13 | 3 | 100 |
| flat_topk | graph_traversal | 0.240 | 0.200 | 0.040 | 0.5708 | [-0.070, 0.140] | 16 | 12 | 100 |
| flat_topk | graphrag | 0.240 | 0.300 | -0.060 | 0.3268 | [-0.160, 0.050] | 10 | 16 | 100 |
| flat_topk | hand_coded_reorg | 0.240 | 0.120 | 0.120 | 0.0518 | [0.010, 0.220] | 22 | 10 | 100 |
| flat_topk | hipporag2_filter | 0.240 | 0.100 | 0.140 | 0.0176 | [0.040, 0.250] | 22 | 8 | 100 |
| flat_topk | hipporag_ppr | 0.240 | 0.210 | 0.030 | 0.7003 | [-0.070, 0.130] | 15 | 12 | 100 |
| flat_topk | hmem_hierarchical | 0.240 | 0.150 | 0.090 | 0.1508 | [-0.020, 0.190] | 20 | 11 | 100 |
| flat_topk | lightrag | 0.240 | 0.300 | -0.060 | 0.3447 | [-0.160, 0.040] | 11 | 17 | 100 |
| flat_topk | magma_multigraph | 0.240 | 0.030 | 0.210 | 3.04e-05 | [0.130, 0.300] | 22 | 1 | 100 |
| flat_topk | mediq_policy | 0.240 | 0.100 | 0.140 | 0.0108 | [0.050, 0.230] | 20 | 6 | 100 |
| flat_topk | one_shot | 0.240 | 0.050 | 0.190 | 5.32e-04 | [0.100, 0.280] | 23 | 4 | 100 |
| flat_topk | pathrag | 0.240 | 0.300 | -0.060 | 0.3768 | [-0.180, 0.050] | 13 | 19 | 100 |
| flat_topk | raptor | 0.240 | 0.160 | 0.080 | 0.1859 | [-0.020, 0.180] | 18 | 10 | 100 |
| flat_topk | reorg_amem | 0.240 | 0.170 | 0.070 | 0.2301 | [-0.020, 0.170] | 16 | 9 | 100 |
| flat_topk | reorg_dreamcoder | 0.240 | 0.170 | 0.070 | 0.3105 | [-0.050, 0.170] | 21 | 14 | 100 |
| flat_topk | reorg_evolver | 0.240 | 0.120 | 0.120 | 0.0446 | [0.020, 0.220] | 21 | 9 | 100 |
| flat_topk | reorg_lilo | 0.240 | 0.140 | 0.100 | 0.1003 | [-0.010, 0.210] | 20 | 10 | 100 |
| flat_topk | reorg_lrll | 0.240 | 0.130 | 0.110 | 0.0371 | [0.020, 0.200] | 17 | 6 | 100 |
| flat_topk | reorg_memp | 0.240 | 0.210 | 0.030 | 0.7194 | [-0.080, 0.140] | 17 | 14 | 100 |
| flat_topk | reorg_memtree | 0.240 | 0.190 | 0.050 | 0.5108 | [-0.070, 0.160] | 21 | 16 | 100 |
| flat_topk | reorg_off | 0.240 | 0.130 | 0.110 | 0.0633 | [0.010, 0.210] | 20 | 9 | 100 |
| flat_topk | reorg_on_graph_mdl_global_plateau | 0.240 | 0.120 | 0.120 | 0.0376 | [0.030, 0.220] | 20 | 8 | 100 |
| flat_topk | reorg_on_trace_mdl_accretive_everyk | 0.240 | 0.110 | 0.130 | 0.0209 | [0.030, 0.230] | 20 | 7 | 100 |
| flat_topk | reorg_sleepgate | 0.240 | 0.160 | 0.080 | 0.2012 | [-0.030, 0.190] | 19 | 11 | 100 |
| flat_topk | reorg_stitch | 0.240 | 0.080 | 0.160 | 0.0046 | [0.060, 0.260] | 22 | 6 | 100 |
| flat_topk | rrmc_multi_round | 0.240 | 0.120 | 0.120 | 0.0446 | [0.020, 0.220] | 21 | 9 | 100 |
| flat_topk | uot_entropy | 0.240 | 0.180 | 0.060 | 0.3268 | [-0.040, 0.160] | 16 | 10 | 100 |
| flat_topk | variant_cue_heavy | 0.240 | 0.220 | 0.020 | 0.8638 | [-0.100, 0.130] | 18 | 16 | 100 |
| flat_topk | variant_dspy_opt | 0.240 | 0.030 | 0.210 | 6.33e-05 | [0.120, 0.300] | 23 | 2 | 100 |
| flat_topk | variant_free_text | 0.240 | 0.100 | 0.140 | 0.0140 | [0.040, 0.240] | 21 | 7 | 100 |
| flat_topk | variant_gepa | 0.240 | 0.050 | 0.190 | 3.18e-04 | [0.100, 0.280] | 22 | 3 | 100 |
| flat_topk | variant_minimal | 0.240 | 0.140 | 0.100 | 0.0890 | [0.000, 0.200] | 19 | 9 | 100 |
| flat_topk | variant_parse | 0.240 | 0.050 | 0.190 | 3.18e-04 | [0.100, 0.280] | 22 | 3 | 100 |
| flat_topk | variant_structured_routine | 0.240 | 0.100 | 0.140 | 0.0108 | [0.040, 0.230] | 20 | 6 | 100 |
| flat_topk | variant_typed_only | 0.240 | 0.080 | 0.160 | 0.0033 | [0.070, 0.250] | 21 | 5 | 100 |
| graph_traversal | graphrag | 0.200 | 0.300 | -0.100 | 0.1003 | [-0.210, 0.010] | 10 | 20 | 100 |
| graph_traversal | hand_coded_reorg | 0.200 | 0.120 | 0.080 | 0.1356 | [-0.010, 0.170] | 15 | 7 | 100 |
| graph_traversal | hipporag2_filter | 0.200 | 0.100 | 0.100 | 0.0662 | [0.010, 0.200] | 17 | 7 | 100 |
| graph_traversal | hipporag_ppr | 0.200 | 0.210 | -0.010 | 1.0000 | [-0.100, 0.080] | 11 | 12 | 100 |
| graph_traversal | hmem_hierarchical | 0.200 | 0.150 | 0.050 | 0.4237 | [-0.050, 0.150] | 15 | 10 | 100 |
| graph_traversal | lightrag | 0.200 | 0.300 | -0.100 | 0.0890 | [-0.200, 0.010] | 9 | 19 | 100 |
| graph_traversal | magma_multigraph | 0.200 | 0.030 | 0.170 | 4.80e-04 | [0.090, 0.250] | 19 | 2 | 100 |
| graph_traversal | mediq_policy | 0.200 | 0.100 | 0.100 | 0.0776 | [0.000, 0.200] | 18 | 8 | 100 |
| graph_traversal | one_shot | 0.200 | 0.050 | 0.150 | 0.0035 | [0.070, 0.240] | 19 | 4 | 100 |
| graph_traversal | pathrag | 0.200 | 0.300 | -0.100 | 0.0890 | [-0.200, 0.000] | 9 | 19 | 100 |
| graph_traversal | raptor | 0.200 | 0.160 | 0.040 | 0.5563 | [-0.060, 0.140] | 15 | 11 | 100 |
| graph_traversal | reorg_amem | 0.200 | 0.170 | 0.030 | 0.7003 | [-0.070, 0.130] | 15 | 12 | 100 |
| graph_traversal | reorg_dreamcoder | 0.200 | 0.170 | 0.030 | 0.6892 | [-0.070, 0.130] | 14 | 11 | 100 |
| graph_traversal | reorg_evolver | 0.200 | 0.120 | 0.080 | 0.1530 | [-0.010, 0.170] | 16 | 8 | 100 |
| graph_traversal | reorg_lilo | 0.200 | 0.140 | 0.060 | 0.3268 | [-0.040, 0.160] | 16 | 10 | 100 |
| graph_traversal | reorg_lrll | 0.200 | 0.130 | 0.070 | 0.2482 | [-0.030, 0.170] | 17 | 10 | 100 |
| graph_traversal | reorg_memp | 0.200 | 0.210 | -0.010 | 1.0000 | [-0.120, 0.100] | 15 | 16 | 100 |
| graph_traversal | reorg_memtree | 0.200 | 0.190 | 0.010 | 1.0000 | [-0.110, 0.120] | 17 | 16 | 100 |
| graph_traversal | reorg_off | 0.200 | 0.130 | 0.070 | 0.2109 | [-0.020, 0.160] | 15 | 8 | 100 |
| graph_traversal | reorg_on_graph_mdl_global_plateau | 0.200 | 0.120 | 0.080 | 0.1698 | [-0.010, 0.180] | 17 | 9 | 100 |
| graph_traversal | reorg_on_trace_mdl_accretive_everyk | 0.200 | 0.110 | 0.090 | 0.1096 | [-0.010, 0.190] | 17 | 8 | 100 |
| graph_traversal | reorg_sleepgate | 0.200 | 0.160 | 0.040 | 0.5708 | [-0.060, 0.140] | 16 | 12 | 100 |
| graph_traversal | reorg_stitch | 0.200 | 0.080 | 0.120 | 0.0095 | [0.040, 0.200] | 15 | 3 | 100 |
| graph_traversal | rrmc_multi_round | 0.200 | 0.120 | 0.080 | 0.1175 | [0.000, 0.170] | 14 | 6 | 100 |
| graph_traversal | uot_entropy | 0.200 | 0.180 | 0.020 | 0.8501 | [-0.080, 0.120] | 15 | 13 | 100 |
| graph_traversal | variant_cue_heavy | 0.200 | 0.220 | -0.020 | 0.8445 | [-0.120, 0.080] | 12 | 14 | 100 |
| graph_traversal | variant_dspy_opt | 0.200 | 0.030 | 0.170 | 2.42e-04 | [0.090, 0.250] | 18 | 1 | 100 |
| graph_traversal | variant_free_text | 0.200 | 0.100 | 0.100 | 0.0776 | [0.000, 0.200] | 18 | 8 | 100 |
| graph_traversal | variant_gepa | 0.200 | 0.050 | 0.150 | 0.0023 | [0.060, 0.240] | 18 | 3 | 100 |
| graph_traversal | variant_minimal | 0.200 | 0.140 | 0.060 | 0.2864 | [-0.030, 0.160] | 14 | 8 | 100 |
| graph_traversal | variant_parse | 0.200 | 0.050 | 0.150 | 0.0035 | [0.060, 0.240] | 19 | 4 | 100 |
| graph_traversal | variant_structured_routine | 0.200 | 0.100 | 0.100 | 0.0776 | [0.000, 0.200] | 18 | 8 | 100 |
| graph_traversal | variant_typed_only | 0.200 | 0.080 | 0.120 | 0.0139 | [0.040, 0.200] | 16 | 4 | 100 |
| graphrag | hand_coded_reorg | 0.300 | 0.120 | 0.180 | 0.0058 | [0.060, 0.290] | 28 | 10 | 100 |
| graphrag | hipporag2_filter | 0.300 | 0.100 | 0.200 | 3.30e-04 | [0.100, 0.300] | 24 | 4 | 100 |
| graphrag | hipporag_ppr | 0.300 | 0.210 | 0.090 | 0.2002 | [-0.040, 0.220] | 24 | 15 | 100 |
| graphrag | hmem_hierarchical | 0.300 | 0.150 | 0.150 | 0.0035 | [0.060, 0.240] | 19 | 4 | 100 |
| graphrag | lightrag | 0.300 | 0.300 | 0.000 | 1.0000 | [-0.110, 0.110] | 16 | 16 | 100 |
| graphrag | magma_multigraph | 0.300 | 0.030 | 0.270 | 1.38e-06 | [0.170, 0.370] | 28 | 1 | 100 |
| graphrag | mediq_policy | 0.300 | 0.100 | 0.200 | 7.83e-04 | [0.090, 0.310] | 26 | 6 | 100 |
| graphrag | one_shot | 0.300 | 0.050 | 0.250 | 2.94e-05 | [0.140, 0.350] | 29 | 4 | 100 |
| graphrag | pathrag | 0.300 | 0.300 | 0.000 | 1.0000 | [-0.130, 0.120] | 18 | 18 | 100 |
| graphrag | raptor | 0.300 | 0.160 | 0.140 | 0.0176 | [0.030, 0.240] | 22 | 8 | 100 |
| graphrag | reorg_amem | 0.300 | 0.170 | 0.130 | 0.0425 | [0.010, 0.240] | 24 | 11 | 100 |
| graphrag | reorg_dreamcoder | 0.300 | 0.170 | 0.130 | 0.0367 | [0.020, 0.240] | 23 | 10 | 100 |
| graphrag | reorg_evolver | 0.300 | 0.120 | 0.180 | 0.0027 | [0.070, 0.280] | 25 | 7 | 100 |
| graphrag | reorg_lilo | 0.300 | 0.140 | 0.160 | 0.0080 | [0.050, 0.260] | 24 | 8 | 100 |
| graphrag | reorg_lrll | 0.300 | 0.130 | 0.170 | 0.0021 | [0.070, 0.270] | 22 | 5 | 100 |
| graphrag | reorg_memp | 0.300 | 0.210 | 0.090 | 0.1237 | [-0.010, 0.190] | 18 | 9 | 100 |
| graphrag | reorg_memtree | 0.300 | 0.190 | 0.110 | 0.1273 | [-0.020, 0.230] | 27 | 16 | 100 |
| graphrag | reorg_off | 0.300 | 0.130 | 0.170 | 0.0030 | [0.070, 0.280] | 23 | 6 | 100 |
| graphrag | reorg_on_graph_mdl_global_plateau | 0.300 | 0.120 | 0.180 | 0.0036 | [0.070, 0.280] | 26 | 8 | 100 |
| graphrag | reorg_on_trace_mdl_accretive_everyk | 0.300 | 0.110 | 0.190 | 0.0017 | [0.080, 0.290] | 26 | 7 | 100 |
| graphrag | reorg_sleepgate | 0.300 | 0.160 | 0.140 | 0.0303 | [0.020, 0.260] | 25 | 11 | 100 |
| graphrag | reorg_stitch | 0.300 | 0.080 | 0.220 | 2.05e-04 | [0.110, 0.320] | 27 | 5 | 100 |
| graphrag | rrmc_multi_round | 0.300 | 0.120 | 0.180 | 0.0027 | [0.070, 0.290] | 25 | 7 | 100 |
| graphrag | uot_entropy | 0.300 | 0.180 | 0.120 | 0.0310 | [0.020, 0.220] | 19 | 7 | 100 |
| graphrag | variant_cue_heavy | 0.300 | 0.220 | 0.080 | 0.2801 | [-0.050, 0.190] | 25 | 17 | 100 |
| graphrag | variant_dspy_opt | 0.300 | 0.030 | 0.270 | 6.01e-06 | [0.170, 0.370] | 30 | 3 | 100 |
| graphrag | variant_free_text | 0.300 | 0.100 | 0.200 | 3.30e-04 | [0.100, 0.300] | 24 | 4 | 100 |
| graphrag | variant_gepa | 0.300 | 0.050 | 0.250 | 2.94e-05 | [0.150, 0.350] | 29 | 4 | 100 |
| graphrag | variant_minimal | 0.300 | 0.140 | 0.160 | 0.0080 | [0.050, 0.270] | 24 | 8 | 100 |
| graphrag | variant_parse | 0.300 | 0.050 | 0.250 | 1.63e-05 | [0.140, 0.350] | 28 | 3 | 100 |
| graphrag | variant_structured_routine | 0.300 | 0.100 | 0.200 | 0.0011 | [0.090, 0.300] | 27 | 7 | 100 |
| graphrag | variant_typed_only | 0.300 | 0.080 | 0.220 | 1.26e-04 | [0.120, 0.330] | 26 | 4 | 100 |
| hipporag2_filter | hipporag_ppr | 0.100 | 0.210 | -0.110 | 0.0543 | [-0.210, -0.010] | 8 | 19 | 100 |
| hipporag2_filter | hmem_hierarchical | 0.100 | 0.150 | -0.050 | 0.3320 | [-0.130, 0.030] | 6 | 11 | 100 |
| hipporag2_filter | lightrag | 0.100 | 0.300 | -0.200 | 5.23e-04 | [-0.300, -0.100] | 5 | 25 | 100 |
| hipporag2_filter | magma_multigraph | 0.100 | 0.030 | 0.070 | 0.0704 | [0.010, 0.140] | 9 | 2 | 100 |
| hipporag2_filter | mediq_policy | 0.100 | 0.100 | 0.000 | 1.0000 | [-0.080, 0.080] | 9 | 9 | 100 |
| hipporag2_filter | one_shot | 0.100 | 0.050 | 0.050 | 0.3017 | [-0.020, 0.120] | 10 | 5 | 100 |
| hipporag2_filter | pathrag | 0.100 | 0.300 | -0.200 | 7.83e-04 | [-0.300, -0.100] | 6 | 26 | 100 |
| hipporag2_filter | raptor | 0.100 | 0.160 | -0.060 | 0.2864 | [-0.150, 0.030] | 8 | 14 | 100 |
| hipporag2_filter | reorg_amem | 0.100 | 0.170 | -0.070 | 0.2301 | [-0.170, 0.030] | 9 | 16 | 100 |
| hipporag2_filter | reorg_dreamcoder | 0.100 | 0.170 | -0.070 | 0.2109 | [-0.160, 0.020] | 8 | 15 | 100 |
| hipporag2_filter | reorg_evolver | 0.100 | 0.120 | -0.020 | 0.8026 | [-0.090, 0.060] | 7 | 9 | 100 |
| hipporag2_filter | reorg_lilo | 0.100 | 0.140 | -0.040 | 0.5023 | [-0.130, 0.040] | 8 | 12 | 100 |
| hipporag2_filter | reorg_lrll | 0.100 | 0.130 | -0.030 | 0.6625 | [-0.120, 0.060] | 9 | 12 | 100 |
| hipporag2_filter | reorg_memp | 0.100 | 0.210 | -0.110 | 0.0371 | [-0.200, -0.020] | 6 | 17 | 100 |
| hipporag2_filter | reorg_memtree | 0.100 | 0.190 | -0.090 | 0.1237 | [-0.190, 0.010] | 9 | 18 | 100 |
| hipporag2_filter | reorg_off | 0.100 | 0.130 | -0.030 | 0.6276 | [-0.110, 0.050] | 7 | 10 | 100 |
| hipporag2_filter | reorg_on_graph_mdl_global_plateau | 0.100 | 0.120 | -0.020 | 0.8137 | [-0.110, 0.060] | 8 | 10 | 100 |
| hipporag2_filter | reorg_on_trace_mdl_accretive_everyk | 0.100 | 0.110 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| hipporag2_filter | reorg_sleepgate | 0.100 | 0.160 | -0.060 | 0.3074 | [-0.160, 0.040] | 9 | 15 | 100 |
| hipporag2_filter | reorg_stitch | 0.100 | 0.080 | 0.020 | 0.8137 | [-0.060, 0.100] | 10 | 8 | 100 |
| hipporag2_filter | rrmc_multi_round | 0.100 | 0.120 | -0.020 | 0.8137 | [-0.110, 0.070] | 8 | 10 | 100 |
| hipporag2_filter | uot_entropy | 0.100 | 0.180 | -0.080 | 0.1175 | [-0.160, 0.000] | 6 | 14 | 100 |
| hipporag2_filter | variant_cue_heavy | 0.100 | 0.220 | -0.120 | 0.0247 | [-0.210, -0.030] | 6 | 18 | 100 |
| hipporag2_filter | variant_dspy_opt | 0.100 | 0.030 | 0.070 | 0.0961 | [0.000, 0.140] | 10 | 3 | 100 |
| hipporag2_filter | variant_free_text | 0.100 | 0.100 | 0.000 | 1.0000 | [-0.080, 0.070] | 8 | 8 | 100 |
| hipporag2_filter | variant_gepa | 0.100 | 0.050 | 0.050 | 0.2278 | [-0.010, 0.120] | 8 | 3 | 100 |
| hipporag2_filter | variant_minimal | 0.100 | 0.140 | -0.040 | 0.4795 | [-0.120, 0.050] | 7 | 11 | 100 |
| hipporag2_filter | variant_parse | 0.100 | 0.050 | 0.050 | 0.2278 | [-0.010, 0.110] | 8 | 3 | 100 |
| hipporag2_filter | variant_structured_routine | 0.100 | 0.100 | 0.000 | 1.0000 | [-0.070, 0.070] | 7 | 7 | 100 |
| hipporag2_filter | variant_typed_only | 0.100 | 0.080 | 0.020 | 0.7893 | [-0.060, 0.090] | 8 | 6 | 100 |
| hipporag_ppr | hmem_hierarchical | 0.210 | 0.150 | 0.060 | 0.2864 | [-0.030, 0.150] | 14 | 8 | 100 |
| hipporag_ppr | lightrag | 0.210 | 0.300 | -0.090 | 0.1237 | [-0.200, 0.010] | 9 | 18 | 100 |
| hipporag_ppr | magma_multigraph | 0.210 | 0.030 | 0.180 | 1.44e-04 | [0.100, 0.270] | 19 | 1 | 100 |
| hipporag_ppr | mediq_policy | 0.210 | 0.100 | 0.110 | 0.0153 | [0.040, 0.190] | 14 | 3 | 100 |
| hipporag_ppr | one_shot | 0.210 | 0.050 | 0.160 | 7.96e-04 | [0.080, 0.250] | 18 | 2 | 100 |
| hipporag_ppr | pathrag | 0.210 | 0.300 | -0.090 | 0.1374 | [-0.190, 0.020] | 10 | 19 | 100 |
| hipporag_ppr | raptor | 0.210 | 0.160 | 0.050 | 0.4237 | [-0.050, 0.150] | 15 | 10 | 100 |
| hipporag_ppr | reorg_amem | 0.210 | 0.170 | 0.040 | 0.5224 | [-0.050, 0.130] | 13 | 9 | 100 |
| hipporag_ppr | reorg_dreamcoder | 0.210 | 0.170 | 0.040 | 0.5403 | [-0.060, 0.130] | 14 | 10 | 100 |
| hipporag_ppr | reorg_evolver | 0.210 | 0.120 | 0.090 | 0.0809 | [0.000, 0.180] | 15 | 6 | 100 |
| hipporag_ppr | reorg_lilo | 0.210 | 0.140 | 0.070 | 0.2301 | [-0.030, 0.170] | 16 | 9 | 100 |
| hipporag_ppr | reorg_lrll | 0.210 | 0.130 | 0.080 | 0.1698 | [-0.020, 0.180] | 17 | 9 | 100 |
| hipporag_ppr | reorg_memp | 0.210 | 0.210 | 0.000 | 1.0000 | [-0.110, 0.110] | 16 | 16 | 100 |
| hipporag_ppr | reorg_memtree | 0.210 | 0.190 | 0.020 | 0.8501 | [-0.090, 0.120] | 15 | 13 | 100 |
| hipporag_ppr | reorg_off | 0.210 | 0.130 | 0.080 | 0.1175 | [-0.010, 0.170] | 14 | 6 | 100 |
| hipporag_ppr | reorg_on_graph_mdl_global_plateau | 0.210 | 0.120 | 0.090 | 0.0953 | [0.000, 0.180] | 16 | 7 | 100 |
| hipporag_ppr | reorg_on_trace_mdl_accretive_everyk | 0.210 | 0.110 | 0.100 | 0.0662 | [0.000, 0.200] | 17 | 7 | 100 |
| hipporag_ppr | reorg_sleepgate | 0.210 | 0.160 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| hipporag_ppr | reorg_stitch | 0.210 | 0.080 | 0.130 | 0.0059 | [0.050, 0.210] | 16 | 3 | 100 |
| hipporag_ppr | rrmc_multi_round | 0.210 | 0.120 | 0.090 | 0.0665 | [0.010, 0.170] | 14 | 5 | 100 |
| hipporag_ppr | uot_entropy | 0.210 | 0.180 | 0.030 | 0.7103 | [-0.080, 0.140] | 16 | 13 | 100 |
| hipporag_ppr | variant_cue_heavy | 0.210 | 0.220 | -0.010 | 1.0000 | [-0.120, 0.100] | 15 | 16 | 100 |
| hipporag_ppr | variant_dspy_opt | 0.210 | 0.030 | 0.180 | 1.44e-04 | [0.100, 0.260] | 19 | 1 | 100 |
| hipporag_ppr | variant_free_text | 0.210 | 0.100 | 0.110 | 0.0371 | [0.010, 0.210] | 17 | 6 | 100 |
| hipporag_ppr | variant_gepa | 0.210 | 0.050 | 0.160 | 0.0014 | [0.070, 0.250] | 19 | 3 | 100 |
| hipporag_ppr | variant_minimal | 0.210 | 0.140 | 0.070 | 0.1904 | [-0.020, 0.160] | 14 | 7 | 100 |
| hipporag_ppr | variant_parse | 0.210 | 0.050 | 0.160 | 0.0014 | [0.070, 0.250] | 19 | 3 | 100 |
| hipporag_ppr | variant_structured_routine | 0.210 | 0.100 | 0.110 | 0.0218 | [0.020, 0.200] | 15 | 4 | 100 |
| hipporag_ppr | variant_typed_only | 0.210 | 0.080 | 0.130 | 0.0036 | [0.060, 0.210] | 15 | 2 | 100 |
| hmem_hierarchical | lightrag | 0.150 | 0.300 | -0.150 | 0.0093 | [-0.250, -0.050] | 7 | 22 | 100 |
| hmem_hierarchical | magma_multigraph | 0.150 | 0.030 | 0.120 | 0.0033 | [0.050, 0.190] | 13 | 1 | 100 |
| hmem_hierarchical | mediq_policy | 0.150 | 0.100 | 0.050 | 0.3320 | [-0.040, 0.130] | 11 | 6 | 100 |
| hmem_hierarchical | one_shot | 0.150 | 0.050 | 0.100 | 0.0244 | [0.020, 0.180] | 13 | 3 | 100 |
| hmem_hierarchical | pathrag | 0.150 | 0.300 | -0.150 | 0.0093 | [-0.250, -0.050] | 7 | 22 | 100 |
| hmem_hierarchical | raptor | 0.150 | 0.160 | -0.010 | 1.0000 | [-0.100, 0.070] | 9 | 10 | 100 |
| hmem_hierarchical | reorg_amem | 0.150 | 0.170 | -0.020 | 0.8383 | [-0.120, 0.080] | 11 | 13 | 100 |
| hmem_hierarchical | reorg_dreamcoder | 0.150 | 0.170 | -0.020 | 0.8137 | [-0.100, 0.070] | 8 | 10 | 100 |
| hmem_hierarchical | reorg_evolver | 0.150 | 0.120 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| hmem_hierarchical | reorg_lilo | 0.150 | 0.140 | 0.010 | 1.0000 | [-0.070, 0.090] | 10 | 9 | 100 |
| hmem_hierarchical | reorg_lrll | 0.150 | 0.130 | 0.020 | 0.8312 | [-0.070, 0.110] | 12 | 10 | 100 |
| hmem_hierarchical | reorg_memp | 0.150 | 0.210 | -0.060 | 0.2636 | [-0.140, 0.030] | 7 | 13 | 100 |
| hmem_hierarchical | reorg_memtree | 0.150 | 0.190 | -0.040 | 0.5708 | [-0.150, 0.060] | 12 | 16 | 100 |
| hmem_hierarchical | reorg_off | 0.150 | 0.130 | 0.020 | 0.8137 | [-0.060, 0.110] | 10 | 8 | 100 |
| hmem_hierarchical | reorg_on_graph_mdl_global_plateau | 0.150 | 0.120 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| hmem_hierarchical | reorg_on_trace_mdl_accretive_everyk | 0.150 | 0.110 | 0.040 | 0.5023 | [-0.040, 0.120] | 12 | 8 | 100 |
| hmem_hierarchical | reorg_sleepgate | 0.150 | 0.160 | -0.010 | 1.0000 | [-0.110, 0.090] | 12 | 13 | 100 |
| hmem_hierarchical | reorg_stitch | 0.150 | 0.080 | 0.070 | 0.1213 | [-0.010, 0.140] | 11 | 4 | 100 |
| hmem_hierarchical | rrmc_multi_round | 0.150 | 0.120 | 0.030 | 0.6464 | [-0.060, 0.120] | 11 | 8 | 100 |
| hmem_hierarchical | uot_entropy | 0.150 | 0.180 | -0.030 | 0.7003 | [-0.130, 0.060] | 12 | 15 | 100 |
| hmem_hierarchical | variant_cue_heavy | 0.150 | 0.220 | -0.070 | 0.2652 | [-0.170, 0.030] | 11 | 18 | 100 |
| hmem_hierarchical | variant_dspy_opt | 0.150 | 0.030 | 0.120 | 0.0060 | [0.050, 0.200] | 14 | 2 | 100 |
| hmem_hierarchical | variant_free_text | 0.150 | 0.100 | 0.050 | 0.3017 | [-0.020, 0.120] | 10 | 5 | 100 |
| hmem_hierarchical | variant_gepa | 0.150 | 0.050 | 0.100 | 0.0339 | [0.020, 0.180] | 14 | 4 | 100 |
| hmem_hierarchical | variant_minimal | 0.150 | 0.140 | 0.010 | 1.0000 | [-0.080, 0.100] | 11 | 10 | 100 |
| hmem_hierarchical | variant_parse | 0.150 | 0.050 | 0.100 | 0.0162 | [0.030, 0.170] | 12 | 2 | 100 |
| hmem_hierarchical | variant_structured_routine | 0.150 | 0.100 | 0.050 | 0.3588 | [-0.030, 0.140] | 12 | 7 | 100 |
| hmem_hierarchical | variant_typed_only | 0.150 | 0.080 | 0.070 | 0.1213 | [-0.010, 0.150] | 11 | 4 | 100 |
| lightrag | magma_multigraph | 0.300 | 0.030 | 0.270 | 5.62e-07 | [0.180, 0.360] | 27 | 0 | 100 |
| lightrag | mediq_policy | 0.300 | 0.100 | 0.200 | 5.23e-04 | [0.090, 0.300] | 25 | 5 | 100 |
| lightrag | one_shot | 0.300 | 0.050 | 0.250 | 8.32e-06 | [0.160, 0.340] | 27 | 2 | 100 |
| lightrag | pathrag | 0.300 | 0.300 | 0.000 | 1.0000 | [-0.110, 0.110] | 15 | 15 | 100 |
| lightrag | raptor | 0.300 | 0.160 | 0.140 | 0.0108 | [0.050, 0.230] | 20 | 6 | 100 |
| lightrag | reorg_amem | 0.300 | 0.170 | 0.130 | 0.0164 | [0.030, 0.230] | 19 | 6 | 100 |
| lightrag | reorg_dreamcoder | 0.300 | 0.170 | 0.130 | 0.0209 | [0.030, 0.230] | 20 | 7 | 100 |
| lightrag | reorg_evolver | 0.300 | 0.120 | 0.180 | 0.0013 | [0.070, 0.280] | 23 | 5 | 100 |
| lightrag | reorg_lilo | 0.300 | 0.140 | 0.160 | 0.0080 | [0.050, 0.270] | 24 | 8 | 100 |
| lightrag | reorg_lrll | 0.300 | 0.130 | 0.170 | 0.0021 | [0.080, 0.260] | 22 | 5 | 100 |
| lightrag | reorg_memp | 0.300 | 0.210 | 0.090 | 0.1374 | [-0.010, 0.190] | 19 | 10 | 100 |
| lightrag | reorg_memtree | 0.300 | 0.190 | 0.110 | 0.0817 | [0.000, 0.220] | 22 | 11 | 100 |
| lightrag | reorg_off | 0.300 | 0.130 | 0.170 | 0.0030 | [0.060, 0.270] | 23 | 6 | 100 |
| lightrag | reorg_on_graph_mdl_global_plateau | 0.300 | 0.120 | 0.180 | 0.0013 | [0.080, 0.270] | 23 | 5 | 100 |
| lightrag | reorg_on_trace_mdl_accretive_everyk | 0.300 | 0.110 | 0.190 | 5.32e-04 | [0.100, 0.280] | 23 | 4 | 100 |
| lightrag | reorg_sleepgate | 0.300 | 0.160 | 0.140 | 0.0140 | [0.040, 0.240] | 21 | 7 | 100 |
| lightrag | reorg_stitch | 0.300 | 0.080 | 0.220 | 3.81e-05 | [0.130, 0.310] | 24 | 2 | 100 |
| lightrag | rrmc_multi_round | 0.300 | 0.120 | 0.180 | 0.0013 | [0.080, 0.270] | 23 | 5 | 100 |
| lightrag | uot_entropy | 0.300 | 0.180 | 0.120 | 0.0310 | [0.020, 0.220] | 19 | 7 | 100 |
| lightrag | variant_cue_heavy | 0.300 | 0.220 | 0.080 | 0.2299 | [-0.040, 0.190] | 21 | 13 | 100 |
| lightrag | variant_dspy_opt | 0.300 | 0.030 | 0.270 | 3.02e-06 | [0.170, 0.370] | 29 | 2 | 100 |
| lightrag | variant_free_text | 0.300 | 0.100 | 0.200 | 3.30e-04 | [0.100, 0.290] | 24 | 4 | 100 |
| lightrag | variant_gepa | 0.300 | 0.050 | 0.250 | 3.86e-06 | [0.160, 0.340] | 26 | 1 | 100 |
| lightrag | variant_minimal | 0.300 | 0.140 | 0.160 | 0.0046 | [0.060, 0.260] | 22 | 6 | 100 |
| lightrag | variant_parse | 0.300 | 0.050 | 0.250 | 8.32e-06 | [0.150, 0.340] | 27 | 2 | 100 |
| lightrag | variant_structured_routine | 0.300 | 0.100 | 0.200 | 5.10e-05 | [0.120, 0.290] | 21 | 1 | 100 |
| lightrag | variant_typed_only | 0.300 | 0.080 | 0.220 | 3.81e-05 | [0.130, 0.310] | 24 | 2 | 100 |
| magma_multigraph | mediq_policy | 0.030 | 0.100 | -0.070 | 0.0704 | [-0.140, -0.010] | 2 | 9 | 100 |
| magma_multigraph | one_shot | 0.030 | 0.050 | -0.020 | 0.7237 | [-0.070, 0.030] | 3 | 5 | 100 |
| magma_multigraph | pathrag | 0.030 | 0.300 | -0.270 | 1.38e-06 | [-0.360, -0.180] | 1 | 28 | 100 |
| magma_multigraph | raptor | 0.030 | 0.160 | -0.130 | 0.0019 | [-0.200, -0.070] | 1 | 14 | 100 |
| magma_multigraph | reorg_amem | 0.030 | 0.170 | -0.140 | 0.0022 | [-0.220, -0.060] | 2 | 16 | 100 |
| magma_multigraph | reorg_dreamcoder | 0.030 | 0.170 | -0.140 | 0.0012 | [-0.210, -0.070] | 1 | 15 | 100 |
| magma_multigraph | reorg_evolver | 0.030 | 0.120 | -0.090 | 0.0265 | [-0.160, -0.020] | 2 | 11 | 100 |
| magma_multigraph | reorg_lilo | 0.030 | 0.140 | -0.110 | 0.0153 | [-0.190, -0.030] | 3 | 14 | 100 |
| magma_multigraph | reorg_lrll | 0.030 | 0.130 | -0.100 | 0.0162 | [-0.180, -0.030] | 2 | 12 | 100 |
| magma_multigraph | reorg_memp | 0.030 | 0.210 | -0.180 | 1.44e-04 | [-0.260, -0.100] | 1 | 19 | 100 |
| magma_multigraph | reorg_memtree | 0.030 | 0.190 | -0.160 | 0.0014 | [-0.250, -0.080] | 3 | 19 | 100 |
| magma_multigraph | reorg_off | 0.030 | 0.130 | -0.100 | 0.0094 | [-0.170, -0.030] | 1 | 11 | 100 |
| magma_multigraph | reorg_on_graph_mdl_global_plateau | 0.030 | 0.120 | -0.090 | 0.0159 | [-0.160, -0.030] | 1 | 10 | 100 |
| magma_multigraph | reorg_on_trace_mdl_accretive_everyk | 0.030 | 0.110 | -0.080 | 0.0433 | [-0.150, -0.020] | 2 | 10 | 100 |
| magma_multigraph | reorg_sleepgate | 0.030 | 0.160 | -0.130 | 0.0036 | [-0.210, -0.050] | 2 | 15 | 100 |
| magma_multigraph | reorg_stitch | 0.030 | 0.080 | -0.050 | 0.1824 | [-0.110, 0.010] | 2 | 7 | 100 |
| magma_multigraph | rrmc_multi_round | 0.030 | 0.120 | -0.090 | 0.0159 | [-0.160, -0.030] | 1 | 10 | 100 |
| magma_multigraph | uot_entropy | 0.030 | 0.180 | -0.150 | 6.85e-04 | [-0.220, -0.080] | 1 | 16 | 100 |
| magma_multigraph | variant_cue_heavy | 0.030 | 0.220 | -0.190 | 1.75e-04 | [-0.280, -0.110] | 2 | 21 | 100 |
| magma_multigraph | variant_dspy_opt | 0.030 | 0.030 | 0.000 | 1.0000 | [-0.050, 0.050] | 3 | 3 | 100 |
| magma_multigraph | variant_free_text | 0.030 | 0.100 | -0.070 | 0.0704 | [-0.140, -0.010] | 2 | 9 | 100 |
| magma_multigraph | variant_gepa | 0.030 | 0.050 | -0.020 | 0.7237 | [-0.080, 0.030] | 3 | 5 | 100 |
| magma_multigraph | variant_minimal | 0.030 | 0.140 | -0.110 | 0.0098 | [-0.190, -0.040] | 2 | 13 | 100 |
| magma_multigraph | variant_parse | 0.030 | 0.050 | -0.020 | 0.6831 | [-0.070, 0.020] | 2 | 4 | 100 |
| magma_multigraph | variant_structured_routine | 0.030 | 0.100 | -0.070 | 0.0704 | [-0.140, -0.010] | 2 | 9 | 100 |
| magma_multigraph | variant_typed_only | 0.030 | 0.080 | -0.050 | 0.1824 | [-0.120, 0.010] | 2 | 7 | 100 |
| pathrag | raptor | 0.300 | 0.160 | 0.140 | 0.0216 | [0.040, 0.240] | 23 | 9 | 100 |
| pathrag | reorg_amem | 0.300 | 0.170 | 0.130 | 0.0547 | [0.010, 0.250] | 26 | 13 | 100 |
| pathrag | reorg_dreamcoder | 0.300 | 0.170 | 0.130 | 0.0311 | [0.030, 0.230] | 22 | 9 | 100 |
| pathrag | reorg_evolver | 0.300 | 0.120 | 0.180 | 8.56e-04 | [0.090, 0.270] | 22 | 4 | 100 |
| pathrag | reorg_lilo | 0.300 | 0.140 | 0.160 | 0.0080 | [0.050, 0.270] | 24 | 8 | 100 |
| pathrag | reorg_lrll | 0.300 | 0.130 | 0.170 | 0.0053 | [0.060, 0.290] | 25 | 8 | 100 |
| pathrag | reorg_memp | 0.300 | 0.210 | 0.090 | 0.1508 | [-0.020, 0.200] | 20 | 11 | 100 |
| pathrag | reorg_memtree | 0.300 | 0.190 | 0.110 | 0.0725 | [0.000, 0.220] | 21 | 10 | 100 |
| pathrag | reorg_off | 0.300 | 0.130 | 0.170 | 0.0053 | [0.070, 0.280] | 25 | 8 | 100 |
| pathrag | reorg_on_graph_mdl_global_plateau | 0.300 | 0.120 | 0.180 | 8.56e-04 | [0.090, 0.270] | 22 | 4 | 100 |
| pathrag | reorg_on_trace_mdl_accretive_everyk | 0.300 | 0.110 | 0.190 | 0.0012 | [0.090, 0.290] | 25 | 6 | 100 |
| pathrag | reorg_sleepgate | 0.300 | 0.160 | 0.140 | 0.0258 | [0.020, 0.250] | 24 | 10 | 100 |
| pathrag | reorg_stitch | 0.300 | 0.080 | 0.220 | 1.81e-05 | [0.140, 0.310] | 23 | 1 | 100 |
| pathrag | rrmc_multi_round | 0.300 | 0.120 | 0.180 | 0.0027 | [0.070, 0.280] | 25 | 7 | 100 |
| pathrag | uot_entropy | 0.300 | 0.180 | 0.120 | 0.0446 | [0.020, 0.230] | 21 | 9 | 100 |
| pathrag | variant_cue_heavy | 0.300 | 0.220 | 0.080 | 0.1698 | [-0.020, 0.190] | 17 | 9 | 100 |
| pathrag | variant_dspy_opt | 0.300 | 0.030 | 0.270 | 6.01e-06 | [0.170, 0.370] | 30 | 3 | 100 |
| pathrag | variant_free_text | 0.300 | 0.100 | 0.200 | 5.23e-04 | [0.100, 0.300] | 25 | 5 | 100 |
| pathrag | variant_gepa | 0.300 | 0.050 | 0.250 | 1.63e-05 | [0.150, 0.350] | 28 | 3 | 100 |
| pathrag | variant_minimal | 0.300 | 0.140 | 0.160 | 0.0062 | [0.050, 0.260] | 23 | 7 | 100 |
| pathrag | variant_parse | 0.300 | 0.050 | 0.250 | 8.32e-06 | [0.160, 0.350] | 27 | 2 | 100 |
| pathrag | variant_structured_routine | 0.300 | 0.100 | 0.200 | 3.30e-04 | [0.110, 0.290] | 24 | 4 | 100 |
| pathrag | variant_typed_only | 0.300 | 0.080 | 0.220 | 7.23e-05 | [0.120, 0.320] | 25 | 3 | 100 |
| raptor | reorg_amem | 0.160 | 0.170 | -0.010 | 1.0000 | [-0.100, 0.080] | 9 | 10 | 100 |
| raptor | reorg_dreamcoder | 0.160 | 0.170 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| raptor | reorg_evolver | 0.160 | 0.120 | 0.040 | 0.5023 | [-0.040, 0.130] | 12 | 8 | 100 |
| raptor | reorg_lilo | 0.160 | 0.140 | 0.020 | 0.8231 | [-0.060, 0.110] | 11 | 9 | 100 |
| raptor | reorg_lrll | 0.160 | 0.130 | 0.030 | 0.6056 | [-0.050, 0.110] | 9 | 6 | 100 |
| raptor | reorg_memp | 0.160 | 0.210 | -0.050 | 0.4042 | [-0.140, 0.040] | 9 | 14 | 100 |
| raptor | reorg_memtree | 0.160 | 0.190 | -0.030 | 0.7003 | [-0.130, 0.070] | 12 | 15 | 100 |
| raptor | reorg_off | 0.160 | 0.130 | 0.030 | 0.6767 | [-0.060, 0.120] | 13 | 10 | 100 |
| raptor | reorg_on_graph_mdl_global_plateau | 0.160 | 0.120 | 0.040 | 0.4227 | [-0.030, 0.120] | 9 | 5 | 100 |
| raptor | reorg_on_trace_mdl_accretive_everyk | 0.160 | 0.110 | 0.050 | 0.3588 | [-0.030, 0.130] | 12 | 7 | 100 |
| raptor | reorg_sleepgate | 0.160 | 0.160 | 0.000 | 1.0000 | [-0.090, 0.090] | 11 | 11 | 100 |
| raptor | reorg_stitch | 0.160 | 0.080 | 0.080 | 0.0614 | [0.010, 0.160] | 11 | 3 | 100 |
| raptor | rrmc_multi_round | 0.160 | 0.120 | 0.040 | 0.5224 | [-0.050, 0.130] | 13 | 9 | 100 |
| raptor | uot_entropy | 0.160 | 0.180 | -0.020 | 0.8501 | [-0.110, 0.080] | 13 | 15 | 100 |
| raptor | variant_cue_heavy | 0.160 | 0.220 | -0.060 | 0.3074 | [-0.160, 0.040] | 9 | 15 | 100 |
| raptor | variant_dspy_opt | 0.160 | 0.030 | 0.130 | 0.0036 | [0.060, 0.210] | 15 | 2 | 100 |
| raptor | variant_free_text | 0.160 | 0.100 | 0.060 | 0.2636 | [-0.030, 0.150] | 13 | 7 | 100 |
| raptor | variant_gepa | 0.160 | 0.050 | 0.110 | 0.0218 | [0.030, 0.190] | 15 | 4 | 100 |
| raptor | variant_minimal | 0.160 | 0.140 | 0.020 | 0.8383 | [-0.070, 0.120] | 13 | 11 | 100 |
| raptor | variant_parse | 0.160 | 0.050 | 0.110 | 0.0153 | [0.040, 0.190] | 14 | 3 | 100 |
| raptor | variant_structured_routine | 0.160 | 0.100 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| raptor | variant_typed_only | 0.160 | 0.080 | 0.080 | 0.0990 | [0.000, 0.160] | 13 | 5 | 100 |
| accretive_prune | adas_style_search | 0.260 | 0.130 | 0.130 | 0.0311 | [0.020, 0.230] | 22 | 9 | 100 |
| accretive_prune | alma_style_metaedit | 0.260 | 0.120 | 0.140 | 0.0108 | [0.040, 0.240] | 20 | 6 | 100 |
| accretive_prune | arcmemo_oe | 0.260 | 0.170 | 0.090 | 0.1237 | [-0.010, 0.190] | 18 | 9 | 100 |
| accretive_prune | arcmemo_ps | 0.260 | 0.110 | 0.150 | 0.0119 | [0.040, 0.250] | 23 | 8 | 100 |
| accretive_prune | barc_seeded | 0.260 | 0.090 | 0.170 | 0.0021 | [0.070, 0.270] | 22 | 5 | 100 |
| accretive_prune | barc_synthetic | 0.260 | 0.030 | 0.230 | 1.08e-05 | [0.150, 0.310] | 24 | 1 | 100 |
| accretive_prune | colbert_rerank | 0.260 | 0.180 | 0.080 | 0.2433 | [-0.040, 0.200] | 22 | 14 | 100 |
| accretive_prune | corpus_hipporag_init | 0.260 | 0.060 | 0.200 | 1.94e-04 | [0.110, 0.290] | 23 | 3 | 100 |
| accretive_prune | empty_start | 0.260 | 0.110 | 0.150 | 0.0071 | [0.050, 0.250] | 21 | 6 | 100 |
| accretive_prune | flat_topk | 0.260 | 0.240 | 0.020 | 0.8551 | [-0.090, 0.130] | 16 | 14 | 100 |
| accretive_prune | graph_traversal | 0.260 | 0.200 | 0.060 | 0.3768 | [-0.060, 0.170] | 19 | 13 | 100 |
| accretive_prune | graphrag | 0.260 | 0.300 | -0.040 | 0.5839 | [-0.160, 0.070] | 13 | 17 | 100 |
| accretive_prune | hand_coded_reorg | 0.260 | 0.120 | 0.140 | 0.0216 | [0.030, 0.240] | 23 | 9 | 100 |
| accretive_prune | hipporag2_filter | 0.260 | 0.100 | 0.160 | 0.0046 | [0.060, 0.260] | 22 | 6 | 100 |
| accretive_prune | hipporag_ppr | 0.260 | 0.210 | 0.050 | 0.4576 | [-0.060, 0.160] | 17 | 12 | 100 |
| accretive_prune | hmem_hierarchical | 0.260 | 0.150 | 0.110 | 0.0543 | [0.020, 0.200] | 19 | 8 | 100 |
| accretive_prune | lightrag | 0.260 | 0.300 | -0.040 | 0.6171 | [-0.160, 0.080] | 16 | 20 | 100 |
| accretive_prune | magma_multigraph | 0.260 | 0.030 | 0.230 | 1.08e-05 | [0.140, 0.320] | 24 | 1 | 100 |
| accretive_prune | mediq_policy | 0.260 | 0.100 | 0.160 | 0.0046 | [0.060, 0.260] | 22 | 6 | 100 |
| accretive_prune | one_shot | 0.260 | 0.050 | 0.210 | 1.19e-04 | [0.120, 0.300] | 24 | 3 | 100 |
| accretive_prune | pathrag | 0.260 | 0.300 | -0.040 | 0.6069 | [-0.150, 0.070] | 15 | 19 | 100 |
| accretive_prune | raptor | 0.260 | 0.160 | 0.100 | 0.1227 | [-0.010, 0.210] | 22 | 12 | 100 |
| accretive_prune | reorg_amem | 0.260 | 0.170 | 0.090 | 0.1374 | [-0.020, 0.190] | 19 | 10 | 100 |
| accretive_prune | reorg_dreamcoder | 0.260 | 0.170 | 0.090 | 0.1637 | [-0.020, 0.200] | 21 | 12 | 100 |
| accretive_prune | reorg_evolver | 0.260 | 0.120 | 0.140 | 0.0080 | [0.050, 0.230] | 19 | 5 | 100 |
| accretive_prune | reorg_lilo | 0.260 | 0.140 | 0.120 | 0.0446 | [0.010, 0.220] | 21 | 9 | 100 |
| accretive_prune | reorg_lrll | 0.260 | 0.130 | 0.130 | 0.0123 | [0.030, 0.220] | 18 | 5 | 100 |
| accretive_prune | reorg_memp | 0.260 | 0.210 | 0.050 | 0.4862 | [-0.060, 0.160] | 19 | 14 | 100 |
| accretive_prune | reorg_memtree | 0.260 | 0.190 | 0.070 | 0.2963 | [-0.040, 0.190] | 20 | 13 | 100 |
| accretive_prune | reorg_off | 0.260 | 0.130 | 0.130 | 0.0259 | [0.030, 0.230] | 21 | 8 | 100 |
| accretive_prune | reorg_on_graph_mdl_global_plateau | 0.260 | 0.120 | 0.140 | 0.0176 | [0.040, 0.240] | 22 | 8 | 100 |
| accretive_prune | reorg_on_trace_mdl_accretive_everyk | 0.260 | 0.110 | 0.150 | 0.0093 | [0.050, 0.260] | 22 | 7 | 100 |
| accretive_prune | reorg_sleepgate | 0.260 | 0.160 | 0.100 | 0.1003 | [-0.010, 0.210] | 20 | 10 | 100 |
| accretive_prune | reorg_stitch | 0.260 | 0.080 | 0.180 | 0.0013 | [0.080, 0.280] | 23 | 5 | 100 |
| accretive_prune | rrmc_multi_round | 0.260 | 0.120 | 0.140 | 0.0140 | [0.040, 0.240] | 21 | 7 | 100 |
| accretive_prune | uot_entropy | 0.260 | 0.180 | 0.080 | 0.2299 | [-0.040, 0.190] | 21 | 13 | 100 |
| accretive_prune | variant_cue_heavy | 0.260 | 0.220 | 0.040 | 0.6171 | [-0.080, 0.150] | 20 | 16 | 100 |
| accretive_prune | variant_dspy_opt | 0.260 | 0.030 | 0.230 | 2.30e-05 | [0.140, 0.320] | 25 | 2 | 100 |
| accretive_prune | variant_free_text | 0.260 | 0.100 | 0.160 | 0.0062 | [0.060, 0.260] | 23 | 7 | 100 |
| accretive_prune | variant_gepa | 0.260 | 0.050 | 0.210 | 2.04e-04 | [0.110, 0.300] | 25 | 4 | 100 |
| accretive_prune | variant_minimal | 0.260 | 0.140 | 0.120 | 0.0446 | [0.020, 0.230] | 21 | 9 | 100 |
| accretive_prune | variant_parse | 0.260 | 0.050 | 0.210 | 2.04e-04 | [0.120, 0.300] | 25 | 4 | 100 |
| accretive_prune | variant_structured_routine | 0.260 | 0.100 | 0.160 | 0.0033 | [0.060, 0.250] | 21 | 5 | 100 |
| accretive_prune | variant_typed_only | 0.260 | 0.080 | 0.180 | 8.56e-04 | [0.080, 0.280] | 22 | 4 | 100 |
| reorg_amem | reorg_dreamcoder | 0.170 | 0.170 | 0.000 | 1.0000 | [-0.090, 0.090] | 11 | 11 | 100 |
| reorg_amem | reorg_evolver | 0.170 | 0.120 | 0.050 | 0.3588 | [-0.040, 0.140] | 12 | 7 | 100 |
| reorg_amem | reorg_lilo | 0.170 | 0.140 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| reorg_amem | reorg_lrll | 0.170 | 0.130 | 0.040 | 0.5023 | [-0.050, 0.120] | 12 | 8 | 100 |
| reorg_amem | reorg_memp | 0.170 | 0.210 | -0.040 | 0.5563 | [-0.140, 0.060] | 11 | 15 | 100 |
| reorg_amem | reorg_memtree | 0.170 | 0.190 | -0.020 | 0.8383 | [-0.110, 0.080] | 11 | 13 | 100 |
| reorg_amem | reorg_off | 0.170 | 0.130 | 0.040 | 0.5224 | [-0.050, 0.130] | 13 | 9 | 100 |
| reorg_amem | reorg_on_graph_mdl_global_plateau | 0.170 | 0.120 | 0.050 | 0.3827 | [-0.030, 0.130] | 13 | 8 | 100 |
| reorg_amem | reorg_on_trace_mdl_accretive_everyk | 0.170 | 0.110 | 0.060 | 0.2864 | [-0.030, 0.150] | 14 | 8 | 100 |
| reorg_amem | reorg_sleepgate | 0.170 | 0.160 | 0.010 | 1.0000 | [-0.080, 0.110] | 11 | 10 | 100 |
| reorg_amem | reorg_stitch | 0.170 | 0.080 | 0.090 | 0.0665 | [0.000, 0.170] | 14 | 5 | 100 |
| reorg_amem | rrmc_multi_round | 0.170 | 0.120 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| reorg_amem | uot_entropy | 0.170 | 0.180 | -0.010 | 1.0000 | [-0.100, 0.090] | 11 | 12 | 100 |
| reorg_amem | variant_cue_heavy | 0.170 | 0.220 | -0.050 | 0.4414 | [-0.150, 0.050] | 11 | 16 | 100 |
| reorg_amem | variant_dspy_opt | 0.170 | 0.030 | 0.140 | 0.0022 | [0.060, 0.220] | 16 | 2 | 100 |
| reorg_amem | variant_free_text | 0.170 | 0.100 | 0.070 | 0.1687 | [-0.010, 0.160] | 13 | 6 | 100 |
| reorg_amem | variant_gepa | 0.170 | 0.050 | 0.120 | 0.0060 | [0.040, 0.200] | 14 | 2 | 100 |
| reorg_amem | variant_minimal | 0.170 | 0.140 | 0.030 | 0.6767 | [-0.060, 0.120] | 13 | 10 | 100 |
| reorg_amem | variant_parse | 0.170 | 0.050 | 0.120 | 0.0095 | [0.040, 0.200] | 15 | 3 | 100 |
| reorg_amem | variant_structured_routine | 0.170 | 0.100 | 0.070 | 0.1213 | [0.000, 0.140] | 11 | 4 | 100 |
| reorg_amem | variant_typed_only | 0.170 | 0.080 | 0.090 | 0.0665 | [0.010, 0.180] | 14 | 5 | 100 |
| reorg_dreamcoder | reorg_evolver | 0.170 | 0.120 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| reorg_dreamcoder | reorg_lilo | 0.170 | 0.140 | 0.030 | 0.6767 | [-0.060, 0.120] | 13 | 10 | 100 |
| reorg_dreamcoder | reorg_lrll | 0.170 | 0.130 | 0.040 | 0.5023 | [-0.050, 0.130] | 12 | 8 | 100 |
| reorg_dreamcoder | reorg_memp | 0.170 | 0.210 | -0.040 | 0.5224 | [-0.130, 0.050] | 9 | 13 | 100 |
| reorg_dreamcoder | reorg_memtree | 0.170 | 0.190 | -0.020 | 0.8312 | [-0.120, 0.080] | 10 | 12 | 100 |
| reorg_dreamcoder | reorg_off | 0.170 | 0.130 | 0.040 | 0.4795 | [-0.040, 0.130] | 11 | 7 | 100 |
| reorg_dreamcoder | reorg_on_graph_mdl_global_plateau | 0.170 | 0.120 | 0.050 | 0.3320 | [-0.030, 0.130] | 11 | 6 | 100 |
| reorg_dreamcoder | reorg_on_trace_mdl_accretive_everyk | 0.170 | 0.110 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| reorg_dreamcoder | reorg_sleepgate | 0.170 | 0.160 | 0.010 | 1.0000 | [-0.080, 0.100] | 11 | 10 | 100 |
| reorg_dreamcoder | reorg_stitch | 0.170 | 0.080 | 0.090 | 0.0389 | [0.020, 0.170] | 12 | 3 | 100 |
| reorg_dreamcoder | rrmc_multi_round | 0.170 | 0.120 | 0.050 | 0.3017 | [-0.030, 0.130] | 10 | 5 | 100 |
| reorg_dreamcoder | uot_entropy | 0.170 | 0.180 | -0.010 | 1.0000 | [-0.100, 0.080] | 11 | 12 | 100 |
| reorg_dreamcoder | variant_cue_heavy | 0.170 | 0.220 | -0.050 | 0.4414 | [-0.150, 0.050] | 11 | 16 | 100 |
| reorg_dreamcoder | variant_dspy_opt | 0.170 | 0.030 | 0.140 | 0.0022 | [0.060, 0.210] | 16 | 2 | 100 |
| reorg_dreamcoder | variant_free_text | 0.170 | 0.100 | 0.070 | 0.1904 | [-0.020, 0.160] | 14 | 7 | 100 |
| reorg_dreamcoder | variant_gepa | 0.170 | 0.050 | 0.120 | 0.0060 | [0.050, 0.200] | 14 | 2 | 100 |
| reorg_dreamcoder | variant_minimal | 0.170 | 0.140 | 0.030 | 0.6767 | [-0.060, 0.130] | 13 | 10 | 100 |
| reorg_dreamcoder | variant_parse | 0.170 | 0.050 | 0.120 | 0.0095 | [0.030, 0.200] | 15 | 3 | 100 |
| reorg_dreamcoder | variant_structured_routine | 0.170 | 0.100 | 0.070 | 0.1456 | [-0.010, 0.150] | 12 | 5 | 100 |
| reorg_dreamcoder | variant_typed_only | 0.170 | 0.080 | 0.090 | 0.0523 | [0.010, 0.180] | 13 | 4 | 100 |
| reorg_evolver | reorg_lilo | 0.120 | 0.140 | -0.020 | 0.8137 | [-0.100, 0.060] | 8 | 10 | 100 |
| reorg_evolver | reorg_lrll | 0.120 | 0.130 | -0.010 | 1.0000 | [-0.100, 0.080] | 9 | 10 | 100 |
| reorg_evolver | reorg_memp | 0.120 | 0.210 | -0.090 | 0.0953 | [-0.180, 0.000] | 7 | 16 | 100 |
| reorg_evolver | reorg_memtree | 0.120 | 0.190 | -0.070 | 0.2301 | [-0.170, 0.030] | 9 | 16 | 100 |
| reorg_evolver | reorg_off | 0.120 | 0.130 | -0.010 | 1.0000 | [-0.100, 0.080] | 10 | 11 | 100 |
| reorg_evolver | reorg_on_graph_mdl_global_plateau | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.070, 0.080] | 8 | 8 | 100 |
| reorg_evolver | reorg_on_trace_mdl_accretive_everyk | 0.120 | 0.110 | 0.010 | 1.0000 | [-0.070, 0.090] | 8 | 7 | 100 |
| reorg_evolver | reorg_sleepgate | 0.120 | 0.160 | -0.040 | 0.5403 | [-0.130, 0.060] | 10 | 14 | 100 |
| reorg_evolver | reorg_stitch | 0.120 | 0.080 | 0.040 | 0.3865 | [-0.030, 0.110] | 8 | 4 | 100 |
| reorg_evolver | rrmc_multi_round | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.070, 0.080] | 7 | 7 | 100 |
| reorg_evolver | uot_entropy | 0.120 | 0.180 | -0.060 | 0.3268 | [-0.150, 0.040] | 10 | 16 | 100 |
| reorg_evolver | variant_cue_heavy | 0.120 | 0.220 | -0.100 | 0.0550 | [-0.190, -0.010] | 6 | 16 | 100 |
| reorg_evolver | variant_dspy_opt | 0.120 | 0.030 | 0.090 | 0.0389 | [0.020, 0.160] | 12 | 3 | 100 |
| reorg_evolver | variant_free_text | 0.120 | 0.100 | 0.020 | 0.8026 | [-0.060, 0.100] | 9 | 7 | 100 |
| reorg_evolver | variant_gepa | 0.120 | 0.050 | 0.070 | 0.1213 | [0.000, 0.140] | 11 | 4 | 100 |
| reorg_evolver | variant_minimal | 0.120 | 0.140 | -0.020 | 0.8026 | [-0.090, 0.060] | 7 | 9 | 100 |
| reorg_evolver | variant_parse | 0.120 | 0.050 | 0.070 | 0.1213 | [-0.010, 0.140] | 11 | 4 | 100 |
| reorg_evolver | variant_structured_routine | 0.120 | 0.100 | 0.020 | 0.7728 | [-0.040, 0.090] | 7 | 5 | 100 |
| reorg_evolver | variant_typed_only | 0.120 | 0.080 | 0.040 | 0.3865 | [-0.030, 0.110] | 8 | 4 | 100 |
| reorg_lilo | reorg_lrll | 0.140 | 0.130 | 0.010 | 1.0000 | [-0.070, 0.100] | 10 | 9 | 100 |
| reorg_lilo | reorg_memp | 0.140 | 0.210 | -0.070 | 0.1687 | [-0.150, 0.010] | 6 | 13 | 100 |
| reorg_lilo | reorg_memtree | 0.140 | 0.190 | -0.050 | 0.3827 | [-0.140, 0.040] | 8 | 13 | 100 |
| reorg_lilo | reorg_off | 0.140 | 0.130 | 0.010 | 1.0000 | [-0.080, 0.100] | 12 | 11 | 100 |
| reorg_lilo | reorg_on_graph_mdl_global_plateau | 0.140 | 0.120 | 0.020 | 0.8137 | [-0.060, 0.100] | 10 | 8 | 100 |
| reorg_lilo | reorg_on_trace_mdl_accretive_everyk | 0.140 | 0.110 | 0.030 | 0.6464 | [-0.050, 0.120] | 11 | 8 | 100 |
| reorg_lilo | reorg_sleepgate | 0.140 | 0.160 | -0.020 | 0.8137 | [-0.100, 0.060] | 8 | 10 | 100 |
| reorg_lilo | reorg_stitch | 0.140 | 0.080 | 0.060 | 0.1814 | [-0.010, 0.140] | 10 | 4 | 100 |
| reorg_lilo | rrmc_multi_round | 0.140 | 0.120 | 0.020 | 0.8137 | [-0.060, 0.110] | 10 | 8 | 100 |
| reorg_lilo | uot_entropy | 0.140 | 0.180 | -0.040 | 0.5563 | [-0.140, 0.050] | 11 | 15 | 100 |
| reorg_lilo | variant_cue_heavy | 0.140 | 0.220 | -0.080 | 0.1530 | [-0.170, 0.010] | 8 | 16 | 100 |
| reorg_lilo | variant_dspy_opt | 0.140 | 0.030 | 0.110 | 0.0153 | [0.030, 0.190] | 14 | 3 | 100 |
| reorg_lilo | variant_free_text | 0.140 | 0.100 | 0.040 | 0.4795 | [-0.040, 0.120] | 11 | 7 | 100 |
| reorg_lilo | variant_gepa | 0.140 | 0.050 | 0.090 | 0.0665 | [0.010, 0.180] | 14 | 5 | 100 |
| reorg_lilo | variant_minimal | 0.140 | 0.140 | 0.000 | 1.0000 | [-0.090, 0.090] | 11 | 11 | 100 |
| reorg_lilo | variant_parse | 0.140 | 0.050 | 0.090 | 0.0523 | [0.010, 0.170] | 13 | 4 | 100 |
| reorg_lilo | variant_structured_routine | 0.140 | 0.100 | 0.040 | 0.4533 | [-0.040, 0.120] | 10 | 6 | 100 |
| reorg_lilo | variant_typed_only | 0.140 | 0.080 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| reorg_lrll | reorg_memp | 0.130 | 0.210 | -0.080 | 0.1356 | [-0.170, 0.010] | 7 | 15 | 100 |
| reorg_lrll | reorg_memtree | 0.130 | 0.190 | -0.060 | 0.3074 | [-0.160, 0.030] | 9 | 15 | 100 |
| reorg_lrll | reorg_off | 0.130 | 0.130 | 0.000 | 1.0000 | [-0.080, 0.090] | 10 | 10 | 100 |
| reorg_lrll | reorg_on_graph_mdl_global_plateau | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.070, 0.090] | 9 | 8 | 100 |
| reorg_lrll | reorg_on_trace_mdl_accretive_everyk | 0.130 | 0.110 | 0.020 | 0.7893 | [-0.050, 0.100] | 8 | 6 | 100 |
| reorg_lrll | reorg_sleepgate | 0.130 | 0.160 | -0.030 | 0.6767 | [-0.120, 0.070] | 10 | 13 | 100 |
| reorg_lrll | reorg_stitch | 0.130 | 0.080 | 0.050 | 0.2673 | [-0.020, 0.120] | 9 | 4 | 100 |
| reorg_lrll | rrmc_multi_round | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.070, 0.090] | 9 | 8 | 100 |
| reorg_lrll | uot_entropy | 0.130 | 0.180 | -0.050 | 0.3588 | [-0.130, 0.040] | 7 | 12 | 100 |
| reorg_lrll | variant_cue_heavy | 0.130 | 0.220 | -0.090 | 0.1237 | [-0.190, 0.010] | 9 | 18 | 100 |
| reorg_lrll | variant_dspy_opt | 0.130 | 0.030 | 0.100 | 0.0244 | [0.020, 0.180] | 13 | 3 | 100 |
| reorg_lrll | variant_free_text | 0.130 | 0.100 | 0.030 | 0.6056 | [-0.040, 0.100] | 9 | 6 | 100 |
| reorg_lrll | variant_gepa | 0.130 | 0.050 | 0.080 | 0.0614 | [0.010, 0.150] | 11 | 3 | 100 |
| reorg_lrll | variant_minimal | 0.130 | 0.140 | -0.010 | 1.0000 | [-0.090, 0.070] | 9 | 10 | 100 |
| reorg_lrll | variant_parse | 0.130 | 0.050 | 0.080 | 0.0801 | [0.010, 0.160] | 12 | 4 | 100 |
| reorg_lrll | variant_structured_routine | 0.130 | 0.100 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| reorg_lrll | variant_typed_only | 0.130 | 0.080 | 0.050 | 0.2278 | [-0.010, 0.120] | 8 | 3 | 100 |
| reorg_memp | reorg_memtree | 0.210 | 0.190 | 0.020 | 0.8445 | [-0.090, 0.120] | 14 | 12 | 100 |
| reorg_memp | reorg_off | 0.210 | 0.130 | 0.080 | 0.1530 | [-0.020, 0.170] | 16 | 8 | 100 |
| reorg_memp | reorg_on_graph_mdl_global_plateau | 0.210 | 0.120 | 0.090 | 0.0809 | [0.000, 0.180] | 15 | 6 | 100 |
| reorg_memp | reorg_on_trace_mdl_accretive_everyk | 0.210 | 0.110 | 0.100 | 0.0442 | [0.010, 0.190] | 15 | 5 | 100 |
| reorg_memp | reorg_sleepgate | 0.210 | 0.160 | 0.050 | 0.3588 | [-0.040, 0.140] | 12 | 7 | 100 |
| reorg_memp | reorg_stitch | 0.210 | 0.080 | 0.130 | 0.0123 | [0.040, 0.220] | 18 | 5 | 100 |
| reorg_memp | rrmc_multi_round | 0.210 | 0.120 | 0.090 | 0.0809 | [0.000, 0.180] | 15 | 6 | 100 |
| reorg_memp | uot_entropy | 0.210 | 0.180 | 0.030 | 0.6892 | [-0.070, 0.130] | 14 | 11 | 100 |
| reorg_memp | variant_cue_heavy | 0.210 | 0.220 | -0.010 | 1.0000 | [-0.110, 0.080] | 13 | 14 | 100 |
| reorg_memp | variant_dspy_opt | 0.210 | 0.030 | 0.180 | 5.20e-04 | [0.090, 0.260] | 21 | 3 | 100 |
| reorg_memp | variant_free_text | 0.210 | 0.100 | 0.110 | 0.0218 | [0.030, 0.190] | 15 | 4 | 100 |
| reorg_memp | variant_gepa | 0.210 | 0.050 | 0.160 | 0.0022 | [0.070, 0.250] | 20 | 4 | 100 |
| reorg_memp | variant_minimal | 0.210 | 0.140 | 0.070 | 0.2482 | [-0.030, 0.170] | 17 | 10 | 100 |
| reorg_memp | variant_parse | 0.210 | 0.050 | 0.160 | 0.0014 | [0.070, 0.250] | 19 | 3 | 100 |
| reorg_memp | variant_structured_routine | 0.210 | 0.100 | 0.110 | 0.0291 | [0.020, 0.200] | 16 | 5 | 100 |
| reorg_memp | variant_typed_only | 0.210 | 0.080 | 0.130 | 0.0123 | [0.030, 0.220] | 18 | 5 | 100 |
| reorg_memtree | reorg_off | 0.190 | 0.130 | 0.060 | 0.3074 | [-0.030, 0.160] | 15 | 9 | 100 |
| reorg_memtree | reorg_on_graph_mdl_global_plateau | 0.190 | 0.120 | 0.070 | 0.1213 | [0.000, 0.150] | 11 | 4 | 100 |
| reorg_memtree | reorg_on_trace_mdl_accretive_everyk | 0.190 | 0.110 | 0.080 | 0.1175 | [-0.010, 0.170] | 14 | 6 | 100 |
| reorg_memtree | reorg_sleepgate | 0.190 | 0.160 | 0.030 | 0.6464 | [-0.060, 0.120] | 11 | 8 | 100 |
| reorg_memtree | reorg_stitch | 0.190 | 0.080 | 0.110 | 0.0291 | [0.020, 0.200] | 16 | 5 | 100 |
| reorg_memtree | rrmc_multi_round | 0.190 | 0.120 | 0.070 | 0.1687 | [-0.010, 0.150] | 13 | 6 | 100 |
| reorg_memtree | uot_entropy | 0.190 | 0.180 | 0.010 | 1.0000 | [-0.070, 0.100] | 12 | 11 | 100 |
| reorg_memtree | variant_cue_heavy | 0.190 | 0.220 | -0.030 | 0.7003 | [-0.140, 0.080] | 12 | 15 | 100 |
| reorg_memtree | variant_dspy_opt | 0.190 | 0.030 | 0.160 | 4.07e-04 | [0.090, 0.240] | 17 | 1 | 100 |
| reorg_memtree | variant_free_text | 0.190 | 0.100 | 0.090 | 0.0665 | [0.010, 0.180] | 14 | 5 | 100 |
| reorg_memtree | variant_gepa | 0.190 | 0.050 | 0.140 | 0.0012 | [0.070, 0.220] | 15 | 1 | 100 |
| reorg_memtree | variant_minimal | 0.190 | 0.140 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| reorg_memtree | variant_parse | 0.190 | 0.050 | 0.140 | 0.0022 | [0.060, 0.230] | 16 | 2 | 100 |
| reorg_memtree | variant_structured_routine | 0.190 | 0.100 | 0.090 | 0.0809 | [0.000, 0.180] | 15 | 6 | 100 |
| reorg_memtree | variant_typed_only | 0.190 | 0.080 | 0.110 | 0.0218 | [0.020, 0.190] | 15 | 4 | 100 |
| reorg_off | reorg_on_graph_mdl_global_plateau | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.070, 0.090] | 9 | 8 | 100 |
| reorg_off | reorg_on_trace_mdl_accretive_everyk | 0.130 | 0.110 | 0.020 | 0.8137 | [-0.060, 0.100] | 10 | 8 | 100 |
| reorg_off | reorg_sleepgate | 0.130 | 0.160 | -0.030 | 0.6056 | [-0.110, 0.040] | 6 | 9 | 100 |
| reorg_off | reorg_stitch | 0.130 | 0.080 | 0.050 | 0.3588 | [-0.040, 0.140] | 12 | 7 | 100 |
| reorg_off | rrmc_multi_round | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.070, 0.080] | 8 | 7 | 100 |
| reorg_off | uot_entropy | 0.130 | 0.180 | -0.050 | 0.3588 | [-0.140, 0.020] | 7 | 12 | 100 |
| reorg_off | variant_cue_heavy | 0.130 | 0.220 | -0.090 | 0.1237 | [-0.190, 0.010] | 9 | 18 | 100 |
| reorg_off | variant_dspy_opt | 0.130 | 0.030 | 0.100 | 0.0162 | [0.030, 0.170] | 12 | 2 | 100 |
| reorg_off | variant_free_text | 0.130 | 0.100 | 0.030 | 0.6056 | [-0.050, 0.110] | 9 | 6 | 100 |
| reorg_off | variant_gepa | 0.130 | 0.050 | 0.080 | 0.0614 | [0.010, 0.150] | 11 | 3 | 100 |
| reorg_off | variant_minimal | 0.130 | 0.140 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| reorg_off | variant_parse | 0.130 | 0.050 | 0.080 | 0.0614 | [0.010, 0.150] | 11 | 3 | 100 |
| reorg_off | variant_structured_routine | 0.130 | 0.100 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| reorg_off | variant_typed_only | 0.130 | 0.080 | 0.050 | 0.3017 | [-0.030, 0.130] | 10 | 5 | 100 |
| reorg_on_graph_mdl_global_plateau | reorg_on_trace_mdl_accretive_everyk | 0.120 | 0.110 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| reorg_on_graph_mdl_global_plateau | reorg_sleepgate | 0.120 | 0.160 | -0.040 | 0.5023 | [-0.120, 0.050] | 8 | 12 | 100 |
| reorg_on_graph_mdl_global_plateau | reorg_stitch | 0.120 | 0.080 | 0.040 | 0.3428 | [-0.020, 0.100] | 7 | 3 | 100 |
| reorg_on_graph_mdl_global_plateau | rrmc_multi_round | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.070, 0.080] | 8 | 8 | 100 |
| reorg_on_graph_mdl_global_plateau | uot_entropy | 0.120 | 0.180 | -0.060 | 0.3074 | [-0.150, 0.030] | 9 | 15 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_cue_heavy | 0.120 | 0.220 | -0.100 | 0.0442 | [-0.190, -0.020] | 5 | 15 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_dspy_opt | 0.120 | 0.030 | 0.090 | 0.0265 | [0.030, 0.160] | 11 | 2 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_free_text | 0.120 | 0.100 | 0.020 | 0.7893 | [-0.050, 0.090] | 8 | 6 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_gepa | 0.120 | 0.050 | 0.070 | 0.1213 | [0.000, 0.150] | 11 | 4 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_minimal | 0.120 | 0.140 | -0.020 | 0.8026 | [-0.090, 0.060] | 7 | 9 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_parse | 0.120 | 0.050 | 0.070 | 0.0961 | [0.010, 0.140] | 10 | 3 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_structured_routine | 0.120 | 0.100 | 0.020 | 0.7893 | [-0.060, 0.090] | 8 | 6 | 100 |
| reorg_on_graph_mdl_global_plateau | variant_typed_only | 0.120 | 0.080 | 0.040 | 0.3865 | [-0.020, 0.110] | 8 | 4 | 100 |
| reorg_on_trace_mdl_accretive_everyk | reorg_sleepgate | 0.110 | 0.160 | -0.050 | 0.3320 | [-0.130, 0.030] | 6 | 11 | 100 |
| reorg_on_trace_mdl_accretive_everyk | reorg_stitch | 0.110 | 0.080 | 0.030 | 0.5050 | [-0.030, 0.090] | 6 | 3 | 100 |
| reorg_on_trace_mdl_accretive_everyk | rrmc_multi_round | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.080, 0.070] | 7 | 8 | 100 |
| reorg_on_trace_mdl_accretive_everyk | uot_entropy | 0.110 | 0.180 | -0.070 | 0.1687 | [-0.150, 0.020] | 6 | 13 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_cue_heavy | 0.110 | 0.220 | -0.110 | 0.0543 | [-0.210, -0.010] | 8 | 19 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_dspy_opt | 0.110 | 0.030 | 0.080 | 0.0614 | [0.010, 0.150] | 11 | 3 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_free_text | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.080, 0.090] | 9 | 8 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_gepa | 0.110 | 0.050 | 0.060 | 0.1489 | [-0.010, 0.130] | 9 | 3 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_minimal | 0.110 | 0.140 | -0.030 | 0.6464 | [-0.120, 0.050] | 8 | 11 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_parse | 0.110 | 0.050 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_structured_routine | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| reorg_on_trace_mdl_accretive_everyk | variant_typed_only | 0.110 | 0.080 | 0.030 | 0.5791 | [-0.050, 0.100] | 8 | 5 | 100 |
| reorg_sleepgate | reorg_stitch | 0.160 | 0.080 | 0.080 | 0.1175 | [-0.010, 0.170] | 14 | 6 | 100 |
| reorg_sleepgate | rrmc_multi_round | 0.160 | 0.120 | 0.040 | 0.5023 | [-0.050, 0.130] | 12 | 8 | 100 |
| reorg_sleepgate | uot_entropy | 0.160 | 0.180 | -0.020 | 0.8312 | [-0.120, 0.070] | 10 | 12 | 100 |
| reorg_sleepgate | variant_cue_heavy | 0.160 | 0.220 | -0.060 | 0.3074 | [-0.160, 0.040] | 9 | 15 | 100 |
| reorg_sleepgate | variant_dspy_opt | 0.160 | 0.030 | 0.130 | 0.0036 | [0.050, 0.210] | 15 | 2 | 100 |
| reorg_sleepgate | variant_free_text | 0.160 | 0.100 | 0.060 | 0.2636 | [-0.030, 0.150] | 13 | 7 | 100 |
| reorg_sleepgate | variant_gepa | 0.160 | 0.050 | 0.110 | 0.0218 | [0.020, 0.200] | 15 | 4 | 100 |
| reorg_sleepgate | variant_minimal | 0.160 | 0.140 | 0.020 | 0.8383 | [-0.070, 0.120] | 13 | 11 | 100 |
| reorg_sleepgate | variant_parse | 0.160 | 0.050 | 0.110 | 0.0218 | [0.030, 0.200] | 15 | 4 | 100 |
| reorg_sleepgate | variant_structured_routine | 0.160 | 0.100 | 0.060 | 0.2386 | [-0.030, 0.150] | 12 | 6 | 100 |
| reorg_sleepgate | variant_typed_only | 0.160 | 0.080 | 0.080 | 0.1356 | [-0.010, 0.170] | 15 | 7 | 100 |
| reorg_stitch | rrmc_multi_round | 0.080 | 0.120 | -0.040 | 0.3865 | [-0.110, 0.030] | 4 | 8 | 100 |
| reorg_stitch | uot_entropy | 0.080 | 0.180 | -0.100 | 0.0550 | [-0.180, -0.020] | 6 | 16 | 100 |
| reorg_stitch | variant_cue_heavy | 0.080 | 0.220 | -0.140 | 0.0080 | [-0.230, -0.050] | 5 | 19 | 100 |
| reorg_stitch | variant_dspy_opt | 0.080 | 0.030 | 0.050 | 0.2278 | [-0.020, 0.120] | 8 | 3 | 100 |
| reorg_stitch | variant_free_text | 0.080 | 0.100 | -0.020 | 0.7893 | [-0.100, 0.050] | 6 | 8 | 100 |
| reorg_stitch | variant_gepa | 0.080 | 0.050 | 0.030 | 0.5465 | [-0.030, 0.100] | 7 | 4 | 100 |
| reorg_stitch | variant_minimal | 0.080 | 0.140 | -0.060 | 0.2113 | [-0.130, 0.010] | 5 | 11 | 100 |
| reorg_stitch | variant_parse | 0.080 | 0.050 | 0.030 | 0.5791 | [-0.040, 0.100] | 8 | 5 | 100 |
| reorg_stitch | variant_structured_routine | 0.080 | 0.100 | -0.020 | 0.7518 | [-0.080, 0.040] | 4 | 6 | 100 |
| reorg_stitch | variant_typed_only | 0.080 | 0.080 | 0.000 | 1.0000 | [-0.060, 0.060] | 4 | 4 | 100 |
| mediq_policy | one_shot | 0.100 | 0.050 | 0.050 | 0.1824 | [-0.010, 0.110] | 7 | 2 | 100 |
| mediq_policy | pathrag | 0.100 | 0.300 | -0.200 | 1.94e-04 | [-0.290, -0.110] | 3 | 23 | 100 |
| mediq_policy | raptor | 0.100 | 0.160 | -0.060 | 0.2386 | [-0.140, 0.020] | 6 | 12 | 100 |
| mediq_policy | reorg_amem | 0.100 | 0.170 | -0.070 | 0.1904 | [-0.160, 0.020] | 7 | 14 | 100 |
| mediq_policy | reorg_dreamcoder | 0.100 | 0.170 | -0.070 | 0.1456 | [-0.160, 0.010] | 5 | 12 | 100 |
| mediq_policy | reorg_evolver | 0.100 | 0.120 | -0.020 | 0.7893 | [-0.090, 0.050] | 6 | 8 | 100 |
| mediq_policy | reorg_lilo | 0.100 | 0.140 | -0.040 | 0.4533 | [-0.120, 0.040] | 6 | 10 | 100 |
| mediq_policy | reorg_lrll | 0.100 | 0.130 | -0.030 | 0.6056 | [-0.110, 0.050] | 6 | 9 | 100 |
| mediq_policy | reorg_memp | 0.100 | 0.210 | -0.110 | 0.0371 | [-0.200, -0.010] | 6 | 17 | 100 |
| mediq_policy | reorg_memtree | 0.100 | 0.190 | -0.090 | 0.0523 | [-0.180, -0.010] | 4 | 13 | 100 |
| mediq_policy | reorg_off | 0.100 | 0.130 | -0.030 | 0.6276 | [-0.110, 0.060] | 7 | 10 | 100 |
| mediq_policy | reorg_on_graph_mdl_global_plateau | 0.100 | 0.120 | -0.020 | 0.7728 | [-0.090, 0.050] | 5 | 7 | 100 |
| mediq_policy | reorg_on_trace_mdl_accretive_everyk | 0.100 | 0.110 | -0.010 | 1.0000 | [-0.090, 0.070] | 7 | 8 | 100 |
| mediq_policy | reorg_sleepgate | 0.100 | 0.160 | -0.060 | 0.2386 | [-0.140, 0.020] | 6 | 12 | 100 |
| mediq_policy | reorg_stitch | 0.100 | 0.080 | 0.020 | 0.7518 | [-0.040, 0.080] | 6 | 4 | 100 |
| mediq_policy | rrmc_multi_round | 0.100 | 0.120 | -0.020 | 0.7518 | [-0.080, 0.040] | 4 | 6 | 100 |
| mediq_policy | uot_entropy | 0.100 | 0.180 | -0.080 | 0.1175 | [-0.160, 0.010] | 6 | 14 | 100 |
| mediq_policy | variant_cue_heavy | 0.100 | 0.220 | -0.120 | 0.0310 | [-0.220, -0.020] | 7 | 19 | 100 |
| mediq_policy | variant_dspy_opt | 0.100 | 0.030 | 0.070 | 0.0961 | [0.000, 0.140] | 10 | 3 | 100 |
| mediq_policy | variant_free_text | 0.100 | 0.100 | 0.000 | 1.0000 | [-0.080, 0.080] | 7 | 7 | 100 |
| mediq_policy | variant_gepa | 0.100 | 0.050 | 0.050 | 0.2673 | [-0.020, 0.120] | 9 | 4 | 100 |
| mediq_policy | variant_minimal | 0.100 | 0.140 | -0.040 | 0.3428 | [-0.100, 0.020] | 3 | 7 | 100 |
| mediq_policy | variant_parse | 0.100 | 0.050 | 0.050 | 0.2278 | [-0.010, 0.120] | 8 | 3 | 100 |
| mediq_policy | variant_structured_routine | 0.100 | 0.100 | 0.000 | 1.0000 | [-0.070, 0.060] | 5 | 5 | 100 |
| mediq_policy | variant_typed_only | 0.100 | 0.080 | 0.020 | 0.6831 | [-0.030, 0.070] | 4 | 2 | 100 |
| one_shot | pathrag | 0.050 | 0.300 | -0.250 | 8.32e-06 | [-0.340, -0.160] | 2 | 27 | 100 |
| one_shot | raptor | 0.050 | 0.160 | -0.110 | 0.0153 | [-0.190, -0.030] | 3 | 14 | 100 |
| one_shot | reorg_amem | 0.050 | 0.170 | -0.120 | 0.0095 | [-0.200, -0.040] | 3 | 15 | 100 |
| one_shot | reorg_dreamcoder | 0.050 | 0.170 | -0.120 | 0.0060 | [-0.200, -0.040] | 2 | 14 | 100 |
| one_shot | reorg_evolver | 0.050 | 0.120 | -0.070 | 0.0704 | [-0.140, -0.010] | 2 | 9 | 100 |
| one_shot | reorg_lilo | 0.050 | 0.140 | -0.090 | 0.0389 | [-0.160, -0.020] | 3 | 12 | 100 |
| one_shot | reorg_lrll | 0.050 | 0.130 | -0.080 | 0.0433 | [-0.150, -0.020] | 2 | 10 | 100 |
| one_shot | reorg_memp | 0.050 | 0.210 | -0.160 | 0.0014 | [-0.240, -0.070] | 3 | 19 | 100 |
| one_shot | reorg_memtree | 0.050 | 0.190 | -0.140 | 5.12e-04 | [-0.210, -0.080] | 0 | 14 | 100 |
| one_shot | reorg_off | 0.050 | 0.130 | -0.080 | 0.0801 | [-0.160, -0.010] | 4 | 12 | 100 |
| one_shot | reorg_on_graph_mdl_global_plateau | 0.050 | 0.120 | -0.070 | 0.0233 | [-0.120, -0.020] | 0 | 7 | 100 |
| one_shot | reorg_on_trace_mdl_accretive_everyk | 0.050 | 0.110 | -0.060 | 0.1138 | [-0.120, 0.000] | 2 | 8 | 100 |
| one_shot | reorg_sleepgate | 0.050 | 0.160 | -0.110 | 0.0218 | [-0.190, -0.030] | 4 | 15 | 100 |
| one_shot | reorg_stitch | 0.050 | 0.080 | -0.030 | 0.4497 | [-0.080, 0.020] | 2 | 5 | 100 |
| one_shot | rrmc_multi_round | 0.050 | 0.120 | -0.070 | 0.0704 | [-0.130, -0.010] | 2 | 9 | 100 |
| one_shot | uot_entropy | 0.050 | 0.180 | -0.130 | 0.0123 | [-0.210, -0.040] | 5 | 18 | 100 |
| one_shot | variant_cue_heavy | 0.050 | 0.220 | -0.170 | 8.49e-04 | [-0.260, -0.080] | 3 | 20 | 100 |
| one_shot | variant_dspy_opt | 0.050 | 0.030 | 0.020 | 0.7237 | [-0.040, 0.070] | 5 | 3 | 100 |
| one_shot | variant_free_text | 0.050 | 0.100 | -0.050 | 0.1824 | [-0.110, 0.010] | 2 | 7 | 100 |
| one_shot | variant_gepa | 0.050 | 0.050 | 0.000 | 1.0000 | [-0.050, 0.050] | 4 | 4 | 100 |
| one_shot | variant_minimal | 0.050 | 0.140 | -0.090 | 0.0389 | [-0.160, -0.020] | 3 | 12 | 100 |
| one_shot | variant_parse | 0.050 | 0.050 | 0.000 | 1.0000 | [-0.060, 0.050] | 4 | 4 | 100 |
| one_shot | variant_structured_routine | 0.050 | 0.100 | -0.050 | 0.1824 | [-0.110, 0.000] | 2 | 7 | 100 |
| one_shot | variant_typed_only | 0.050 | 0.080 | -0.030 | 0.4497 | [-0.080, 0.020] | 2 | 5 | 100 |
| rrmc_multi_round | uot_entropy | 0.120 | 0.180 | -0.060 | 0.2636 | [-0.150, 0.020] | 7 | 13 | 100 |
| rrmc_multi_round | variant_cue_heavy | 0.120 | 0.220 | -0.100 | 0.0662 | [-0.190, -0.010] | 7 | 17 | 100 |
| rrmc_multi_round | variant_dspy_opt | 0.120 | 0.030 | 0.090 | 0.0389 | [0.020, 0.160] | 12 | 3 | 100 |
| rrmc_multi_round | variant_free_text | 0.120 | 0.100 | 0.020 | 0.7893 | [-0.050, 0.090] | 8 | 6 | 100 |
| rrmc_multi_round | variant_gepa | 0.120 | 0.050 | 0.070 | 0.0961 | [0.000, 0.140] | 10 | 3 | 100 |
| rrmc_multi_round | variant_minimal | 0.120 | 0.140 | -0.020 | 0.7728 | [-0.080, 0.040] | 5 | 7 | 100 |
| rrmc_multi_round | variant_parse | 0.120 | 0.050 | 0.070 | 0.1213 | [0.000, 0.150] | 11 | 4 | 100 |
| rrmc_multi_round | variant_structured_routine | 0.120 | 0.100 | 0.020 | 0.7728 | [-0.050, 0.090] | 7 | 5 | 100 |
| rrmc_multi_round | variant_typed_only | 0.120 | 0.080 | 0.040 | 0.2207 | [-0.010, 0.090] | 5 | 1 | 100 |
| uot_entropy | variant_cue_heavy | 0.180 | 0.220 | -0.040 | 0.5959 | [-0.150, 0.070] | 14 | 18 | 100 |
| uot_entropy | variant_dspy_opt | 0.180 | 0.030 | 0.150 | 0.0023 | [0.070, 0.230] | 18 | 3 | 100 |
| uot_entropy | variant_free_text | 0.180 | 0.100 | 0.080 | 0.0990 | [0.000, 0.150] | 13 | 5 | 100 |
| uot_entropy | variant_gepa | 0.180 | 0.050 | 0.130 | 0.0019 | [0.060, 0.200] | 14 | 1 | 100 |
| uot_entropy | variant_minimal | 0.180 | 0.140 | 0.040 | 0.4795 | [-0.040, 0.120] | 11 | 7 | 100 |
| uot_entropy | variant_parse | 0.180 | 0.050 | 0.130 | 0.0036 | [0.060, 0.210] | 15 | 2 | 100 |
| uot_entropy | variant_structured_routine | 0.180 | 0.100 | 0.080 | 0.1356 | [-0.010, 0.170] | 15 | 7 | 100 |
| uot_entropy | variant_typed_only | 0.180 | 0.080 | 0.100 | 0.0442 | [0.020, 0.180] | 15 | 5 | 100 |
| arcmemo_oe | arcmemo_ps | 0.170 | 0.110 | 0.060 | 0.2386 | [-0.020, 0.140] | 12 | 6 | 100 |
| arcmemo_oe | barc_seeded | 0.170 | 0.090 | 0.080 | 0.0990 | [0.000, 0.160] | 13 | 5 | 100 |
| arcmemo_oe | barc_synthetic | 0.170 | 0.030 | 0.140 | 0.0012 | [0.070, 0.220] | 15 | 1 | 100 |
| arcmemo_oe | colbert_rerank | 0.170 | 0.180 | -0.010 | 1.0000 | [-0.110, 0.100] | 14 | 15 | 100 |
| arcmemo_oe | corpus_hipporag_init | 0.170 | 0.060 | 0.110 | 0.0291 | [0.020, 0.190] | 16 | 5 | 100 |
| arcmemo_oe | empty_start | 0.170 | 0.110 | 0.060 | 0.2386 | [-0.020, 0.150] | 12 | 6 | 100 |
| arcmemo_oe | flat_topk | 0.170 | 0.240 | -0.070 | 0.2482 | [-0.170, 0.030] | 10 | 17 | 100 |
| arcmemo_oe | graph_traversal | 0.170 | 0.200 | -0.030 | 0.6892 | [-0.130, 0.060] | 11 | 14 | 100 |
| arcmemo_oe | graphrag | 0.170 | 0.300 | -0.130 | 0.0209 | [-0.230, -0.020] | 7 | 20 | 100 |
| arcmemo_oe | hand_coded_reorg | 0.170 | 0.120 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| arcmemo_oe | hipporag2_filter | 0.170 | 0.100 | 0.070 | 0.2301 | [-0.030, 0.170] | 16 | 9 | 100 |
| arcmemo_oe | hipporag_ppr | 0.170 | 0.210 | -0.040 | 0.5563 | [-0.140, 0.070] | 11 | 15 | 100 |
| arcmemo_oe | hmem_hierarchical | 0.170 | 0.150 | 0.020 | 0.8231 | [-0.070, 0.110] | 11 | 9 | 100 |
| arcmemo_oe | lightrag | 0.170 | 0.300 | -0.130 | 0.0164 | [-0.230, -0.040] | 6 | 19 | 100 |
| arcmemo_oe | magma_multigraph | 0.170 | 0.030 | 0.140 | 0.0012 | [0.070, 0.220] | 15 | 1 | 100 |
| arcmemo_oe | mediq_policy | 0.170 | 0.100 | 0.070 | 0.1213 | [-0.010, 0.150] | 11 | 4 | 100 |
| arcmemo_oe | one_shot | 0.170 | 0.050 | 0.120 | 0.0060 | [0.050, 0.190] | 14 | 2 | 100 |
| arcmemo_oe | pathrag | 0.170 | 0.300 | -0.130 | 0.0311 | [-0.240, -0.020] | 9 | 22 | 100 |
| arcmemo_oe | raptor | 0.170 | 0.160 | 0.010 | 1.0000 | [-0.070, 0.090] | 10 | 9 | 100 |
| arcmemo_oe | reorg_amem | 0.170 | 0.170 | 0.000 | 1.0000 | [-0.080, 0.080] | 9 | 9 | 100 |
| arcmemo_oe | reorg_dreamcoder | 0.170 | 0.170 | 0.000 | 1.0000 | [-0.080, 0.070] | 7 | 7 | 100 |
| arcmemo_oe | reorg_evolver | 0.170 | 0.120 | 0.050 | 0.3588 | [-0.030, 0.130] | 12 | 7 | 100 |
| arcmemo_oe | reorg_lilo | 0.170 | 0.140 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| arcmemo_oe | reorg_lrll | 0.170 | 0.130 | 0.040 | 0.4227 | [-0.030, 0.120] | 9 | 5 | 100 |
| arcmemo_oe | reorg_memp | 0.170 | 0.210 | -0.040 | 0.5403 | [-0.130, 0.050] | 10 | 14 | 100 |
| arcmemo_oe | reorg_memtree | 0.170 | 0.190 | -0.020 | 0.8445 | [-0.120, 0.080] | 12 | 14 | 100 |
| arcmemo_oe | reorg_off | 0.170 | 0.130 | 0.040 | 0.5224 | [-0.050, 0.140] | 13 | 9 | 100 |
| arcmemo_oe | reorg_on_graph_mdl_global_plateau | 0.170 | 0.120 | 0.050 | 0.3827 | [-0.040, 0.140] | 13 | 8 | 100 |
| arcmemo_oe | reorg_on_trace_mdl_accretive_everyk | 0.170 | 0.110 | 0.060 | 0.2386 | [-0.020, 0.140] | 12 | 6 | 100 |
| arcmemo_oe | reorg_sleepgate | 0.170 | 0.160 | 0.010 | 1.0000 | [-0.080, 0.110] | 11 | 10 | 100 |
| arcmemo_oe | reorg_stitch | 0.170 | 0.080 | 0.090 | 0.0389 | [0.010, 0.160] | 12 | 3 | 100 |
| arcmemo_oe | rrmc_multi_round | 0.170 | 0.120 | 0.050 | 0.3320 | [-0.030, 0.130] | 11 | 6 | 100 |
| arcmemo_oe | uot_entropy | 0.170 | 0.180 | -0.010 | 1.0000 | [-0.100, 0.090] | 11 | 12 | 100 |
| arcmemo_oe | variant_cue_heavy | 0.170 | 0.220 | -0.050 | 0.4414 | [-0.150, 0.050] | 11 | 16 | 100 |
| arcmemo_oe | variant_dspy_opt | 0.170 | 0.030 | 0.140 | 0.0037 | [0.060, 0.220] | 17 | 3 | 100 |
| arcmemo_oe | variant_free_text | 0.170 | 0.100 | 0.070 | 0.1904 | [-0.020, 0.160] | 14 | 7 | 100 |
| arcmemo_oe | variant_gepa | 0.170 | 0.050 | 0.120 | 0.0139 | [0.040, 0.200] | 16 | 4 | 100 |
| arcmemo_oe | variant_minimal | 0.170 | 0.140 | 0.030 | 0.6767 | [-0.060, 0.130] | 13 | 10 | 100 |
| arcmemo_oe | variant_parse | 0.170 | 0.050 | 0.120 | 0.0095 | [0.040, 0.200] | 15 | 3 | 100 |
| arcmemo_oe | variant_structured_routine | 0.170 | 0.100 | 0.070 | 0.1213 | [0.000, 0.140] | 11 | 4 | 100 |
| arcmemo_oe | variant_typed_only | 0.170 | 0.080 | 0.090 | 0.0389 | [0.010, 0.170] | 12 | 3 | 100 |
| arcmemo_ps | barc_seeded | 0.110 | 0.090 | 0.020 | 0.7728 | [-0.050, 0.090] | 7 | 5 | 100 |
| arcmemo_ps | barc_synthetic | 0.110 | 0.030 | 0.080 | 0.0269 | [0.020, 0.140] | 9 | 1 | 100 |
| arcmemo_ps | colbert_rerank | 0.110 | 0.180 | -0.070 | 0.1687 | [-0.150, 0.020] | 6 | 13 | 100 |
| arcmemo_ps | corpus_hipporag_init | 0.110 | 0.060 | 0.050 | 0.2673 | [-0.020, 0.120] | 9 | 4 | 100 |
| arcmemo_ps | empty_start | 0.110 | 0.110 | 0.000 | 1.0000 | [-0.080, 0.080] | 9 | 9 | 100 |
| arcmemo_ps | flat_topk | 0.110 | 0.240 | -0.130 | 0.0259 | [-0.230, -0.030] | 8 | 21 | 100 |
| arcmemo_ps | graph_traversal | 0.110 | 0.200 | -0.090 | 0.1237 | [-0.190, 0.010] | 9 | 18 | 100 |
| arcmemo_ps | graphrag | 0.110 | 0.300 | -0.190 | 0.0017 | [-0.300, -0.080] | 7 | 26 | 100 |
| arcmemo_ps | hand_coded_reorg | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.080, 0.060] | 5 | 6 | 100 |
| arcmemo_ps | hipporag2_filter | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.070, 0.100] | 10 | 9 | 100 |
| arcmemo_ps | hipporag_ppr | 0.110 | 0.210 | -0.100 | 0.0550 | [-0.190, 0.000] | 6 | 16 | 100 |
| arcmemo_ps | hmem_hierarchical | 0.110 | 0.150 | -0.040 | 0.5224 | [-0.130, 0.050] | 9 | 13 | 100 |
| arcmemo_ps | lightrag | 0.110 | 0.300 | -0.190 | 5.32e-04 | [-0.290, -0.090] | 4 | 23 | 100 |
| arcmemo_ps | magma_multigraph | 0.110 | 0.030 | 0.080 | 0.0614 | [0.010, 0.160] | 11 | 3 | 100 |
| arcmemo_ps | mediq_policy | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| arcmemo_ps | one_shot | 0.110 | 0.050 | 0.060 | 0.1138 | [0.000, 0.120] | 8 | 2 | 100 |
| arcmemo_ps | pathrag | 0.110 | 0.300 | -0.190 | 0.0012 | [-0.290, -0.090] | 6 | 25 | 100 |
| arcmemo_ps | raptor | 0.110 | 0.160 | -0.050 | 0.3588 | [-0.130, 0.030] | 7 | 12 | 100 |
| arcmemo_ps | reorg_amem | 0.110 | 0.170 | -0.060 | 0.2113 | [-0.140, 0.020] | 5 | 11 | 100 |
| arcmemo_ps | reorg_dreamcoder | 0.110 | 0.170 | -0.060 | 0.2113 | [-0.140, 0.010] | 5 | 11 | 100 |
| arcmemo_ps | reorg_evolver | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| arcmemo_ps | reorg_lilo | 0.110 | 0.140 | -0.030 | 0.5791 | [-0.100, 0.030] | 5 | 8 | 100 |
| arcmemo_ps | reorg_lrll | 0.110 | 0.130 | -0.020 | 0.7893 | [-0.090, 0.050] | 6 | 8 | 100 |
| arcmemo_ps | reorg_memp | 0.110 | 0.210 | -0.100 | 0.0442 | [-0.180, -0.010] | 5 | 15 | 100 |
| arcmemo_ps | reorg_memtree | 0.110 | 0.190 | -0.080 | 0.0614 | [-0.150, -0.010] | 3 | 11 | 100 |
| arcmemo_ps | reorg_off | 0.110 | 0.130 | -0.020 | 0.7893 | [-0.090, 0.060] | 6 | 8 | 100 |
| arcmemo_ps | reorg_on_graph_mdl_global_plateau | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.080, 0.060] | 5 | 6 | 100 |
| arcmemo_ps | reorg_on_trace_mdl_accretive_everyk | 0.110 | 0.110 | 0.000 | 1.0000 | [-0.070, 0.080] | 7 | 7 | 100 |
| arcmemo_ps | reorg_sleepgate | 0.110 | 0.160 | -0.050 | 0.2673 | [-0.120, 0.020] | 4 | 9 | 100 |
| arcmemo_ps | reorg_stitch | 0.110 | 0.080 | 0.030 | 0.5791 | [-0.040, 0.100] | 8 | 5 | 100 |
| arcmemo_ps | rrmc_multi_round | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.080, 0.070] | 6 | 7 | 100 |
| arcmemo_ps | uot_entropy | 0.110 | 0.180 | -0.070 | 0.1687 | [-0.150, 0.010] | 6 | 13 | 100 |
| arcmemo_ps | variant_cue_heavy | 0.110 | 0.220 | -0.110 | 0.0371 | [-0.200, -0.020] | 6 | 17 | 100 |
| arcmemo_ps | variant_dspy_opt | 0.110 | 0.030 | 0.080 | 0.0433 | [0.020, 0.150] | 10 | 2 | 100 |
| arcmemo_ps | variant_free_text | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.070, 0.090] | 8 | 7 | 100 |
| arcmemo_ps | variant_gepa | 0.110 | 0.050 | 0.060 | 0.1489 | [-0.010, 0.130] | 9 | 3 | 100 |
| arcmemo_ps | variant_minimal | 0.110 | 0.140 | -0.030 | 0.6276 | [-0.100, 0.050] | 7 | 10 | 100 |
| arcmemo_ps | variant_parse | 0.110 | 0.050 | 0.060 | 0.2113 | [-0.020, 0.150] | 11 | 5 | 100 |
| arcmemo_ps | variant_structured_routine | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| arcmemo_ps | variant_typed_only | 0.110 | 0.080 | 0.030 | 0.5791 | [-0.040, 0.100] | 8 | 5 | 100 |
| variant_cue_heavy | variant_dspy_opt | 0.220 | 0.030 | 0.190 | 1.75e-04 | [0.110, 0.280] | 21 | 2 | 100 |
| variant_cue_heavy | variant_free_text | 0.220 | 0.100 | 0.120 | 0.0190 | [0.030, 0.210] | 17 | 5 | 100 |
| variant_cue_heavy | variant_gepa | 0.220 | 0.050 | 0.170 | 8.49e-04 | [0.080, 0.260] | 20 | 3 | 100 |
| variant_cue_heavy | variant_minimal | 0.220 | 0.140 | 0.080 | 0.1530 | [-0.010, 0.170] | 16 | 8 | 100 |
| variant_cue_heavy | variant_parse | 0.220 | 0.050 | 0.170 | 4.80e-04 | [0.090, 0.260] | 19 | 2 | 100 |
| variant_cue_heavy | variant_structured_routine | 0.220 | 0.100 | 0.120 | 0.0190 | [0.030, 0.210] | 17 | 5 | 100 |
| variant_cue_heavy | variant_typed_only | 0.220 | 0.080 | 0.140 | 0.0056 | [0.050, 0.220] | 18 | 4 | 100 |
| variant_dspy_opt | variant_free_text | 0.030 | 0.100 | -0.070 | 0.0961 | [-0.140, 0.000] | 3 | 10 | 100 |
| variant_dspy_opt | variant_gepa | 0.030 | 0.050 | -0.020 | 0.7237 | [-0.080, 0.040] | 3 | 5 | 100 |
| variant_dspy_opt | variant_minimal | 0.030 | 0.140 | -0.110 | 0.0098 | [-0.180, -0.030] | 2 | 13 | 100 |
| variant_dspy_opt | variant_parse | 0.030 | 0.050 | -0.020 | 0.7237 | [-0.070, 0.040] | 3 | 5 | 100 |
| variant_dspy_opt | variant_structured_routine | 0.030 | 0.100 | -0.070 | 0.0961 | [-0.140, 0.000] | 3 | 10 | 100 |
| variant_dspy_opt | variant_typed_only | 0.030 | 0.080 | -0.050 | 0.2278 | [-0.120, 0.010] | 3 | 8 | 100 |
| variant_free_text | variant_gepa | 0.100 | 0.050 | 0.050 | 0.2278 | [-0.010, 0.110] | 8 | 3 | 100 |
| variant_free_text | variant_minimal | 0.100 | 0.140 | -0.040 | 0.3865 | [-0.110, 0.020] | 4 | 8 | 100 |
| variant_free_text | variant_parse | 0.100 | 0.050 | 0.050 | 0.1306 | [0.000, 0.110] | 6 | 1 | 100 |
| variant_free_text | variant_structured_routine | 0.100 | 0.100 | 0.000 | 1.0000 | [-0.070, 0.070] | 7 | 7 | 100 |
| variant_free_text | variant_typed_only | 0.100 | 0.080 | 0.020 | 0.7518 | [-0.050, 0.080] | 6 | 4 | 100 |
| variant_gepa | variant_minimal | 0.050 | 0.140 | -0.090 | 0.0389 | [-0.160, -0.020] | 3 | 12 | 100 |
| variant_gepa | variant_parse | 0.050 | 0.050 | 0.000 | 1.0000 | [-0.040, 0.050] | 3 | 3 | 100 |
| variant_gepa | variant_structured_routine | 0.050 | 0.100 | -0.050 | 0.2278 | [-0.120, 0.010] | 3 | 8 | 100 |
| variant_gepa | variant_typed_only | 0.050 | 0.080 | -0.030 | 0.5050 | [-0.090, 0.030] | 3 | 6 | 100 |
| variant_minimal | variant_parse | 0.140 | 0.050 | 0.090 | 0.0265 | [0.030, 0.160] | 11 | 2 | 100 |
| variant_minimal | variant_structured_routine | 0.140 | 0.100 | 0.040 | 0.4795 | [-0.040, 0.120] | 11 | 7 | 100 |
| variant_minimal | variant_typed_only | 0.140 | 0.080 | 0.060 | 0.0771 | [0.010, 0.110] | 7 | 1 | 100 |
| variant_parse | variant_structured_routine | 0.050 | 0.100 | -0.050 | 0.2673 | [-0.120, 0.020] | 4 | 9 | 100 |
| variant_parse | variant_typed_only | 0.050 | 0.080 | -0.030 | 0.5050 | [-0.100, 0.030] | 3 | 6 | 100 |
| variant_structured_routine | variant_typed_only | 0.100 | 0.080 | 0.020 | 0.7518 | [-0.040, 0.080] | 6 | 4 | 100 |
| adas_style_search | alma_style_metaedit | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| adas_style_search | arcmemo_oe | 0.130 | 0.170 | -0.040 | 0.4795 | [-0.120, 0.040] | 7 | 11 | 100 |
| adas_style_search | arcmemo_ps | 0.130 | 0.110 | 0.020 | 0.7518 | [-0.040, 0.080] | 6 | 4 | 100 |
| adas_style_search | barc_seeded | 0.130 | 0.090 | 0.040 | 0.3428 | [-0.020, 0.100] | 7 | 3 | 100 |
| adas_style_search | barc_synthetic | 0.130 | 0.030 | 0.100 | 0.0094 | [0.040, 0.170] | 11 | 1 | 100 |
| adas_style_search | colbert_rerank | 0.130 | 0.180 | -0.050 | 0.4042 | [-0.140, 0.040] | 9 | 14 | 100 |
| adas_style_search | corpus_hipporag_init | 0.130 | 0.060 | 0.070 | 0.1213 | [0.000, 0.140] | 11 | 4 | 100 |
| adas_style_search | empty_start | 0.130 | 0.110 | 0.020 | 0.8137 | [-0.060, 0.100] | 10 | 8 | 100 |
| adas_style_search | flat_topk | 0.130 | 0.240 | -0.110 | 0.0817 | [-0.210, 0.000] | 11 | 22 | 100 |
| adas_style_search | graph_traversal | 0.130 | 0.200 | -0.070 | 0.2301 | [-0.160, 0.030] | 9 | 16 | 100 |
| adas_style_search | graphrag | 0.130 | 0.300 | -0.170 | 0.0068 | [-0.280, -0.060] | 9 | 26 | 100 |
| adas_style_search | hand_coded_reorg | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.050, 0.070] | 6 | 5 | 100 |
| adas_style_search | hipporag2_filter | 0.130 | 0.100 | 0.030 | 0.6276 | [-0.050, 0.110] | 10 | 7 | 100 |
| adas_style_search | hipporag_ppr | 0.130 | 0.210 | -0.080 | 0.1175 | [-0.160, 0.010] | 6 | 14 | 100 |
| adas_style_search | hmem_hierarchical | 0.130 | 0.150 | -0.020 | 0.8231 | [-0.110, 0.070] | 9 | 11 | 100 |
| adas_style_search | lightrag | 0.130 | 0.300 | -0.170 | 0.0021 | [-0.270, -0.070] | 5 | 22 | 100 |
| adas_style_search | magma_multigraph | 0.130 | 0.030 | 0.100 | 0.0094 | [0.040, 0.170] | 11 | 1 | 100 |
| adas_style_search | mediq_policy | 0.130 | 0.100 | 0.030 | 0.5791 | [-0.040, 0.100] | 8 | 5 | 100 |
| adas_style_search | one_shot | 0.130 | 0.050 | 0.080 | 0.0433 | [0.020, 0.150] | 10 | 2 | 100 |
| adas_style_search | pathrag | 0.130 | 0.300 | -0.170 | 0.0030 | [-0.260, -0.070] | 6 | 23 | 100 |
| adas_style_search | raptor | 0.130 | 0.160 | -0.030 | 0.6464 | [-0.120, 0.050] | 8 | 11 | 100 |
| adas_style_search | reorg_amem | 0.130 | 0.170 | -0.040 | 0.4795 | [-0.120, 0.040] | 7 | 11 | 100 |
| adas_style_search | reorg_dreamcoder | 0.130 | 0.170 | -0.040 | 0.4533 | [-0.120, 0.040] | 6 | 10 | 100 |
| adas_style_search | reorg_evolver | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| adas_style_search | reorg_lilo | 0.130 | 0.140 | -0.010 | 1.0000 | [-0.090, 0.060] | 7 | 8 | 100 |
| adas_style_search | reorg_lrll | 0.130 | 0.130 | 0.000 | 1.0000 | [-0.080, 0.080] | 9 | 9 | 100 |
| adas_style_search | reorg_memp | 0.130 | 0.210 | -0.080 | 0.0801 | [-0.160, 0.000] | 4 | 12 | 100 |
| adas_style_search | reorg_memtree | 0.130 | 0.190 | -0.060 | 0.2636 | [-0.150, 0.030] | 7 | 13 | 100 |
| adas_style_search | reorg_off | 0.130 | 0.130 | 0.000 | 1.0000 | [-0.070, 0.070] | 7 | 7 | 100 |
| adas_style_search | reorg_on_graph_mdl_global_plateau | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.060, 0.080] | 8 | 7 | 100 |
| adas_style_search | reorg_on_trace_mdl_accretive_everyk | 0.130 | 0.110 | 0.020 | 0.8026 | [-0.060, 0.100] | 9 | 7 | 100 |
| adas_style_search | reorg_sleepgate | 0.130 | 0.160 | -0.030 | 0.5791 | [-0.100, 0.040] | 5 | 8 | 100 |
| adas_style_search | reorg_stitch | 0.130 | 0.080 | 0.050 | 0.2673 | [-0.010, 0.120] | 9 | 4 | 100 |
| adas_style_search | rrmc_multi_round | 0.130 | 0.120 | 0.010 | 1.0000 | [-0.050, 0.080] | 6 | 5 | 100 |
| adas_style_search | uot_entropy | 0.130 | 0.180 | -0.050 | 0.4042 | [-0.140, 0.040] | 9 | 14 | 100 |
| adas_style_search | variant_cue_heavy | 0.130 | 0.220 | -0.090 | 0.0665 | [-0.170, -0.010] | 5 | 14 | 100 |
| adas_style_search | variant_dspy_opt | 0.130 | 0.030 | 0.100 | 0.0244 | [0.030, 0.170] | 13 | 3 | 100 |
| adas_style_search | variant_free_text | 0.130 | 0.100 | 0.030 | 0.5791 | [-0.040, 0.100] | 8 | 5 | 100 |
| adas_style_search | variant_gepa | 0.130 | 0.050 | 0.080 | 0.0614 | [0.010, 0.160] | 11 | 3 | 100 |
| adas_style_search | variant_minimal | 0.130 | 0.140 | -0.010 | 1.0000 | [-0.090, 0.070] | 9 | 10 | 100 |
| adas_style_search | variant_parse | 0.130 | 0.050 | 0.080 | 0.0801 | [0.010, 0.160] | 12 | 4 | 100 |
| adas_style_search | variant_structured_routine | 0.130 | 0.100 | 0.030 | 0.5465 | [-0.040, 0.100] | 7 | 4 | 100 |
| adas_style_search | variant_typed_only | 0.130 | 0.080 | 0.050 | 0.2673 | [-0.010, 0.120] | 9 | 4 | 100 |
| alma_style_metaedit | arcmemo_oe | 0.120 | 0.170 | -0.050 | 0.3588 | [-0.130, 0.030] | 7 | 12 | 100 |
| alma_style_metaedit | arcmemo_ps | 0.120 | 0.110 | 0.010 | 1.0000 | [-0.070, 0.090] | 9 | 8 | 100 |
| alma_style_metaedit | barc_seeded | 0.120 | 0.090 | 0.030 | 0.4497 | [-0.020, 0.080] | 5 | 2 | 100 |
| alma_style_metaedit | barc_synthetic | 0.120 | 0.030 | 0.090 | 0.0077 | [0.040, 0.150] | 9 | 0 | 100 |
| alma_style_metaedit | colbert_rerank | 0.120 | 0.180 | -0.060 | 0.2113 | [-0.140, 0.020] | 5 | 11 | 100 |
| alma_style_metaedit | corpus_hipporag_init | 0.120 | 0.060 | 0.060 | 0.1814 | [-0.020, 0.130] | 10 | 4 | 100 |
| alma_style_metaedit | empty_start | 0.120 | 0.110 | 0.010 | 1.0000 | [-0.070, 0.090] | 9 | 8 | 100 |
| alma_style_metaedit | flat_topk | 0.120 | 0.240 | -0.120 | 0.0446 | [-0.230, -0.010] | 9 | 21 | 100 |
| alma_style_metaedit | graph_traversal | 0.120 | 0.200 | -0.080 | 0.1530 | [-0.180, 0.010] | 8 | 16 | 100 |
| alma_style_metaedit | graphrag | 0.120 | 0.300 | -0.180 | 0.0027 | [-0.290, -0.070] | 7 | 25 | 100 |
| alma_style_metaedit | hand_coded_reorg | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.060, 0.060] | 5 | 5 | 100 |
| alma_style_metaedit | hipporag2_filter | 0.120 | 0.100 | 0.020 | 0.8231 | [-0.060, 0.110] | 11 | 9 | 100 |
| alma_style_metaedit | hipporag_ppr | 0.120 | 0.210 | -0.090 | 0.0953 | [-0.190, 0.000] | 7 | 16 | 100 |
| alma_style_metaedit | hmem_hierarchical | 0.120 | 0.150 | -0.030 | 0.6464 | [-0.120, 0.050] | 8 | 11 | 100 |
| alma_style_metaedit | lightrag | 0.120 | 0.300 | -0.180 | 8.56e-04 | [-0.270, -0.090] | 4 | 22 | 100 |
| alma_style_metaedit | magma_multigraph | 0.120 | 0.030 | 0.090 | 0.0159 | [0.030, 0.150] | 10 | 1 | 100 |
| alma_style_metaedit | mediq_policy | 0.120 | 0.100 | 0.020 | 0.8026 | [-0.070, 0.100] | 9 | 7 | 100 |
| alma_style_metaedit | one_shot | 0.120 | 0.050 | 0.070 | 0.0704 | [0.010, 0.130] | 9 | 2 | 100 |
| alma_style_metaedit | pathrag | 0.120 | 0.300 | -0.180 | 0.0019 | [-0.290, -0.080] | 6 | 24 | 100 |
| alma_style_metaedit | raptor | 0.120 | 0.160 | -0.040 | 0.4795 | [-0.120, 0.040] | 7 | 11 | 100 |
| alma_style_metaedit | reorg_amem | 0.120 | 0.170 | -0.050 | 0.4042 | [-0.150, 0.040] | 9 | 14 | 100 |
| alma_style_metaedit | reorg_dreamcoder | 0.120 | 0.170 | -0.050 | 0.3588 | [-0.140, 0.040] | 7 | 12 | 100 |
| alma_style_metaedit | reorg_evolver | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.080, 0.080] | 9 | 9 | 100 |
| alma_style_metaedit | reorg_lilo | 0.120 | 0.140 | -0.020 | 0.8137 | [-0.110, 0.060] | 8 | 10 | 100 |
| alma_style_metaedit | reorg_lrll | 0.120 | 0.130 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| alma_style_metaedit | reorg_memp | 0.120 | 0.210 | -0.090 | 0.0809 | [-0.180, 0.000] | 6 | 15 | 100 |
| alma_style_metaedit | reorg_memtree | 0.120 | 0.190 | -0.070 | 0.1904 | [-0.170, 0.020] | 7 | 14 | 100 |
| alma_style_metaedit | reorg_off | 0.120 | 0.130 | -0.010 | 1.0000 | [-0.100, 0.080] | 9 | 10 | 100 |
| alma_style_metaedit | reorg_on_graph_mdl_global_plateau | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.070, 0.070] | 7 | 7 | 100 |
| alma_style_metaedit | reorg_on_trace_mdl_accretive_everyk | 0.120 | 0.110 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| alma_style_metaedit | reorg_sleepgate | 0.120 | 0.160 | -0.040 | 0.4533 | [-0.120, 0.040] | 6 | 10 | 100 |
| alma_style_metaedit | reorg_stitch | 0.120 | 0.080 | 0.040 | 0.3428 | [-0.020, 0.100] | 7 | 3 | 100 |
| alma_style_metaedit | rrmc_multi_round | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.080, 0.080] | 8 | 8 | 100 |
| alma_style_metaedit | uot_entropy | 0.120 | 0.180 | -0.060 | 0.2864 | [-0.150, 0.030] | 8 | 14 | 100 |
| alma_style_metaedit | variant_cue_heavy | 0.120 | 0.220 | -0.100 | 0.0776 | [-0.200, -0.010] | 8 | 18 | 100 |
| alma_style_metaedit | variant_dspy_opt | 0.120 | 0.030 | 0.090 | 0.0389 | [0.010, 0.160] | 12 | 3 | 100 |
| alma_style_metaedit | variant_free_text | 0.120 | 0.100 | 0.020 | 0.7728 | [-0.050, 0.080] | 7 | 5 | 100 |
| alma_style_metaedit | variant_gepa | 0.120 | 0.050 | 0.070 | 0.0961 | [0.000, 0.140] | 10 | 3 | 100 |
| alma_style_metaedit | variant_minimal | 0.120 | 0.140 | -0.020 | 0.8231 | [-0.110, 0.070] | 9 | 11 | 100 |
| alma_style_metaedit | variant_parse | 0.120 | 0.050 | 0.070 | 0.0961 | [0.000, 0.140] | 10 | 3 | 100 |
| alma_style_metaedit | variant_structured_routine | 0.120 | 0.100 | 0.020 | 0.7728 | [-0.050, 0.090] | 7 | 5 | 100 |
| alma_style_metaedit | variant_typed_only | 0.120 | 0.080 | 0.040 | 0.4533 | [-0.040, 0.110] | 10 | 6 | 100 |
| hand_coded_reorg | hipporag2_filter | 0.120 | 0.100 | 0.020 | 0.8231 | [-0.060, 0.110] | 11 | 9 | 100 |
| hand_coded_reorg | hipporag_ppr | 0.120 | 0.210 | -0.090 | 0.0953 | [-0.180, 0.000] | 7 | 16 | 100 |
| hand_coded_reorg | hmem_hierarchical | 0.120 | 0.150 | -0.030 | 0.6625 | [-0.120, 0.060] | 9 | 12 | 100 |
| hand_coded_reorg | lightrag | 0.120 | 0.300 | -0.180 | 0.0013 | [-0.270, -0.080] | 5 | 23 | 100 |
| hand_coded_reorg | magma_multigraph | 0.120 | 0.030 | 0.090 | 0.0389 | [0.020, 0.170] | 12 | 3 | 100 |
| hand_coded_reorg | mediq_policy | 0.120 | 0.100 | 0.020 | 0.7893 | [-0.050, 0.090] | 8 | 6 | 100 |
| hand_coded_reorg | one_shot | 0.120 | 0.050 | 0.070 | 0.0455 | [0.020, 0.130] | 8 | 1 | 100 |
| hand_coded_reorg | pathrag | 0.120 | 0.300 | -0.180 | 0.0019 | [-0.280, -0.080] | 6 | 24 | 100 |
| hand_coded_reorg | raptor | 0.120 | 0.160 | -0.040 | 0.4227 | [-0.110, 0.030] | 5 | 9 | 100 |
| hand_coded_reorg | reorg_amem | 0.120 | 0.170 | -0.050 | 0.3588 | [-0.140, 0.030] | 7 | 12 | 100 |
| hand_coded_reorg | reorg_dreamcoder | 0.120 | 0.170 | -0.050 | 0.3320 | [-0.130, 0.040] | 6 | 11 | 100 |
| hand_coded_reorg | reorg_evolver | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.070, 0.070] | 7 | 7 | 100 |
| hand_coded_reorg | reorg_lilo | 0.120 | 0.140 | -0.020 | 0.8137 | [-0.100, 0.060] | 8 | 10 | 100 |
| hand_coded_reorg | reorg_lrll | 0.120 | 0.130 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| hand_coded_reorg | reorg_memp | 0.120 | 0.210 | -0.090 | 0.0809 | [-0.170, 0.000] | 6 | 15 | 100 |
| hand_coded_reorg | reorg_memtree | 0.120 | 0.190 | -0.070 | 0.1456 | [-0.150, 0.010] | 5 | 12 | 100 |
| hand_coded_reorg | reorg_off | 0.120 | 0.130 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| hand_coded_reorg | reorg_on_graph_mdl_global_plateau | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.060, 0.060] | 5 | 5 | 100 |
| hand_coded_reorg | reorg_on_trace_mdl_accretive_everyk | 0.120 | 0.110 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| hand_coded_reorg | reorg_sleepgate | 0.120 | 0.160 | -0.040 | 0.4533 | [-0.120, 0.040] | 6 | 10 | 100 |
| hand_coded_reorg | reorg_stitch | 0.120 | 0.080 | 0.040 | 0.3428 | [-0.020, 0.110] | 7 | 3 | 100 |
| hand_coded_reorg | rrmc_multi_round | 0.120 | 0.120 | 0.000 | 1.0000 | [-0.070, 0.070] | 7 | 7 | 100 |
| hand_coded_reorg | uot_entropy | 0.120 | 0.180 | -0.060 | 0.3268 | [-0.150, 0.040] | 10 | 16 | 100 |
| hand_coded_reorg | variant_cue_heavy | 0.120 | 0.220 | -0.100 | 0.0339 | [-0.180, -0.020] | 4 | 14 | 100 |
| hand_coded_reorg | variant_dspy_opt | 0.120 | 0.030 | 0.090 | 0.0265 | [0.030, 0.160] | 11 | 2 | 100 |
| hand_coded_reorg | variant_free_text | 0.120 | 0.100 | 0.020 | 0.7893 | [-0.060, 0.100] | 8 | 6 | 100 |
| hand_coded_reorg | variant_gepa | 0.120 | 0.050 | 0.070 | 0.0961 | [0.010, 0.140] | 10 | 3 | 100 |
| hand_coded_reorg | variant_minimal | 0.120 | 0.140 | -0.020 | 0.8026 | [-0.090, 0.060] | 7 | 9 | 100 |
| hand_coded_reorg | variant_parse | 0.120 | 0.050 | 0.070 | 0.1213 | [0.000, 0.150] | 11 | 4 | 100 |
| hand_coded_reorg | variant_structured_routine | 0.120 | 0.100 | 0.020 | 0.7518 | [-0.040, 0.080] | 6 | 4 | 100 |
| hand_coded_reorg | variant_typed_only | 0.120 | 0.080 | 0.040 | 0.4227 | [-0.030, 0.110] | 9 | 5 | 100 |
| barc_seeded | barc_synthetic | 0.090 | 0.030 | 0.060 | 0.0771 | [0.010, 0.110] | 7 | 1 | 100 |
| barc_seeded | colbert_rerank | 0.090 | 0.180 | -0.090 | 0.0809 | [-0.180, 0.000] | 6 | 15 | 100 |
| barc_seeded | corpus_hipporag_init | 0.090 | 0.060 | 0.030 | 0.5050 | [-0.030, 0.090] | 6 | 3 | 100 |
| barc_seeded | empty_start | 0.090 | 0.110 | -0.020 | 0.7893 | [-0.090, 0.050] | 6 | 8 | 100 |
| barc_seeded | flat_topk | 0.090 | 0.240 | -0.150 | 0.0119 | [-0.260, -0.040] | 8 | 23 | 100 |
| barc_seeded | graph_traversal | 0.090 | 0.200 | -0.110 | 0.0455 | [-0.210, -0.010] | 7 | 18 | 100 |
| barc_seeded | graphrag | 0.090 | 0.300 | -0.210 | 7.23e-04 | [-0.320, -0.090] | 7 | 28 | 100 |
| barc_seeded | hand_coded_reorg | 0.090 | 0.120 | -0.030 | 0.5050 | [-0.090, 0.020] | 3 | 6 | 100 |
| barc_seeded | hipporag2_filter | 0.090 | 0.100 | -0.010 | 1.0000 | [-0.080, 0.070] | 7 | 8 | 100 |
| barc_seeded | hipporag_ppr | 0.090 | 0.210 | -0.120 | 0.0247 | [-0.210, -0.020] | 6 | 18 | 100 |
| barc_seeded | hmem_hierarchical | 0.090 | 0.150 | -0.060 | 0.2636 | [-0.140, 0.030] | 7 | 13 | 100 |
| barc_seeded | lightrag | 0.090 | 0.300 | -0.210 | 2.04e-04 | [-0.310, -0.110] | 4 | 25 | 100 |
| barc_seeded | magma_multigraph | 0.090 | 0.030 | 0.060 | 0.1489 | [0.000, 0.140] | 9 | 3 | 100 |
| barc_seeded | mediq_policy | 0.090 | 0.100 | -0.010 | 1.0000 | [-0.080, 0.060] | 6 | 7 | 100 |
| barc_seeded | one_shot | 0.090 | 0.050 | 0.040 | 0.3428 | [-0.020, 0.100] | 7 | 3 | 100 |
| barc_seeded | pathrag | 0.090 | 0.300 | -0.210 | 2.04e-04 | [-0.310, -0.110] | 4 | 25 | 100 |
| barc_seeded | raptor | 0.090 | 0.160 | -0.070 | 0.1687 | [-0.150, 0.010] | 6 | 13 | 100 |
| barc_seeded | reorg_amem | 0.090 | 0.170 | -0.080 | 0.1175 | [-0.170, 0.000] | 6 | 14 | 100 |
| barc_seeded | reorg_dreamcoder | 0.090 | 0.170 | -0.080 | 0.1175 | [-0.170, 0.010] | 6 | 14 | 100 |
| barc_seeded | reorg_evolver | 0.090 | 0.120 | -0.030 | 0.6056 | [-0.100, 0.040] | 6 | 9 | 100 |
| barc_seeded | reorg_lilo | 0.090 | 0.140 | -0.050 | 0.3320 | [-0.130, 0.030] | 6 | 11 | 100 |
| barc_seeded | reorg_lrll | 0.090 | 0.130 | -0.040 | 0.4533 | [-0.120, 0.030] | 6 | 10 | 100 |
| barc_seeded | reorg_memp | 0.090 | 0.210 | -0.120 | 0.0190 | [-0.210, -0.030] | 5 | 17 | 100 |
| barc_seeded | reorg_memtree | 0.090 | 0.190 | -0.100 | 0.0339 | [-0.190, -0.020] | 4 | 14 | 100 |
| barc_seeded | reorg_off | 0.090 | 0.130 | -0.040 | 0.4533 | [-0.120, 0.040] | 6 | 10 | 100 |
| barc_seeded | reorg_on_graph_mdl_global_plateau | 0.090 | 0.120 | -0.030 | 0.6056 | [-0.110, 0.050] | 6 | 9 | 100 |
| barc_seeded | reorg_on_trace_mdl_accretive_everyk | 0.090 | 0.110 | -0.020 | 0.7893 | [-0.090, 0.060] | 6 | 8 | 100 |
| barc_seeded | reorg_sleepgate | 0.090 | 0.160 | -0.070 | 0.1213 | [-0.150, 0.000] | 4 | 11 | 100 |
| barc_seeded | reorg_stitch | 0.090 | 0.080 | 0.010 | 1.0000 | [-0.050, 0.080] | 6 | 5 | 100 |
| barc_seeded | rrmc_multi_round | 0.090 | 0.120 | -0.030 | 0.6056 | [-0.100, 0.040] | 6 | 9 | 100 |
| barc_seeded | uot_entropy | 0.090 | 0.180 | -0.090 | 0.0809 | [-0.170, 0.000] | 6 | 15 | 100 |
| barc_seeded | variant_cue_heavy | 0.090 | 0.220 | -0.130 | 0.0123 | [-0.220, -0.040] | 5 | 18 | 100 |
| barc_seeded | variant_dspy_opt | 0.090 | 0.030 | 0.060 | 0.1489 | [-0.010, 0.130] | 9 | 3 | 100 |
| barc_seeded | variant_free_text | 0.090 | 0.100 | -0.010 | 1.0000 | [-0.070, 0.050] | 5 | 6 | 100 |
| barc_seeded | variant_gepa | 0.090 | 0.050 | 0.040 | 0.3428 | [-0.020, 0.110] | 7 | 3 | 100 |
| barc_seeded | variant_minimal | 0.090 | 0.140 | -0.050 | 0.3017 | [-0.120, 0.030] | 5 | 10 | 100 |
| barc_seeded | variant_parse | 0.090 | 0.050 | 0.040 | 0.3865 | [-0.020, 0.120] | 8 | 4 | 100 |
| barc_seeded | variant_structured_routine | 0.090 | 0.100 | -0.010 | 1.0000 | [-0.070, 0.050] | 4 | 5 | 100 |
| barc_seeded | variant_typed_only | 0.090 | 0.080 | 0.010 | 1.0000 | [-0.060, 0.080] | 7 | 6 | 100 |
| barc_synthetic | colbert_rerank | 0.030 | 0.180 | -0.150 | 3.01e-04 | [-0.230, -0.080] | 0 | 15 | 100 |
| barc_synthetic | corpus_hipporag_init | 0.030 | 0.060 | -0.030 | 0.4497 | [-0.090, 0.020] | 2 | 5 | 100 |
| barc_synthetic | empty_start | 0.030 | 0.110 | -0.080 | 0.0269 | [-0.140, -0.020] | 1 | 9 | 100 |
| barc_synthetic | flat_topk | 0.030 | 0.240 | -0.210 | 1.19e-04 | [-0.300, -0.120] | 3 | 24 | 100 |
| barc_synthetic | graph_traversal | 0.030 | 0.200 | -0.170 | 8.49e-04 | [-0.260, -0.080] | 3 | 20 | 100 |
| barc_synthetic | graphrag | 0.030 | 0.300 | -0.270 | 3.02e-06 | [-0.370, -0.180] | 2 | 29 | 100 |
| barc_synthetic | hand_coded_reorg | 0.030 | 0.120 | -0.090 | 0.0077 | [-0.150, -0.040] | 0 | 9 | 100 |
| barc_synthetic | hipporag2_filter | 0.030 | 0.100 | -0.070 | 0.0961 | [-0.140, 0.000] | 3 | 10 | 100 |
| barc_synthetic | hipporag_ppr | 0.030 | 0.210 | -0.180 | 1.44e-04 | [-0.260, -0.100] | 1 | 19 | 100 |
| barc_synthetic | hmem_hierarchical | 0.030 | 0.150 | -0.120 | 0.0033 | [-0.190, -0.050] | 1 | 13 | 100 |
| barc_synthetic | lightrag | 0.030 | 0.300 | -0.270 | 1.38e-06 | [-0.360, -0.180] | 1 | 28 | 100 |
| barc_synthetic | magma_multigraph | 0.030 | 0.030 | 0.000 | 1.0000 | [-0.040, 0.050] | 3 | 3 | 100 |
| barc_synthetic | mediq_policy | 0.030 | 0.100 | -0.070 | 0.0455 | [-0.140, -0.020] | 1 | 8 | 100 |
| barc_synthetic | one_shot | 0.030 | 0.050 | -0.020 | 0.4795 | [-0.050, 0.000] | 0 | 2 | 100 |
| barc_synthetic | pathrag | 0.030 | 0.300 | -0.270 | 3.02e-06 | [-0.370, -0.180] | 2 | 29 | 100 |
| barc_synthetic | raptor | 0.030 | 0.160 | -0.130 | 0.0036 | [-0.210, -0.060] | 2 | 15 | 100 |
| barc_synthetic | reorg_amem | 0.030 | 0.170 | -0.140 | 0.0022 | [-0.220, -0.060] | 2 | 16 | 100 |
| barc_synthetic | reorg_dreamcoder | 0.030 | 0.170 | -0.140 | 0.0012 | [-0.210, -0.070] | 1 | 15 | 100 |
| barc_synthetic | reorg_evolver | 0.030 | 0.120 | -0.090 | 0.0159 | [-0.150, -0.030] | 1 | 10 | 100 |
| barc_synthetic | reorg_lilo | 0.030 | 0.140 | -0.110 | 0.0055 | [-0.180, -0.050] | 1 | 12 | 100 |
| barc_synthetic | reorg_lrll | 0.030 | 0.130 | -0.100 | 0.0094 | [-0.170, -0.040] | 1 | 11 | 100 |
| barc_synthetic | reorg_memp | 0.030 | 0.210 | -0.180 | 1.44e-04 | [-0.260, -0.100] | 1 | 19 | 100 |
| barc_synthetic | reorg_memtree | 0.030 | 0.190 | -0.160 | 1.77e-04 | [-0.240, -0.090] | 0 | 16 | 100 |
| barc_synthetic | reorg_off | 0.030 | 0.130 | -0.100 | 0.0162 | [-0.170, -0.030] | 2 | 12 | 100 |
| barc_synthetic | reorg_on_graph_mdl_global_plateau | 0.030 | 0.120 | -0.090 | 0.0077 | [-0.150, -0.040] | 0 | 9 | 100 |
| barc_synthetic | reorg_on_trace_mdl_accretive_everyk | 0.030 | 0.110 | -0.080 | 0.0269 | [-0.140, -0.020] | 1 | 9 | 100 |
| barc_synthetic | reorg_sleepgate | 0.030 | 0.160 | -0.130 | 0.0036 | [-0.210, -0.060] | 2 | 15 | 100 |
| barc_synthetic | reorg_stitch | 0.030 | 0.080 | -0.050 | 0.1306 | [-0.110, 0.000] | 1 | 6 | 100 |
| barc_synthetic | rrmc_multi_round | 0.030 | 0.120 | -0.090 | 0.0159 | [-0.150, -0.030] | 1 | 10 | 100 |
| barc_synthetic | uot_entropy | 0.030 | 0.180 | -0.150 | 0.0023 | [-0.230, -0.070] | 3 | 18 | 100 |
| barc_synthetic | variant_cue_heavy | 0.030 | 0.220 | -0.190 | 1.75e-04 | [-0.270, -0.100] | 2 | 21 | 100 |
| barc_synthetic | variant_dspy_opt | 0.030 | 0.030 | 0.000 | 1.0000 | [-0.050, 0.050] | 3 | 3 | 100 |
| barc_synthetic | variant_free_text | 0.030 | 0.100 | -0.070 | 0.0233 | [-0.120, -0.030] | 0 | 7 | 100 |
| barc_synthetic | variant_gepa | 0.030 | 0.050 | -0.020 | 0.7237 | [-0.080, 0.030] | 3 | 5 | 100 |
| barc_synthetic | variant_minimal | 0.030 | 0.140 | -0.110 | 0.0055 | [-0.180, -0.050] | 1 | 12 | 100 |
| barc_synthetic | variant_parse | 0.030 | 0.050 | -0.020 | 0.6831 | [-0.070, 0.020] | 2 | 4 | 100 |
| barc_synthetic | variant_structured_routine | 0.030 | 0.100 | -0.070 | 0.0455 | [-0.130, -0.010] | 1 | 8 | 100 |
| barc_synthetic | variant_typed_only | 0.030 | 0.080 | -0.050 | 0.1306 | [-0.110, 0.000] | 1 | 6 | 100 |
| corpus_hipporag_init | empty_start | 0.060 | 0.110 | -0.050 | 0.2673 | [-0.120, 0.020] | 4 | 9 | 100 |
| corpus_hipporag_init | flat_topk | 0.060 | 0.240 | -0.180 | 0.0013 | [-0.280, -0.080] | 5 | 23 | 100 |
| corpus_hipporag_init | graph_traversal | 0.060 | 0.200 | -0.140 | 0.0108 | [-0.240, -0.040] | 6 | 20 | 100 |
| corpus_hipporag_init | graphrag | 0.060 | 0.300 | -0.240 | 8.00e-05 | [-0.340, -0.130] | 5 | 29 | 100 |
| corpus_hipporag_init | hand_coded_reorg | 0.060 | 0.120 | -0.060 | 0.1814 | [-0.130, 0.010] | 4 | 10 | 100 |
| corpus_hipporag_init | hipporag2_filter | 0.060 | 0.100 | -0.040 | 0.3865 | [-0.110, 0.030] | 4 | 8 | 100 |
| corpus_hipporag_init | hipporag_ppr | 0.060 | 0.210 | -0.150 | 0.0035 | [-0.240, -0.060] | 4 | 19 | 100 |
| corpus_hipporag_init | hmem_hierarchical | 0.060 | 0.150 | -0.090 | 0.0389 | [-0.160, -0.020] | 3 | 12 | 100 |
| corpus_hipporag_init | lightrag | 0.060 | 0.300 | -0.240 | 4.79e-05 | [-0.340, -0.140] | 4 | 28 | 100 |
| corpus_hipporag_init | magma_multigraph | 0.060 | 0.030 | 0.030 | 0.5050 | [-0.030, 0.090] | 6 | 3 | 100 |
| corpus_hipporag_init | mediq_policy | 0.060 | 0.100 | -0.040 | 0.4227 | [-0.120, 0.030] | 5 | 9 | 100 |
| corpus_hipporag_init | one_shot | 0.060 | 0.050 | 0.010 | 1.0000 | [-0.050, 0.070] | 5 | 4 | 100 |
| corpus_hipporag_init | pathrag | 0.060 | 0.300 | -0.240 | 2.68e-05 | [-0.340, -0.150] | 3 | 27 | 100 |
| corpus_hipporag_init | raptor | 0.060 | 0.160 | -0.100 | 0.0550 | [-0.190, -0.010] | 6 | 16 | 100 |
| corpus_hipporag_init | reorg_amem | 0.060 | 0.170 | -0.110 | 0.0291 | [-0.200, -0.030] | 5 | 16 | 100 |
| corpus_hipporag_init | reorg_dreamcoder | 0.060 | 0.170 | -0.110 | 0.0153 | [-0.180, -0.040] | 3 | 14 | 100 |
| corpus_hipporag_init | reorg_evolver | 0.060 | 0.120 | -0.060 | 0.2113 | [-0.130, 0.020] | 5 | 11 | 100 |
| corpus_hipporag_init | reorg_lilo | 0.060 | 0.140 | -0.080 | 0.0990 | [-0.170, 0.000] | 5 | 13 | 100 |
| corpus_hipporag_init | reorg_lrll | 0.060 | 0.130 | -0.070 | 0.1456 | [-0.150, 0.010] | 5 | 12 | 100 |
| corpus_hipporag_init | reorg_memp | 0.060 | 0.210 | -0.150 | 0.0035 | [-0.240, -0.060] | 4 | 19 | 100 |
| corpus_hipporag_init | reorg_memtree | 0.060 | 0.190 | -0.130 | 0.0088 | [-0.220, -0.040] | 4 | 17 | 100 |
| corpus_hipporag_init | reorg_off | 0.060 | 0.130 | -0.070 | 0.1213 | [-0.150, 0.010] | 4 | 11 | 100 |
| corpus_hipporag_init | reorg_on_graph_mdl_global_plateau | 0.060 | 0.120 | -0.060 | 0.2113 | [-0.140, 0.030] | 5 | 11 | 100 |
| corpus_hipporag_init | reorg_on_trace_mdl_accretive_everyk | 0.060 | 0.110 | -0.050 | 0.2673 | [-0.120, 0.020] | 4 | 9 | 100 |
| corpus_hipporag_init | reorg_sleepgate | 0.060 | 0.160 | -0.100 | 0.0442 | [-0.190, -0.010] | 5 | 15 | 100 |
| corpus_hipporag_init | reorg_stitch | 0.060 | 0.080 | -0.020 | 0.7728 | [-0.090, 0.050] | 5 | 7 | 100 |
| corpus_hipporag_init | rrmc_multi_round | 0.060 | 0.120 | -0.060 | 0.1814 | [-0.140, 0.010] | 4 | 10 | 100 |
| corpus_hipporag_init | uot_entropy | 0.060 | 0.180 | -0.120 | 0.0190 | [-0.210, -0.030] | 5 | 17 | 100 |
| corpus_hipporag_init | variant_cue_heavy | 0.060 | 0.220 | -0.160 | 0.0033 | [-0.250, -0.070] | 5 | 21 | 100 |
| corpus_hipporag_init | variant_dspy_opt | 0.060 | 0.030 | 0.030 | 0.4497 | [-0.020, 0.090] | 5 | 2 | 100 |
| corpus_hipporag_init | variant_free_text | 0.060 | 0.100 | -0.040 | 0.3865 | [-0.110, 0.030] | 4 | 8 | 100 |
| corpus_hipporag_init | variant_gepa | 0.060 | 0.050 | 0.010 | 1.0000 | [-0.050, 0.070] | 5 | 4 | 100 |
| corpus_hipporag_init | variant_minimal | 0.060 | 0.140 | -0.080 | 0.0990 | [-0.160, 0.000] | 5 | 13 | 100 |
| corpus_hipporag_init | variant_parse | 0.060 | 0.050 | 0.010 | 1.0000 | [-0.050, 0.080] | 6 | 5 | 100 |
| corpus_hipporag_init | variant_structured_routine | 0.060 | 0.100 | -0.040 | 0.3428 | [-0.100, 0.020] | 3 | 7 | 100 |
| corpus_hipporag_init | variant_typed_only | 0.060 | 0.080 | -0.020 | 0.7728 | [-0.090, 0.040] | 5 | 7 | 100 |
| empty_start | flat_topk | 0.110 | 0.240 | -0.130 | 0.0209 | [-0.230, -0.030] | 7 | 20 | 100 |
| empty_start | graph_traversal | 0.110 | 0.200 | -0.090 | 0.1096 | [-0.200, 0.010] | 8 | 17 | 100 |
| empty_start | graphrag | 0.110 | 0.300 | -0.190 | 5.32e-04 | [-0.290, -0.090] | 4 | 23 | 100 |
| empty_start | hand_coded_reorg | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.090, 0.070] | 8 | 9 | 100 |
| empty_start | hipporag2_filter | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.070, 0.100] | 10 | 9 | 100 |
| empty_start | hipporag_ppr | 0.110 | 0.210 | -0.100 | 0.0776 | [-0.200, 0.010] | 8 | 18 | 100 |
| empty_start | hmem_hierarchical | 0.110 | 0.150 | -0.040 | 0.4795 | [-0.120, 0.040] | 7 | 11 | 100 |
| empty_start | lightrag | 0.110 | 0.300 | -0.190 | 8.30e-04 | [-0.290, -0.090] | 5 | 24 | 100 |
| empty_start | magma_multigraph | 0.110 | 0.030 | 0.080 | 0.0614 | [0.010, 0.160] | 11 | 3 | 100 |
| empty_start | mediq_policy | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.070, 0.080] | 8 | 7 | 100 |
| empty_start | one_shot | 0.110 | 0.050 | 0.060 | 0.1489 | [-0.010, 0.130] | 9 | 3 | 100 |
| empty_start | pathrag | 0.110 | 0.300 | -0.190 | 0.0017 | [-0.300, -0.080] | 7 | 26 | 100 |
| empty_start | raptor | 0.110 | 0.160 | -0.050 | 0.3827 | [-0.140, 0.040] | 8 | 13 | 100 |
| empty_start | reorg_amem | 0.110 | 0.170 | -0.060 | 0.2864 | [-0.150, 0.030] | 8 | 14 | 100 |
| empty_start | reorg_dreamcoder | 0.110 | 0.170 | -0.060 | 0.2864 | [-0.150, 0.030] | 8 | 14 | 100 |
| empty_start | reorg_evolver | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.100, 0.080] | 9 | 10 | 100 |
| empty_start | reorg_lilo | 0.110 | 0.140 | -0.030 | 0.6276 | [-0.110, 0.050] | 7 | 10 | 100 |
| empty_start | reorg_lrll | 0.110 | 0.130 | -0.020 | 0.7893 | [-0.090, 0.050] | 6 | 8 | 100 |
| empty_start | reorg_memp | 0.110 | 0.210 | -0.100 | 0.0550 | [-0.180, -0.010] | 6 | 16 | 100 |
| empty_start | reorg_memtree | 0.110 | 0.190 | -0.080 | 0.1530 | [-0.180, 0.010] | 8 | 16 | 100 |
| empty_start | reorg_off | 0.110 | 0.130 | -0.020 | 0.8137 | [-0.100, 0.060] | 8 | 10 | 100 |
| empty_start | reorg_on_graph_mdl_global_plateau | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.100, 0.070] | 9 | 10 | 100 |
| empty_start | reorg_on_trace_mdl_accretive_everyk | 0.110 | 0.110 | 0.000 | 1.0000 | [-0.080, 0.080] | 8 | 8 | 100 |
| empty_start | reorg_sleepgate | 0.110 | 0.160 | -0.050 | 0.3588 | [-0.130, 0.040] | 7 | 12 | 100 |
| empty_start | reorg_stitch | 0.110 | 0.080 | 0.030 | 0.5791 | [-0.040, 0.100] | 8 | 5 | 100 |
| empty_start | rrmc_multi_round | 0.110 | 0.120 | -0.010 | 1.0000 | [-0.090, 0.060] | 8 | 9 | 100 |
| empty_start | uot_entropy | 0.110 | 0.180 | -0.070 | 0.1904 | [-0.160, 0.020] | 7 | 14 | 100 |
| empty_start | variant_cue_heavy | 0.110 | 0.220 | -0.110 | 0.0725 | [-0.210, 0.000] | 10 | 21 | 100 |
| empty_start | variant_dspy_opt | 0.110 | 0.030 | 0.080 | 0.0614 | [0.010, 0.150] | 11 | 3 | 100 |
| empty_start | variant_free_text | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.050, 0.070] | 6 | 5 | 100 |
| empty_start | variant_gepa | 0.110 | 0.050 | 0.060 | 0.2113 | [-0.020, 0.140] | 11 | 5 | 100 |
| empty_start | variant_minimal | 0.110 | 0.140 | -0.030 | 0.6276 | [-0.110, 0.040] | 7 | 10 | 100 |
| empty_start | variant_parse | 0.110 | 0.050 | 0.060 | 0.2113 | [-0.010, 0.140] | 11 | 5 | 100 |
| empty_start | variant_structured_routine | 0.110 | 0.100 | 0.010 | 1.0000 | [-0.070, 0.090] | 9 | 8 | 100 |
| empty_start | variant_typed_only | 0.110 | 0.080 | 0.030 | 0.5791 | [-0.050, 0.100] | 8 | 5 | 100 |
