# Phase G-Lite Results - 2026-05-13

## Configuration
- Conditions: 6
- Seeds: 42, 43
- Problems per seed: 50
- Iters: 1
- Cache: disabled
- Max workers: 512
- Model: deepseek/deepseek-v4-flash
- Tracer: enabled
- Started UTC: 2026-05-17T15:10:47+00:00
- Wall time: 16.12 minutes
- Total LLM calls: 502
- Total spend: unknown

## Per-condition results

| Axis | Condition | Parity grade | n per seed | seed 42 | seed 43 | Mean | Std | LLM calls | Cost | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | arcmemo_ps | baseline | 50 | 0.480 | 0.500 | 0.490 | 0.014 | 83 | unknown | OK |
| 4 | arcmemo_ps_v4_both | tuning-experimental | 50 | 0.420 | 0.420 | 0.420 | 0.000 | 82 | unknown | OK |
| 4 | arcmemo_ps_v3b_k3 | tuning-experimental | 50 | 0.400 | 0.380 | 0.390 | 0.014 | 83 | unknown | OK |
| 4 | arcmemo_ps_v3_k5 | tuning-experimental | 50 | 0.300 | 0.460 | 0.380 | 0.113 | 84 | unknown | OK |
| 4 | arcmemo_ps_v2_query | tuning-experimental | 50 | 0.340 | 0.360 | 0.350 | 0.014 | 83 | unknown | OK |
| 6 | empty_start | baseline | 50 | 0.380 | 0.360 | 0.370 | 0.014 | 87 | unknown | OK |

## Findings to inspect

- No failed condition-seed runs.
- No per-seed scores below 0.05 or above 0.95.

## Surface-tier footnotes

- No surface-tier rows detected in this aggregate.

## Per-run trace links

- Axis 4 `arcmemo_ps` seed 42: [2026-05-17T15-10-47Z_arcmemo_ps_n50_seed42_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-10-47Z_arcmemo_ps_n50_seed42_arcmemo-tune-2ndcut-2026-05-17/), score=0.480
- Axis 4 `arcmemo_ps` seed 43: [2026-05-17T15-10-47Z_arcmemo_ps_n50_seed43_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-10-47Z_arcmemo_ps_n50_seed43_arcmemo-tune-2ndcut-2026-05-17/), score=0.500
- Axis 4 `arcmemo_ps_v2_query` seed 42: [2026-05-17T15-10-47Z_arcmemo_ps_v2_query_n50_seed42_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-10-47Z_arcmemo_ps_v2_query_n50_seed42_arcmemo-tune-2ndcut-2026-05-17/), score=0.340
- Axis 4 `arcmemo_ps_v2_query` seed 43: [2026-05-17T15-10-47Z_arcmemo_ps_v2_query_n50_seed43_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-10-47Z_arcmemo_ps_v2_query_n50_seed43_arcmemo-tune-2ndcut-2026-05-17/), score=0.360
- Axis 4 `arcmemo_ps_v3_k5` seed 42: [2026-05-17T15-10-47Z_arcmemo_ps_v3_k5_n50_seed42_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-10-47Z_arcmemo_ps_v3_k5_n50_seed42_arcmemo-tune-2ndcut-2026-05-17/), score=0.300
- Axis 4 `arcmemo_ps_v3_k5` seed 43: [2026-05-17T15-16-06Z_arcmemo_ps_v3_k5_n50_seed43_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-16-06Z_arcmemo_ps_v3_k5_n50_seed43_arcmemo-tune-2ndcut-2026-05-17/), score=0.460
- Axis 4 `arcmemo_ps_v3b_k3` seed 42: [2026-05-17T15-16-11Z_arcmemo_ps_v3b_k3_n50_seed42_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-16-11Z_arcmemo_ps_v3b_k3_n50_seed42_arcmemo-tune-2ndcut-2026-05-17/), score=0.400
- Axis 4 `arcmemo_ps_v3b_k3` seed 43: [2026-05-17T15-16-11Z_arcmemo_ps_v3b_k3_n50_seed43_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-16-11Z_arcmemo_ps_v3b_k3_n50_seed43_arcmemo-tune-2ndcut-2026-05-17/), score=0.380
- Axis 4 `arcmemo_ps_v4_both` seed 42: [2026-05-17T15-16-09Z_arcmemo_ps_v4_both_n50_seed42_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-16-09Z_arcmemo_ps_v4_both_n50_seed42_arcmemo-tune-2ndcut-2026-05-17/), score=0.420
- Axis 4 `arcmemo_ps_v4_both` seed 43: [2026-05-17T15-16-10Z_arcmemo_ps_v4_both_n50_seed43_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-16-10Z_arcmemo_ps_v4_both_n50_seed43_arcmemo-tune-2ndcut-2026-05-17/), score=0.420
- Axis 6 `empty_start` seed 42: [2026-05-17T15-21-31Z_empty_start_n50_seed42_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-21-31Z_empty_start_n50_seed42_arcmemo-tune-2ndcut-2026-05-17/), score=0.380
- Axis 6 `empty_start` seed 43: [2026-05-17T15-21-32Z_empty_start_n50_seed43_arcmemo-tune-2ndcut-2026-05-17](case_studies/runs/2026-05-17T15-21-32Z_empty_start_n50_seed43_arcmemo-tune-2ndcut-2026-05-17/), score=0.360
