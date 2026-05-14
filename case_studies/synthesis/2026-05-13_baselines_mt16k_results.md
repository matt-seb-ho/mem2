# Phase G-Lite Results - 2026-05-13

## Configuration
- Conditions: 8
- Seeds: 42, 43
- Problems per seed: 50
- Iters: 1
- Cache: disabled
- Max workers: 512
- Model: deepseek/deepseek-v4-flash
- Tracer: enabled
- Started UTC: 2026-05-14T01:57:26+00:00
- Wall time: 22.07 minutes
- Total LLM calls: 667
- Total spend: unknown

## Per-condition results

| Axis | Condition | Parity grade | n per seed | seed 42 | seed 43 | Mean | Std | LLM calls | Cost | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | flat_topk | baseline | 50 | 0.580 | 0.440 | 0.510 | 0.099 | 82 | unknown | OK |
| 1 | lightrag | reduced-but-honest | 50 | 0.500 | 0.440 | 0.470 | 0.042 | 87 | unknown | OK |
| 2 | accretive_prune | unknown | 50 | 0.520 | 0.480 | 0.500 | 0.028 | 83 | unknown | OK |
| 2 | reorg_off | baseline | 50 | 0.420 | 0.520 | 0.470 | 0.071 | 85 | unknown | OK |
| 3 | one_shot | baseline | 50 | 0.560 | 0.440 | 0.500 | 0.085 | 81 | unknown | OK |
| 4 | arcmemo_ps | baseline | 50 | 0.440 | 0.500 | 0.470 | 0.042 | 85 | unknown | OK |
| 5 | hand_coded_reorg | baseline | 50 | 0.400 | 0.460 | 0.430 | 0.042 | 84 | unknown | OK |
| 6 | empty_start | baseline | 50 | 0.480 | 0.500 | 0.490 | 0.014 | 80 | unknown | OK |

## Findings to inspect

- No failed condition-seed runs.
- No per-seed scores below 0.05 or above 0.95.

## Surface-tier footnotes

- No surface-tier rows detected in this aggregate.

## Per-run trace links

- Axis 1 `flat_topk` seed 42: [2026-05-14T01-57-26Z_flat_topk_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T01-57-26Z_flat_topk_n50_seed42_baselines-mt16k-2026-05-13/), score=0.580
- Axis 1 `flat_topk` seed 43: [2026-05-14T01-57-26Z_flat_topk_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T01-57-26Z_flat_topk_n50_seed43_baselines-mt16k-2026-05-13/), score=0.440
- Axis 1 `lightrag` seed 42: [2026-05-14T01-57-26Z_lightrag_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T01-57-26Z_lightrag_n50_seed42_baselines-mt16k-2026-05-13/), score=0.500
- Axis 1 `lightrag` seed 43: [2026-05-14T01-57-26Z_lightrag_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T01-57-26Z_lightrag_n50_seed43_baselines-mt16k-2026-05-13/), score=0.440
- Axis 2 `accretive_prune` seed 42: [2026-05-14T02-03-02Z_accretive_prune_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-03-02Z_accretive_prune_n50_seed42_baselines-mt16k-2026-05-13/), score=0.520
- Axis 2 `accretive_prune` seed 43: [2026-05-14T02-03-04Z_accretive_prune_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-03-04Z_accretive_prune_n50_seed43_baselines-mt16k-2026-05-13/), score=0.480
- Axis 2 `reorg_off` seed 42: [2026-05-14T01-57-26Z_reorg_off_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T01-57-26Z_reorg_off_n50_seed42_baselines-mt16k-2026-05-13/), score=0.420
- Axis 2 `reorg_off` seed 43: [2026-05-14T02-02-55Z_reorg_off_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-02-55Z_reorg_off_n50_seed43_baselines-mt16k-2026-05-13/), score=0.520
- Axis 3 `one_shot` seed 42: [2026-05-14T02-03-09Z_one_shot_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-03-09Z_one_shot_n50_seed42_baselines-mt16k-2026-05-13/), score=0.560
- Axis 3 `one_shot` seed 43: [2026-05-14T02-03-12Z_one_shot_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-03-12Z_one_shot_n50_seed43_baselines-mt16k-2026-05-13/), score=0.440
- Axis 4 `arcmemo_ps` seed 42: [2026-05-14T02-08-31Z_arcmemo_ps_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-08-31Z_arcmemo_ps_n50_seed42_baselines-mt16k-2026-05-13/), score=0.440
- Axis 4 `arcmemo_ps` seed 43: [2026-05-14T02-08-34Z_arcmemo_ps_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-08-34Z_arcmemo_ps_n50_seed43_baselines-mt16k-2026-05-13/), score=0.500
- Axis 5 `hand_coded_reorg` seed 42: [2026-05-14T02-08-38Z_hand_coded_reorg_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-08-38Z_hand_coded_reorg_n50_seed42_baselines-mt16k-2026-05-13/), score=0.400
- Axis 5 `hand_coded_reorg` seed 43: [2026-05-14T02-08-43Z_hand_coded_reorg_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-08-43Z_hand_coded_reorg_n50_seed43_baselines-mt16k-2026-05-13/), score=0.460
- Axis 6 `empty_start` seed 42: [2026-05-14T02-08-45Z_empty_start_n50_seed42_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-08-45Z_empty_start_n50_seed42_baselines-mt16k-2026-05-13/), score=0.480
- Axis 6 `empty_start` seed 43: [2026-05-14T02-14-02Z_empty_start_n50_seed43_baselines-mt16k-2026-05-13](case_studies/runs/2026-05-14T02-14-02Z_empty_start_n50_seed43_baselines-mt16k-2026-05-13/), score=0.500
