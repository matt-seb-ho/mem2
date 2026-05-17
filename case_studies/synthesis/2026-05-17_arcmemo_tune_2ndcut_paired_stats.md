# Paired Statistical Comparison - ArcMemo Tune 2nd Cut - 2026-05-17

## Methodology
- Comparisons are paired against empty_start only, as requested.
- McNemar test uses paired correct/wrong asymmetry with Yates continuity correction.
- Bootstrap 95% CIs use 10000 resamples over paired seed plus ARC problem UID units.
- Bonferroni correction across 5 variant-vs-empty_start comparisons: alpha_corrected=0.010.
- Significance marker `*` means McNemar p < alpha_corrected.

## Condition accuracy

| Condition | Mean acc | 95% CI | n |
|---|---:|---|---:|
| empty_start | 0.370 | [0.280, 0.470] | 100 |
| arcmemo_ps | 0.490 | [0.390, 0.590] | 100 |
| arcmemo_ps_v2_query | 0.350 | [0.260, 0.440] | 100 |
| arcmemo_ps_v3_k5 | 0.380 | [0.290, 0.480] | 100 |
| arcmemo_ps_v4_both | 0.420 | [0.320, 0.510] | 100 |
| arcmemo_ps_v3b_k3 | 0.390 | [0.290, 0.490] | 100 |

## Paired vs empty_start

| Variant | Variant acc | empty_start acc | Gap | Gap 95% CI | McNemar p | b | c | n | Significant |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| arcmemo_ps | 0.490 | 0.370 | +12.0pp | [+4.0pp, +21.0pp] | 0.0139 | 16 | 4 | 100 |  |
| arcmemo_ps_v2_query | 0.350 | 0.370 | -2.0pp | [-12.0pp, +8.0pp] | 0.8383 | 11 | 13 | 100 |  |
| arcmemo_ps_v3_k5 | 0.380 | 0.370 | +1.0pp | [-8.0pp, +10.0pp] | 1.0000 | 11 | 10 | 100 |  |
| arcmemo_ps_v4_both | 0.420 | 0.370 | +5.0pp | [-5.0pp, +15.0pp] | 0.4237 | 15 | 10 | 100 |  |
| arcmemo_ps_v3b_k3 | 0.390 | 0.370 | +2.0pp | [-8.0pp, +12.0pp] | 0.8383 | 13 | 11 | 100 |  |
