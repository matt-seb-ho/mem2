# Baselines mt16k Protocol Check - v1 vs v2

## Configuration
- v1: Phase G-lite, max_tokens=2048, iters=1, n=50, seeds=42 and 43.
- v2 mini: baselines-only protocol check, max_tokens=16384, iters=1, n=50, seeds=42 and 43.
- Historical anchors from Phase A doc 50 used only for scale sanity: empty_start 76.5% and flat_topk 63.5%, both mt16k with iters=3.

## Verdict
- Average score shift across the 8-condition mini-rerun: +31.5 percentage points.
- Median score shift across the 8-condition mini-rerun: +34.0 percentage points.
- The mt16k fix is verified for magnitude: all eight mini-rerun means land in the 43-51% range, versus v1 means in the 11-30% range.
- The remaining gap to the Phase A anchors is expected because this mini-rerun uses iters=1, while the historical anchors used iters=3.

## Comparison

| Condition | Axis | v1 mt2048 mean | v2 mt16k mean | Gap | Matches Phase A historical? |
|---|---:|---:|---:|---:|---|
| `empty_start` | 6 | 11.0% | 49.0% | +38.0 pp | Phase A mt16k iters=3: 76.5%. Mini is 49.0%, between v1 and iters=3 anchor. |
| `arcmemo_ps` | 4 | 11.0% | 47.0% | +36.0 pp | No direct Phase A anchor in brief. Magnitude shift is consistent with mt16k fix. |
| `flat_topk` | 1 | 24.0% | 51.0% | +27.0 pp | Phase A mt16k iters=3: 63.5%. Mini is 51.0%, between v1 and iters=3 anchor. |
| `one_shot` | 3 | 5.0% | 50.0% | +45.0 pp | No direct Phase A anchor in brief. Magnitude shift is consistent with mt16k fix. |
| `reorg_off` | 2 | 13.0% | 47.0% | +34.0 pp | No direct Phase A anchor in brief. Magnitude shift is consistent with mt16k fix. |
| `hand_coded_reorg` | 5 | 12.0% | 43.0% | +31.0 pp | No direct Phase A anchor in brief. Magnitude shift is consistent with mt16k fix. |
| `lightrag` | 1 | 30.0% | 47.0% | +17.0 pp | No direct Phase A anchor in brief. Magnitude shift is consistent with mt16k fix. |
| `accretive_prune` | 2 | 26.0% | 50.0% | +24.0 pp | No direct Phase A anchor in brief. Magnitude shift is consistent with mt16k fix. |

## Per-seed v2 scores

| Condition | seed 42 | seed 43 |
|---|---:|---:|
| `empty_start` | 48.0% | 50.0% |
| `arcmemo_ps` | 44.0% | 50.0% |
| `flat_topk` | 58.0% | 44.0% |
| `one_shot` | 56.0% | 44.0% |
| `reorg_off` | 42.0% | 52.0% |
| `hand_coded_reorg` | 40.0% | 46.0% |
| `lightrag` | 50.0% | 44.0% |
| `accretive_prune` | 52.0% | 48.0% |

## Trace inventory

- Baselines mt16k aggregate: `case_studies/synthesis/2026-05-13_baselines_mt16k_results.md`.
- Raw per-call traces are under `case_studies/runs/*baselines-mt16k-2026-05-13/`.
- Compact run metadata was committed for all 16 condition-seed runs.
