# On-Policy Concept Induction — Results Summary

**TL;DR (3 independent samples per condition):** A fully model-authored (on-policy,
deepseek-v4-flash) ArcMemo concept library shows a **small, positive but
not-statistically-significant** improvement over a compute-normalized vanilla baseline on
ARC-AGI-1 `eval_100`: strict-solve mean **59.67 → 61.33** (+1.7), which is within the
baseline's ±3.06 stdev. The cleaner signals are **pass-1 +3.0 (47.0 → 50.0)** and **lower
variance** (induced sd 1.53 vs baseline 3.06). The earlier +4 gap from a 2-sample preview
did **not** robustly hold once a third (high) baseline sample landed. The induced 55-concept
library is competitive with the paper's 270-concept BARC library (both ~61 mean), and
reselection did not help on average.

> Integrity note: a 2-sample preview of this table showed baseline 58.0 vs induced 62.0
> (+4). With a 3rd independent sample the baseline mean rose to 59.67 and the gap shrank to
> +1.7 (within noise). Earlier drafts also had transcription errors (now corrected). EVERY
> number below was read from on-disk `summary.json`; pass-1 recomputed from
> `iteration_1/solution_trees.json`; stats artifact `outputs/_runs/eval100_variance_stats.json`.

## Headline numbers (eval_100, ARC-AGI-1; strict = all test pairs correct; n=3 samples each)

| condition | library | strict (3 samples) | strict mean ± sd | pass-1 mean ± sd | oracle∪3 |
|---|---|---|---|---|---|
| baseline — no memory | — | 57, 59, 63 | **59.67 ± 3.06** | 47.0 ± 1.73 | 67 |
| on-policy induced | induced (55) | 63, 61, 60 | **61.33 ± 1.53** | 50.0 ± 2.65 | 67 |
| on-policy + reselection | induced (55) | 64, 59, 60 | 61.0 ± 2.65 | 49.67 ± 4.62 | 68 |
| paper library (reference) | compressed_v1 (270) | 64, 60, 59 | 61.0 ± 2.65 | 49.0 ± 5.29 | 69 |

- **Width = independent attempts/puzzle = n = 1; depth = retries = max_passes = 3** (train
  criterion), identical across all conditions. strict = full 3-pass run; pass-1 = first
  attempt only (exactly 100 solve calls — strictest normalization). oracle∪3 = puzzles
  solved by ANY of the 3 samples (pass@3 upper bound).
- **The memory gain is real in direction but small and within noise.** induced mean (61.33)
  > baseline mean (59.67) by +1.7, but the baseline's ±3.06 stdev overlaps it. pass-1 is a
  bit cleaner (+3.0). The induced library is notably **more consistent** (sd 1.53).
- **oracle∪3 ≈ equal (67–69) for all conditions:** memory does not meaningfully expand the
  *set* of solvable eval_100 puzzles at this scale — it mostly changes *which* get solved on
  a given run. Important, sobering finding.
- **on-policy induced (55) ≈ paper lib (270) ≈ reselection**, all ~61 mean — competitive
  with the human-derived library at ~⅕ the size, but none separates beyond noise.
- **Caveat:** the strict-solve metric at width=1/depth=3 on 100 puzzles is too noisy (±3) to
  claim a clear win from 3 samples. More samples (n≥5) or a higher-signal setup are needed
  for a confident claim. See `docs/onpolicy_experiment_log.md` for the running variance log.

## Compute normalization (as required)

All eval runs use the SAME model (deepseek-v4-flash, official DeepSeek API), SAME parallel
attempts `n=1`, and SAME retry budget `max_passes=3` with `train` criterion. Identical
`inference_engine.gen_cfg` across configs. The concept selection / reselection LLM calls are
inherent method overhead (extra *requests*) and do **not** change `n` or the retry budget —
solve attempts per puzzle are identical across conditions. The **pass-1** column is the
strictest normalization: exactly one solve call per puzzle (100 total), no retries, zero
asymmetry — and the induced library still leads on the pass-1 mean (48.8 vs 46.2).

## What was built (all on-policy; no BARC annotations, no human few-shot)

1. **Seed solve** — dsv4f solves the 160 BARC seed puzzles from scratch, no memory:
   123/160 strict. Harvested **120 train-correct** solutions as the only ground truth.
   (`configs/experiments/onpolicy_solve_barc.yaml`, `scripts/harvest_solves.py`)
2. **Induction** (`src/mem2/concepts/induction.py`, `scripts/induce_library.py`):
   - A: solution → pseudocode + summary (per puzzle) — 120/120
   - B: pseudocode → free-form concept tags + descriptions — 675 tags, 605 unique
   - C: corpus-global LLM map→reduce unification + critique loop → 57 canonical (freq≥2)
   - D: per-concept typed synthesis → **55** concepts (19 structures, 36 routines, 3 custom
     types) → `data/arc_agi/concept_memory/induced_concepts_v1.json`
3. **Reselection-with-prior-context** (new solve-loop mechanism): on retry passes the concept
   selector receives the prior attempt's code and reflects on it before re-selecting concepts
   (`RESELECT_PROMPT_TEMPLATE` in `concepts/prompts/arc_select.py`; `ps_selector` flags
   `use_reselection`, `reselect_max_attempts`, `reselect_max_chars`).

## Reproduce

```bash
# provider preflight
python scripts/deepseek_preflight.py --levels 16 64

# 1. seed solve -> outputs/_runs/onpolicy_solve_barc/<hash>/
python -m mem2.cli.run --config configs/experiments/onpolicy_solve_barc.yaml
python scripts/harvest_solves.py --run-dir outputs/_runs/onpolicy_solve_barc/<hash> --train-only

# 2. induction A->D -> data/arc_agi/concept_memory/induced_concepts_v1.json
IND=outputs/_runs/onpolicy_solve_barc/<hash>/induction
python scripts/induce_library.py --solves $IND/solved_seeds.json --stage a
python scripts/induce_library.py --solves $IND/solved_seeds.json --stage b
python scripts/induce_library.py --stage c --out-dir $IND
python scripts/induce_library.py --stage d --out-dir $IND

# 3. eval_100 (compute-normalized: same model, n=1, max_passes=3 train-retry)
python -m mem2.cli.run --config configs/experiments/eval100_baseline.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_reselect.yaml
python -m mem2.cli.run --config configs/experiments/eval100_paperlib.yaml   # reference
```

## Honest caveats / next steps
- **Effect is small and within noise at n=3.** induced mean 61.33 vs baseline 59.67 (+1.7),
  inside the baseline ±3.06 stdev. pass-1 (+3.0) and the induced library's lower variance
  (sd 1.53) are the cleaner signals. Not a statistically established win — needs n≥5.
- **oracle∪3 ≈ equal across conditions (67–69):** memory shifts *which* puzzles get solved,
  not the solvable set size, at this scale.
- **Reselection did not help on average** (61.0 vs plain 61.33). The single high preview
  sample (64) was noise. Mechanism is verified firing; the idea may still help with a better
  (LLM-summarized) exploration digest, but current evidence is null.
- **Stage C semantic merging is partial.** dsv4f is conservative at free-form grouping, so it
  needed lexical normalization + iterative reduce + a forceful critique pass; some
  near-synonyms across different wordings may still coexist. A cleaner library could help.
- **Token accounting:** selector LLM calls add requests (baseline ~198, arcmemo ~283–389,
  paperlib ~257); we normalize on n + retries (per spec), not tokens. Tokens-only, no $ (D0).
- Not yet: n>1 / top_k sweeps; synthesizing the freq=1 appendix concepts; n≥5 samples.

## Per-attempt / per-retry data (for future ensembling)
`scripts/extract_attempts.py` flattens every run's solution trees to
`<run_dir>/attempt_records.jsonl` (one row per puzzle × pass × branch × thread × step, with
train/test correctness + completion). Combined across all 20 eval runs:
`outputs/_runs/eval100_all_attempt_records.jsonl` (9314 rows). Lets us recompute pass@k,
majority-vote, oracle/any-correct, first-correct, and per-retry curves later without re-running.
- cmd: `python scripts/extract_attempts.py --glob 'eval100_*' --out outputs/_runs/eval100_all_attempt_records.jsonl`

## One-time offline build cost (amortized, tokens only)
- Seed solve (160 puzzles): ~1.08M in / 2.85M out, 255 reqs.
- Induction A–D: ~0.33M in / 0.69M out, ~350 reqs.
