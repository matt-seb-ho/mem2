# On-Policy Concept Induction — Results Summary

**TL;DR (5 independent samples per condition; all numbers from
`docs/onpolicy_variance_stats.json`):** A fully model-authored (on-policy,
deepseek-v4-flash) ArcMemo concept library shows a **small, directionally-positive but
not-statistically-significant** improvement over a compute-normalized vanilla baseline on
ARC-AGI-1 `eval_100`. Full-run strict mean **60.4 → 61.4 (+1.0, Welch p=0.44, NOT
significant)**; first-attempt mean **46.2 → 48.8 (+2.6, p=0.093 two-tailed / 0.046
one-tailed, marginal)**. The most robust effect is the induced library's **lower variance**
(strict sd 1.14 vs baseline 2.41) — memory makes dsv4f more *consistent*. The induced
55-concept library ≥ the paper's 270-concept BARC library in strict mean (61.4 vs 59.4) at
~⅕ the size. Reselection adds nothing on average. Critically, the **union of puzzles solved
across 5 samples is ≈identical (69–71) for all conditions** — memory shifts *which* puzzles
get solved and how consistently, NOT the set of solvable puzzles.

> Integrity note: this metric is noisy at this scale and my running estimate moved as
> samples accumulated — a 2-sample preview showed +4 (58→62), a 3-sample read +1.7, and an
> intermediate draft even mis-stated the n=5 numbers (59.0→61.8/p=0.02) before the rep4/rep5
> runs were read back. The n=5 table below is the authoritative, disk-verified result. EVERY
> number was read from on-disk `summary.json`; pass-1 recomputed from
> `iteration_1/solution_trees.json`; p-values via a pure-Python Welch t-test (scipy absent).

## Headline numbers (eval_100, ARC-AGI-1; strict = all test pairs correct; n=5 samples each)

| condition | library | strict (5 samples) | strict mean ± sd | pass-1 (5 samples) | pass-1 mean ± sd | oracle∪5 |
|---|---|---|---|---|---|---|
| baseline — no memory | — | 57, 59, 63, 62, 61 | **60.4 ± 2.41** | 46, 46, 49, 45, 45 | 46.2 ± 1.64 | 70 |
| on-policy induced | induced (55) | 63, 61, 60, 61, 62 | **61.4 ± 1.14** | 53, 49, 48, 47, 47 | 48.8 ± 2.49 | 70 |
| on-policy + reselection | induced (55) | 64, 59, 60, 61, 58 | 60.4 ± 2.30 | 55, 47, 47, 43, 50 | 48.4 ± 4.45 | 71 |
| paper library (reference) | compressed_v1 (270) | 64, 60, 59, 58, 56 | 59.4 ± 2.97 | 55, 47, 45, 41, 46 | 46.8 ± 5.12 | 69 |

- **Width = independent attempts/puzzle = n = 1; depth = retries = max_passes = 3** (train
  criterion), identical across all conditions. strict = full 3-pass run; pass-1 = first
  attempt only (exactly 100 solve calls — the strictest compute normalization). oracle∪5 =
  puzzles solved by ANY of the 5 samples (pass@5 upper bound).
- **Significance (Welch's t, baseline vs on-policy induced):** strict +1.0, t=0.84, df≈5.7,
  **p=0.44 two-tailed — NOT significant**; pass-1 +2.6, t=1.95, df≈6.9, **p=0.093 two-tailed
  (0.046 one-tailed) — marginal**. pass-1 is the cleaner metric (removes retry stochasticity)
  and is the only place the gain approaches significance.
- **Lower variance is the most robust effect:** induced strict sd 1.14 vs baseline 2.41 — the
  induced library never scored below 60; baseline ranged 57–63.
- **oracle∪5 ≈ 69–71 for ALL conditions:** memory does NOT expand the set of solvable
  eval_100 puzzles (pass@5 ceiling) — its modest effect is solving more of that fixed set per
  run.
- **on-policy induced (55) ≥ paper lib (270)** in strict mean (61.4 vs 59.4) at ~⅕ the size,
  though within noise. Reselection (60.4) adds nothing over plain selection (61.4) on average.

## Compute normalization (as required)

All eval runs use the SAME model (deepseek-v4-flash, official DeepSeek API), SAME width
(`n=1` independent attempts/puzzle), and SAME depth (`max_passes=3`, `train` retry
criterion). Identical `inference_engine.gen_cfg` across configs. The concept selection /
reselection LLM calls are inherent method overhead (extra *requests*) and do **not** change
width or depth — solve attempts per puzzle are identical across conditions. The **pass-1**
column removes even that ambiguity: exactly one solve call per puzzle, no retries.

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

# 3. eval_100 (compute-normalized; 5 samples/condition via rep2..rep5, ignore_cache:true)
python -m mem2.cli.run --config configs/experiments/eval100_baseline.yaml      # + _rep2.._rep5
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo.yaml       # + _rep2.._rep5
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_reselect.yaml
python -m mem2.cli.run --config configs/experiments/eval100_paperlib.yaml      # reference

# 4. per-attempt records for ensembling + recompute stats
python scripts/extract_attempts.py --glob 'eval100_*' --out outputs/_runs/eval100_all_attempt_records.jsonl
```

## Honest caveats / next steps
- **At n=5 the strict-solve gain is small and NOT significant** (+1.0, p=0.44). pass-1 (+2.6)
  is marginal (p=0.093 two-tailed / 0.046 one-tailed). The clearest effect is the induced
  library's lower variance (strict sd 1.14 vs 2.41). A confident strict-metric win would need
  more samples and/or a higher-signal setup (width>1, per-puzzle paired McNemar).
- **oracle∪5 ≈ equal across conditions (69–71):** memory shifts which puzzles get solved and
  how consistently, not the solvable-set size (pass@5 ceiling) at this scale.
- **Reselection did not help on average** (strict 60.4 vs plain 61.4). Mechanism is verified
  firing; it may still help with a better (LLM-summarized) exploration digest — current
  evidence is null.
- **Stage C semantic merging is partial.** dsv4f is conservative at free-form grouping, so it
  needed lexical normalization + iterative reduce + a forceful critique pass; some
  near-synonyms across different wordings may still coexist.
- **Token accounting:** selector LLM calls add requests; we normalize on width + depth (per
  spec), not tokens. Tokens-only, no $ (decision D0).
- Not yet: width>1 (n>1) / top_k sweeps; synthesizing the freq=1 appendix concepts; n≥8–10
  strict samples; per-puzzle paired McNemar on pass-1.

## Per-attempt / per-retry data (for future ensembling)
`scripts/extract_attempts.py` flattens every run's solution trees to
`<run_dir>/attempt_records.jsonl` (one row per puzzle × pass × branch × thread × step, with
train/test correctness + completion). Combined across all 20 eval runs:
`outputs/_runs/eval100_all_attempt_records.jsonl` (8956 rows). Lets us recompute pass@k,
majority-vote, oracle/any-correct, first-correct, and per-retry curves later without re-running.
- cmd: `python scripts/extract_attempts.py --glob 'eval100_*' --out outputs/_runs/eval100_all_attempt_records.jsonl`
