# On-Policy Concept Induction — Results Summary

**TL;DR:** A fully model-authored (on-policy, deepseek-v4-flash) ArcMemo concept library
improves dsv4f over a compute-normalized vanilla baseline on ARC-AGI-1 `eval_100`:
**58.0 → 62.0** strict-solved mean (baseline → on-policy induced library). With reselection
it reaches **64**, matching the paper's BARC-derived 270-concept library (64) despite being
~5× smaller and built entirely from the model's own solves. The pass-1 view (perfectly
compute-normalized, exactly 100 solve calls) shows the same gain (46 → 51 → 55), ruling out
a retry-count confound.

> Integrity note: earlier drafts of this file contained wrong numbers (some written before
> runs finished, one mistyped from memory). Those were corrected. EVERY number below was
> read from the on-disk `summary.json` of a completed run (single authoritative pass); pass-1
> recomputed from each run's `iteration_1/solution_trees.json`. Run dirs listed for audit.

## Headline numbers (eval_100, ARC-AGI-1; strict = all test pairs correct)

| condition | library | strict /100 | pass-1 /100 | run dir(s) |
|---|---|---|---|---|
| baseline — vanilla, no memory | — | 57, 59 (mean **58.0**) | 46, 46 (46.0) | eval100_baseline/7f6af99c7234, _rep2/83e80339b1a0 |
| on-policy induced | induced_concepts_v1 (55) | 63, 61 (mean **62.0**) | 53, 49 (51.0) | eval100_arcmemo/5286d74e39b2, _rep2/86d94e24c8e2 |
| on-policy + reselection | induced_concepts_v1 (55) | **64** | 55 | eval100_arcmemo_reselect/cd6c4624f8ea |
| paper library (reference) | compressed_v1 (270) | **64** | 55 | eval100_paperlib/4bbcb1af4941 |

- **Induced library beats baseline by +4 strict (mean 58.0→62.0), +5 pass-1 (46→51).**
  Direction is stable across two independent samples each (cache off): both arcmemo samples
  (63, 61) exceed both baseline samples (57, 59).
- **Reselection** is the strongest induced variant (64 strict, 55 pass-1), edging plain
  selection (mean 62) — but it is a single sample, so treat the reselection-vs-plain delta
  as suggestive, not established. Mechanism verified firing (prior-attempt block +
  `<reflection>` present in the reselection prompts; `iteration_*/reselect_concepts/`
  artifacts written).
- **Paper's 270-concept library (64)** ties on-policy+reselection (64) and edges plain
  on-policy selection (mean 62): the model-authored library is competitive with the
  human-BARC-derived one at ~⅕ the size (55 vs 270 concepts).

## Compute normalization (as required)

All eval runs use the SAME model (deepseek-v4-flash, official DeepSeek API), SAME parallel
attempts `n=1`, and SAME retry budget `max_passes=3` with `train` criterion. Identical
`inference_engine.gen_cfg` across configs. The concept selection / reselection LLM calls are
inherent method overhead (extra *requests*) and do **not** change `n` or the retry budget —
solve attempts per puzzle are identical across conditions. The **pass-1** column is the
strictest normalization: exactly one solve call per puzzle (100 total), no retries, zero
asymmetry — and the induced library still wins (46 → 51 → 55).

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
- **Small effect, few samples.** +4 strict (mean of 2) is real and directionally stable
  (both induced > both baseline), but run-to-run noise is ~±2 and we have 2 samples/condition.
  More seeds would tighten it / establish significance.
- **Reselection promising but single-sample.** 64 vs plain mean 62; needs replication. The
  digest currently injects prior code — an LLM-summarized exploration digest may help more.
- **Stage C semantic merging is partial.** dsv4f is conservative at free-form grouping, so it
  needed lexical normalization + iterative reduce + a forceful critique pass; some
  near-synonyms across different wordings may still coexist. A cleaner library could help.
- **Token accounting:** selector LLM calls add requests (baseline ~198, arcmemo ~283–389,
  paperlib ~257); we normalize on n + retries (per spec), not tokens. Tokens-only, no $ (D0).
- Not yet: n>1 / top_k sweeps; synthesizing the freq=1 appendix concepts; multiple eval seeds.

## One-time offline build cost (amortized, tokens only)
- Seed solve (160 puzzles): ~1.08M in / 2.85M out, 255 reqs.
- Induction A–D: ~0.33M in / 0.69M out, ~350 reqs.
