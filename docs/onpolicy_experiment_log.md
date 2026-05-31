# On-Policy Concept Induction — Experiment Log

Running log of progress, decisions, commands, outputs, and results for the on-policy
(dsv4f, official DeepSeek API) ArcMemo concept-induction effort. Plan:
`docs/onpolicy_concept_induction_plan.md`. Headline results: `docs/onpolicy_results_summary.md`.

**Goal:** demonstrate that a fully model-authored concept library improves dsv4f over a
compute-normalized vanilla (no-memory) baseline on the ARC-AGI-1 `eval_100` subset.

**Compute normalization (hard requirement):** baseline and arcmemo runs use the SAME model
(dsv4f), SAME parallel attempts `n`, and SAME retry budget (`max_passes`, `retry_criterion`).
Memory selection/reselection LLM calls are method overhead (reported as extra requests), but
do not change `n` or `max_passes`.

---

## 2026-05-31 — Phases 0–2 + Stage A/B
- Provider wired (`deepseek-v4-flash` on official `Provider.DEEPSEEK`); preflight clean to
  concurrency 128, using 64. (`scripts/deepseek_preflight.py`)
- **Phase 1 vanilla solve on 160 BARC seeds:** 123/160 strict-solved.
  - cmd: `python -m mem2.cli.run --config configs/experiments/onpolicy_solve_barc.yaml`
  - out: `outputs/_runs/onpolicy_solve_barc/0fbc37e9d7d2/`
- **Phase 2 harvest:** 120 train-correct (117 also test-correct).
  - cmd: `python scripts/harvest_solves.py --run-dir outputs/_runs/onpolicy_solve_barc/0fbc37e9d7d2 --train-only`
  - out: `.../induction/solved_seeds.json`
- **Stage A/B (per-puzzle pseudocode + free-form tags):**
  - cmd: `python scripts/induce_library.py --solves .../solved_seeds.json --stage a` (then `--stage b`)
  - out: `.../induction/stageA_pseudocode.json` (120/120), `.../induction/stageB_tags.json`
    (675 tags, 605 unique).
  - takeaway: heavy synonymy across exact strings → LLM semantic grouping needed for Stage C.

## 2026-05-31 — Stage C (unification) + Stage D (typed synthesis)
**Stage C — corpus-global vocabulary unification (LLM map/reduce + critique).**
- Key finding: dsv4f is VERY conservative at free-form merging; a single-shot reduce over a
  flat 583-item list did 0 merges (the predicted "giant call is lossy" failure). Fixes:
  (1) lexical surface-form normalization in `aggregate_tags` (singular/plural, hyphens,
  stopwords — NOT semantic clustering, so the LLM still owns semantic grouping);
  (2) iterative agglomerative reduce in bounded chunks with alternating sort so cross-shard
  synonyms meet over rounds; (3) the critique/merge loop runs on the PRIMARY pool only
  (~57 concepts) with a forceful merge prompt, up to 3 rounds.
- cmd: `python scripts/induce_library.py --stage c --out-dir <IND>`
- out: `<IND>/stageC_vocab.json` — **57 primary (freq≥2)**, 479 appendix.
- takeaway: needs mechanical normalization + iterative reduce + focused critique to get usable
  dedup out of dsv4f; pure single-call grouping fails. Cross-wording semantic merges remain
  partial (acceptable; the retriever tolerates mild redundancy).

**Stage D — per-concept typed synthesis → ConceptMemory schema.**
- cmd: `python scripts/induce_library.py --stage d --out-dir <IND>`
- out: `data/arc_agi/concept_memory/induced_concepts_v1.json` (+ copy in `<IND>/`).
- result: **55 typed concepts (19 structures, 36 routines), 3 custom types.** Loads cleanly
  into `ConceptMemory`; each has description/cues/implementation/parameters/used_in.
- Stage A–D token cost recorded in `<IND>/usage/stage_*.json`.

### Induced library
`data/arc_agi/concept_memory/induced_concepts_v1.json` — 55 concepts, fully dsv4f-authored
from its own 120 correct seed solves. On-policy analogue of the paper's BARC-derived
`compressed_v1.json` (270 concepts).

## 2026-05-31 — Phase 4 eval_100 — VERIFIED RESULTS (single authoritative disk read)

All numbers below were read from each completed run's `summary.json`; pass-1 recomputed from
`iteration_1/solution_trees.json`. Two independent samples (cache off, `ignore_cache: true`)
for baseline and arcmemo.

| condition | library | strict/100 | pass1/100 | run dir |
|---|---|---|---|---|
| baseline (no memory)   | —                  | 57, 59 (μ58.0) | 46, 46 | eval100_baseline/7f6af99c7234, _rep2/83e80339b1a0 |
| on-policy induced      | induced_v1 (55)    | 63, 61 (μ62.0) | 53, 49 | eval100_arcmemo/5286d74e39b2, _rep2/86d94e24c8e2 |
| on-policy + reselect   | induced_v1 (55)    | 64             | 55     | eval100_arcmemo_reselect/cd6c4624f8ea |
| paper lib (reference)  | compressed_v1 (270)| 64             | 55     | eval100_paperlib/4bbcb1af4941 |

**Run commands**
```
python -m mem2.cli.run --config configs/experiments/eval100_baseline.yaml
python -m mem2.cli.run --config configs/experiments/eval100_baseline_rep2.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_rep2.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_reselect.yaml
python -m mem2.cli.run --config configs/experiments/eval100_paperlib.yaml
```

**Takeaways**
- ✅ The on-policy, fully model-authored library improves dsv4f over the compute-normalized
  vanilla baseline: strict mean **58.0 → 62.0** (+4), pass-1 **46 → 51** (+5). Direction is
  stable across 2 samples each (both arcmemo 63,61 > both baseline 57,59). Pass-1 (exactly 100
  solve calls, zero asymmetry) confirms it is not a retry-count artifact.
- ✅ The 55-concept on-policy library is competitive with the paper's 270-concept BARC library
  (μ62 / 64-with-reselect vs 64) at ~1/5 the size.
- ⚠️ Reselection-with-prior-context is the best induced variant (64 strict, 55 pass-1) but is a
  single sample; the reselection-vs-plain delta is suggestive, not established. Mechanism
  verified firing (prior-attempt block + `<reflection>` in reselect prompts;
  `iteration_*/reselect_concepts/` artifacts present).
- Normalization check: solve attempts/puzzle identical across conditions; memory runs issue
  extra *selection* requests (baseline 198, arcmemo 283/389, reselect 264, paperlib 257) —
  method overhead, reported, not a change to n or retry budget. The pass-1 column removes even
  that ambiguity.

### Integrity / process lessons
- Twice during this session I reported eval numbers that did not match disk: once by reading
  `summary.json` before runs finished (caused by batching launch+wait+read into one parallel
  tool block), and once by typing means from memory instead of from disk. Both were caught by
  re-reading disk and corrected. The numbers in this table are the disk-verified ones.
- Rules adopted: (1) never batch a run's launch + wait + summary-read into parallel tool
  calls — run to completion, then read; (2) never transcribe results from memory — always
  read `summary.json`.

## 2026-05-31 — Variance experiment (3 independent samples per condition)

**Why:** confirm the +4 strict trend holds with error bars; a collaborator is seeing
slightly different numbers, so we need mean ± stdev over independent samples to compare.

**Width / depth (identical across ALL conditions — the compute-normalization invariant):**
- **Width = independent parallel attempts per puzzle = `n` = 1.**
- **Depth = retries within an attempt = `max_passes` = 3**, retry criterion = train-correct
  (1 initial solve + up to 2 retries; retry only while train pairs fail).
- Same model (deepseek-v4-flash, official API), same `inference_engine.gen_cfg`
  (temperature 0.3, max_tokens 16384, top_p 1.0) across every condition.
- Memory conditions additionally make concept selection / reselection LLM calls — these are
  method overhead (extra *requests*), they do NOT change width or depth (solve attempts/puzzle
  are identical to baseline).

**Independence:** each sample uses `ignore_cache: true` so the DeepSeek response cache is
bypassed (DeepSeek ignores `seed` and caches by default; without this, repeats would return
identical completions). Temperature 0.3 → genuinely independent samples. NOTE: the original
rep1 configs (eval100_baseline.yaml, eval100_arcmemo.yaml, eval100_arcmemo_reselect.yaml,
eval100_paperlib.yaml) used `ignore_cache: false`; they were the FIRST run of each so the
cache was cold → effectively fresh samples. All NEW samples force cache off.

**Target: 3 independent samples per row.** Already have: baseline ×2, induced ×2, reselect
×1, paperlib ×1. Adding: baseline ×1, induced ×1, reselect ×2, paperlib ×2 (6 new runs).

**New configs** (all `ignore_cache: true`):
- `eval100_baseline_rep3.yaml`
- `eval100_arcmemo_rep3.yaml`
- `eval100_arcmemo_reselect_rep2.yaml`, `eval100_arcmemo_reselect_rep3.yaml`
- `eval100_paperlib_rep2.yaml`, `eval100_paperlib_rep3.yaml`

**Run commands**
```
python -m mem2.cli.run --config configs/experiments/eval100_baseline_rep3.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_rep3.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_reselect_rep2.yaml
python -m mem2.cli.run --config configs/experiments/eval100_arcmemo_reselect_rep3.yaml
python -m mem2.cli.run --config configs/experiments/eval100_paperlib_rep2.yaml
python -m mem2.cli.run --config configs/experiments/eval100_paperlib_rep3.yaml
```

**Results (3 independent samples each; read from disk; stats artifact
`outputs/_runs/eval100_variance_stats.json`):**

| condition | library | strict (3 samples) | strict mean ± sd | pass-1 (3 samples) | pass-1 mean ± sd | oracle∪3 | robust∩3 |
|---|---|---|---|---|---|---|---|
| baseline — no memory | — | 57, 59, 63 | **59.67 ± 3.06** | 46, 46, 49 | 47.0 ± 1.73 | 67 | 52 |
| on-policy induced | induced (55) | 63, 61, 60 | **61.33 ± 1.53** | 53, 49, 48 | 50.0 ± 2.65 | 67 | 54 |
| on-policy + reselection | induced (55) | 64, 59, 60 | 61.0 ± 2.65 | 55, 47, 47 | 49.67 ± 4.62 | 68 | 53 |
| paper lib (reference) | compressed_v1 (270) | 64, 60, 59 | 61.0 ± 2.65 | 55, 47, 45 | 49.0 ± 5.29 | 69 | 52 |

- strict = full run (depth 3); pass-1 = first attempt only (1 solve call/puzzle, perfect
  width/depth normalization). oracle∪3 = puzzles solved by ANY of the 3 samples (pass@3
  upper bound). robust∩3 = solved by ALL 3 samples.

**Honest takeaways (this is the corrected, 3-sample picture):**
- ⚠️ **The +4 strict gap from the 2-sample preview did NOT robustly hold.** With a 3rd
  sample, baseline drew a high run (63), lifting baseline mean to 59.67. The memory vs
  no-memory gap is now **+1.7 strict (59.67 → 61.33)**, which is WITHIN the baseline's
  ±3.06 stdev — i.e. not statistically distinguishable at n=3.
- The cleaner signal is **pass-1: +3.0 (47.0 → 50.0)** and **lower variance** for the
  induced library (sd 1.53 vs baseline 3.06). Memory makes dsv4f more *consistent* more than
  it raises the ceiling.
- **oracle∪3 is ~equal across all conditions (67–69):** memory does NOT meaningfully expand
  the SET of solvable eval_100 puzzles at this scale; it mostly shifts which get solved on a
  given run. This is an important, sobering finding.
- on-policy induced (55 concepts) ≈ paper lib (270) ≈ reselection — all ~61 mean. No
  condition separates from another beyond noise. Reselection did not help on average.
- **Bottom line:** at width=1 / depth=3 on eval_100 (100 puzzles), the strict-solve metric
  is too noisy (±3) to claim a clear memory win from 3 samples. Direction of the mean is
  still positive (induced > baseline on mean and much tighter), but it is not significant.
  More samples and/or a higher-signal setup (e.g. width>1, or per-puzzle paired analysis)
  are needed to make a confident claim.

**Per-attempt / per-retry data saved (for future ensembling):**
`scripts/extract_attempts.py` flattens every run's solution trees into
`<run_dir>/attempt_records.jsonl` (one row per puzzle×pass×branch×thread×step, with
train/test correctness + completion). Combined: `outputs/_runs/eval100_all_attempt_records.jsonl`
(5315 rows). Enables recomputing pass@k, majority-vote, oracle, first-correct, per-retry
curves without re-running.
- cmd: `python scripts/extract_attempts.py --glob 'eval100_*' --out outputs/_runs/eval100_all_attempt_records.jsonl`
