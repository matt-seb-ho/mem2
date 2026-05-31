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
(8956 rows). Enables recomputing pass@k, majority-vote, oracle, first-correct, per-retry
curves without re-running.
- cmd: `python scripts/extract_attempts.py --glob 'eval100_*' --out outputs/_runs/eval100_all_attempt_records.jsonl`

## 2026-05-31 (overnight) — Extend to n=5 samples for significance

**Decision (autonomous):** the n=3 result is noise-limited (baseline ±3.06; induced gap
+1.7 within that). To resolve whether the positive mean gap is real, extend every condition
to 5 independent samples (add rep4, rep5). Same invariant: width n=1, depth max_passes=3
train-retry, ignore_cache:true, same model/gen_cfg. 8 new runs (baseline/induced/reselect/
paperlib × 2). Will recompute mean±sd, a Welch t-test (baseline vs induced) and a per-puzzle
paired McNemar on pass-1, then update tables + commit.

New configs: eval100_{baseline,arcmemo,arcmemo_reselect,paperlib}_rep{4,5}.yaml

## 2026-05-31 (overnight) — Variance extended to n=5 (FINAL, disk-verified)

Added rep4/rep5 for all four conditions (8 runs) so every row has 5 independent samples.
Same invariant: width n=1, depth max_passes=3 (train), ignore_cache:true, same model/gen_cfg.
Stats artifact: `outputs/_runs/eval100_variance_stats.json` (copy `docs/onpolicy_variance_stats.json`).
p-values via pure-Python Welch t-test (scipy not in env).

Run commands:
```
for c in baseline arcmemo arcmemo_reselect paperlib; do
  for r in rep4 rep5; do
    python -m mem2.cli.run --config configs/experiments/eval100_${c}_${r}.yaml
  done
done
```

| condition | library | strict (5 samples) | strict mean ± sd | pass-1 mean ± sd | oracle∪5 | robust∩5 |
|---|---|---|---|---|---|---|
| baseline — no memory | — | 57,59,63,62,61 | **60.4 ± 2.41** | 46.2 ± 1.64 | 70 | 50 |
| on-policy induced | induced (55) | 63,61,60,61,62 | **61.4 ± 1.14** | 48.8 ± 2.49 | 70 | 49 |
| on-policy + reselection | induced (55) | 64,59,60,61,58 | 60.4 ± 2.30 | 48.4 ± 4.45 | 71 | 50 |
| paper lib (reference) | compressed_v1 (270) | 64,60,59,58,56 | 59.4 ± 2.97 | 46.8 ± 5.12 | 69 | 47 |

**Welch t-test (baseline vs on-policy induced):**
- strict: +1.0 (60.4→61.4), t=0.84, df≈5.7, two-tailed **p=0.44 — NOT significant**.
- pass-1: +2.6 (46.2→48.8), t=1.95, df≈6.9, two-tailed **p=0.092** (one-tailed 0.046) — marginal.

**Conclusion (n=5, FINAL for this round):** The memory improvement is **real in direction but
small and not statistically established** at this scale. Full-run strict gain is +1.0
(p=0.44); first-attempt gain is +2.6 (marginal, p≈0.09). The most robust effect is the
induced library's **lower variance** (strict sd 1.14 vs 2.41 — it never dropped below 60,
baseline ranged 57–63). on-policy induced (55 concepts) ≥ paper lib (270) in strict mean
(61.4 vs 59.4), competitive at ~1/5 the size, though within noise. Reselection does not help
on average (60.4 vs 61.4). **oracle∪5 ≈ 69–71 for all conditions → memory does NOT expand the
solvable-puzzle set (pass@5 ceiling); it improves per-run consistency/selection.** A clear
strict-metric win would need more samples or a higher-signal setup (width>1, paired McNemar).

NOTE: this corrects an intermediate draft that mis-stated the n=5 numbers as 59.0→61.8 /
p=0.02 (typed before rep4/rep5 were read back from disk). The table above is disk-verified.

**Per-attempt data (all 20 runs):** `scripts/extract_attempts.py --glob 'eval100_*'` →
`<run_dir>/attempt_records.jsonl` + combined `outputs/_runs/eval100_all_attempt_records.jsonl`
(8956 rows: one per puzzle×pass×branch×thread×step with train/test correctness + completion)
for future ensembling (pass@k, majority vote, oracle, per-retry curves).

## 2026-05-31 (overnight) — Oracle@2 (pairwise-ensemble ceiling)

**Question:** how much does a 2nd independent sample buy? For each condition take all
C(5,2)=10 pairs of the 5 runs; each pair's oracle score = #eval_100 puzzles solved
(test-correct / strict) by EITHER run in the pair; average over the 10 pairs. Test-cases
only. Width n=1, depth max_passes=3 (train) throughout — the 5 samples are independent
re-runs (ignore_cache:true), so a "pair" = 2 independent attempts ensembled by oracle/any-correct.

- cmd: `python scripts/compute_oracle2.py`
- out: `outputs/_runs/eval100_oracle2_stats.json` (per-pair lists + means) and
  `outputs/_runs/eval100_oracle2_summary.txt`. Per-run strict sets cross-checked against
  each run's `summary.json`.

| condition | oracle@1 (single mean) | oracle@2 (pair mean ± sd) [min,max] | oracle@5 (union) |
|---|---|---|---|
| baseline — no memory | 60.4 | **65.1 ± 2.33** [62, 69] | 70 |
| on-policy induced (55) | 61.4 | **66.5 ± 1.27** [64, 69] | 70 |
| on-policy + reselection | 60.4 | 65.8 ± 1.69 [63, 69] | 71 |
| paper lib (270, ref) | 59.4 | 65.0 ± 1.89 [63, 68] | 69 |

**Takeaways:**
- A 2nd independent sample is worth **~+5 puzzles** (oracle@1 ~60 → oracle@2 ~65–66.5) — far
  more than memory's ~+1 single-run edge. Width (independent attempts) dominates here.
- At oracle@2 the conditions are close (65.0–66.5, ~1.5-puzzle spread). The induced library
  is highest (66.5) and lowest-variance (sd 1.27), and its oracle@2 lead over baseline (+1.4)
  is a touch larger than its single-run lead (+1.0) — but still within noise. Largely
  consistent with oracle∪5 ≈ equal (69–71): a 2nd random sample captures most of the headroom
  memory would, so memory's small edge is mostly (not entirely) washed out by ensembling.
- Paper lib (270) has the lowest oracle@1 (59.4) AND the lowest oracle@2 (65.0): ensembling
  two of its (more variable, sd 2.97) runs does not close the gap to the induced library.

## 2026-05-31 (overnight) — Does memory unlock NEW solves? (union-of-solves diff)

**Question:** comparing each condition's union-of-solves over its 5 runs (oracle@5 set,
test-correct), does any memory method solve puzzles baseline NEVER solves (unlocked), and
does it lose any baseline solves (lost)?

- cmd: `python scripts/compare_union_solves.py` → `outputs/_runs/eval100_union_diff.json`
- Width n=1 / depth max_passes=3 throughout; union = solved test-correct in ANY of the 5 runs.

| method | union | unlocked (mem-only) | lost (baseline-only) | net |
|---|---|---|---|---|
| baseline | 70 | — | — | — |
| induced (55) | 70 | 4 | 4 | +0 |
| reselect | 71 | 4 | 3 | +1 |
| paper lib (270) | 69 | 4 | 5 | −1 |
| **ANY memory (∪ of 3)** | **75** | **8** | **3** | **+5** |

**Findings:**
- **Yes — memory unlocks new solves, but it also loses some.** Each individual method
  unlocks ~4 puzzles baseline never gets, while losing ~3–5 baseline solves → near-zero net
  per method (induced +0, reselect +1, paperlib −1). This is exactly why the aggregate
  oracle@5 numbers looked "≈equal": they net out, but the underlying solved SETS genuinely
  differ.
- **The unlocks are partly systematic, partly noise.** 2 puzzles are unlocked by ALL THREE
  memory methods (`963f59bc`, `cb227835`) — these look like genuine memory-enabled solves.
  The rest are method-specific (induced-only: `4c177718`, `69889d6e`; reselect-only:
  `9bebae7a`, `d931c21c`; paperlib-only: `b7f8a4d8`, `ecaa0ec1`) and may be sampling luck.
- **Memory + baseline are complementary.** Pooling baseline with all memory methods reaches
  **75** distinct solved puzzles vs baseline's 70 — i.e. memory contributes **8 puzzles no
  baseline run ever solves**, at the cost of **3 puzzles every memory method misses but
  baseline gets** (`103eff5b`, `351d6448`, `c62e2108`). Net **+5** capability if ensembled
  with baseline.
- Caveat: "never in 5 runs" is a strong-but-not-infinite bar; some single-method unlocks
  could flip with more baseline samples. The all-three-method unlocks (2) and the
  baseline-only-never-memory set (3) are the most robust signals.

**Interpretation:** memory's value here is **not** a higher single-run mean (that's ~+1,
n.s.) — it's that memory explores a *different, partly-complementary* region of the solution
space. The strongest practical takeaway: an ensemble of baseline + a memory method covers
more puzzles (75) than either alone (70 / ≤71).

## 2026-05-31 (overnight) — Depth curve: memory vs baseline across retry depth

**Question:** how does the memory-vs-baseline gap change as retry DEPTH increases?
Each run has iteration_1/2/3 = pass 1 (initial) + passes 2,3 (train-feedback retries).
"Solved by depth k" = test-correct in ANY pass ≤ k (cumulative). Mean ± sd over the 5
independent samples per condition. Width held at n=1 throughout. depth1 = pass-1;
depth3 = full run (== strict_solved). Depth-3 cumulative cross-checked == summary.json
strict for all 20 runs.

- cmd: `python scripts/compute_depth_curve.py` → `outputs/_runs/eval100_depth_curve.json`

| condition | depth1 (pass-1) | depth2 | depth3 (full) | +pass2 | +pass3 |
|---|---|---|---|---|---|
| baseline — no memory | 46.2 ± 1.64 | 55.6 ± 1.52 | 60.4 ± 2.41 | +9.4 | +4.8 |
| on-policy induced (55) | 48.8 ± 2.49 | 57.2 ± 3.42 | 61.4 ± 1.14 | +8.4 | +4.2 |
| on-policy + reselection | 48.4 ± 4.45 | 57.0 ± 3.32 | 60.4 ± 2.30 | +8.6 | +3.4 |
| paper lib (270, ref) | 46.8 ± 5.12 | 55.6 ± 4.72 | 59.4 ± 2.97 | +8.8 | +3.8 |

memory(induced) − baseline gap by depth: **+2.6 (d1) → +1.6 (d2) → +1.0 (d3)**.

**Findings:**
- **Retries help everyone a lot, and roughly equally.** Each condition gains ~+9 from the
  first retry (pass 2) and ~+4 from the second (pass 3); baseline 46→55.6→60.4, induced
  48.8→57.2→61.4. The depth curves are near-parallel.
- **The memory edge is largest at depth 1 (+2.6) and SHRINKS with depth** (+1.6 at d2, +1.0
  at d3). Retries are a partial substitute for memory: a no-memory model with 2 retries
  (60.4) ≈ a memory model with retries (61.4), and the memory model's *first-attempt* lead
  (+2.6) is its biggest. Intuitively, train-feedback retries let baseline rediscover much of
  what memory front-loads.
- **Reselection does NOT improve the retry slope.** Its whole premise is better retries
  (prior-attempt-informed concept reselection on passes 2-3), but its pass-2/pass-3 gains
  (+8.6/+3.4) are not above plain induced (+8.4/+4.2) — if anything pass-3 is slightly worse.
  Reselection is not buying better depth scaling here.
- All depth-3 differences remain within noise (sds 1.1–3.0), consistent with the n=5
  significance result (strict +1.0, p=0.44). The depth view's cleaner signal is the
  **first-attempt** gap (+2.6), matching the earlier pass-1 finding (marginal, p≈0.09).

**Interpretation:** memory mostly helps the *first* attempt; retry depth and memory are
partially redundant (both surface the same fixes), so the gap compresses as depth grows.
For compute-limited single-attempt use, memory's relative value is highest; with a retry
budget, plain retries close most of the gap.
