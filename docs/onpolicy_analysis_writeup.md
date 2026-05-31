# On-Policy Concept Induction for ARC — Full Analysis Write-Up

**Date:** 2026-05-31
**Branch:** `onpolicy-concept-induction`
**Model:** DeepSeek-V4-Flash (`deepseek-v4-flash`), official DeepSeek API, throughout.
**Benchmark:** ARC-AGI-1 `eval_100` (the 100-puzzle subset used in the ArcMemo paper).

> Every number in this document was read from on-disk run artifacts
> (`summary.json`, `iteration_*/solution_trees.json`) and the computed stats files
> (`outputs/_runs/eval100_{variance,oracle2,union_diff,depth_curve}*`). Companion docs:
> `onpolicy_concept_induction_plan.md` (design), `onpolicy_experiment_log.md` (chronological
> log), `onpolicy_results_summary.md` (headline), `onpolicy_variance_stats.json` (stats).

---

## 0. Question and setup

**Research question.** Does a concept "memory" that is *entirely model-authored and
on-policy* — built by DeepSeek-V4-Flash (dsv4f) from its *own* correct solutions, with no
human BARC annotations and no human few-shot examples — improve dsv4f on ARC `eval_100` over
a **compute-normalized** no-memory baseline?

**The pipeline (all dsv4f, fully on-policy).**
1. **Seed solve.** dsv4f solves the 160 BARC seed puzzles from scratch with *no memory*:
   **123/160** strict-solved. We harvest the **120** train-correct solutions (117 also
   test-correct) as the only ground truth — the model's own work.
2. **Induction (4 LLM stages, zero human few-shot).**
   - **A** solution → pseudocode + summary (per puzzle; 120/120).
   - **B** pseudocode → free-form concept tags + descriptions → **675 tags, 605 unique**.
   - **C** corpus-global LLM map→reduce unification + critique → **57 canonical concepts**
     (freq ≥ 2; 479 singleton appendix). *Finding: dsv4f won't merge a flat 600-item list
     (single-shot reduce made 0 merges); it needed lexical surface-form normalization +
     iterative agglomerative reduce + a forceful critique pass.*
   - **D** per-concept typed synthesis → `induced_concepts_v1.json`: **55 concepts** (19
     structures, 36 routines, 3 custom types), drop-in for the ArcMemo `ConceptMemory` schema.
3. **Evaluation.** The original ArcMemo solve loop (LLM concept selection → op3 hint →
   python-transform-retry) on `eval_100`, dsv4f throughout.

**Four conditions compared.**
| short name | memory | size |
|---|---|---|
| **baseline** | none (vanilla) | — |
| **induced** | on-policy model-authored | 55 concepts |
| **reselect** | induced + reselection-with-prior-context (new mechanism) | 55 concepts |
| **paperlib** | the paper's BARC-derived library (reference) | 270 concepts |

**Compute normalization (the hard invariant).** All conditions use the **same model**, the
**same width** (independent attempts per puzzle, `n = 1`), and the **same depth** (retry
budget, `max_passes = 3`, train-correct retry criterion), with identical generation config.
The memory conditions issue extra *concept-selection* requests — inherent method overhead,
reported separately — but these do **not** change `n` or the retry budget. Two axes:
- **Width** = number of independent attempts per puzzle (we vary this by running 5 times).
- **Depth** = number of retries within an attempt (`max_passes`; passes 1→3).

**Metric.** "Strict-solved" = all held-out test pairs correct. "pass-1" = solved on the
first attempt only (exactly one solve call/puzzle — the strictest compute-normalized view).

---

## 1. First result (2-sample preview) — and why it didn't survive

The very first comparison used **2 independent samples** per condition (cache cold on the
first run, `ignore_cache:true` on the second):

| condition | strict (2 samples) | mean | pass-1 |
|---|---|---|---|
| baseline | 57, 59 | **58.0** | 46, 46 |
| induced | 63, 61 | **62.0** | 53, 49 |

This looked like a clean **+4 strict** win for memory, stable in direction (both induced runs
beat both baseline runs). **But two samples cannot estimate variance**, and ARC strict-solve
turned out to be noisy. This motivated the variance study below — and, as it happened, the
extra samples substantially revised the headline. (Honest note: an intermediate draft also
briefly reported a +8 figure produced before the runs had finished; that was a process error,
caught and corrected by always reading completed `summary.json` from disk. All numbers in
this document are disk-verified.)

---

## 2. Variance: 5 independent samples per condition (the WIDTH axis)

We extended every condition to **5 independent samples** (re-runs with `ignore_cache:true`,
temperature 0.3 → genuinely independent), holding width n=1 and depth max_passes=3 fixed.

| condition | library | strict (5 samples) | strict mean ± sd | pass-1 mean ± sd |
|---|---|---|---|---|
| baseline — no memory | — | 57, 59, 63, 62, 61 | **60.4 ± 2.41** | 46.2 ± 1.64 |
| on-policy induced | induced (55) | 63, 61, 60, 61, 62 | **61.4 ± 1.14** | 48.8 ± 2.49 |
| on-policy + reselection | induced (55) | 64, 59, 60, 61, 58 | 60.4 ± 2.30 | 48.4 ± 4.45 |
| paper lib (reference) | compressed_v1 (270) | 64, 60, 59, 58, 56 | 59.4 ± 2.97 | 46.8 ± 5.12 |

**Significance (Welch's t-test, baseline vs on-policy induced; pure-Python, scipy absent):**
- **strict:** +1.0 (60.4 → 61.4), t = 0.84, df ≈ 5.7, **p = 0.44 two-tailed — NOT significant**.
- **pass-1:** +2.6 (46.2 → 48.8), t = 1.95, df ≈ 6.9, **p = 0.093 two-tailed (0.046 one-tailed) — marginal**.

**What changed from the 2-sample preview.** The baseline's 3rd–5th samples included high
runs (63, 62), lifting its mean from 58.0 to **60.4**. The memory gain shrank from the
apparent +4 to **+1.0 (strict, n.s.)**. The lesson: at this scale the strict metric has
~±2.4 run-to-run noise, so ≥5 samples are needed; a 2-sample delta is unreliable. (This very
effect likely explains the "slightly different numbers" a collaborator saw on another machine.)

**The most robust effect is lower variance.** The induced library's strict sd is **1.14 vs
baseline 2.41** — it never scored below 60, while baseline ranged 57–63. Memory makes dsv4f
more *consistent* more than it raises the mean.

**Library size.** The on-policy induced library (55 concepts) ≥ the paper's BARC library (270
concepts) in strict mean (61.4 vs 59.4), at ~⅕ the size — competitive though within noise.

---

## 3. Single-pass (pass-1) view — the strictest normalization

"pass-1" counts only the first attempt (exactly 100 solve calls/condition, zero retries — no
width or depth asymmetry whatsoever). It is the cleanest place to read memory's effect:

| condition | pass-1 mean ± sd |
|---|---|
| baseline | 46.2 ± 1.64 |
| induced | **48.8 ± 2.49** |
| reselect | 48.4 ± 4.45 |
| paperlib | 46.8 ± 5.12 |

memory(induced) − baseline = **+2.6, marginal (p ≈ 0.09)** — larger and cleaner than the
full-run strict gap (+1.0, n.s.). **Memory helps most on the first attempt** (it front-loads
the right concepts before any retry feedback exists). This sets up the depth analysis in §6.

---

## 4. Union of solves over 5 runs — does memory unlock NEW capability?

Per condition, take the **union of puzzles solved (test-correct) across its 5 runs** (= the
pass@5 / oracle∪5 set), then diff against baseline's union.

| condition | union (oracle∪5) | unlocked (mem solves, baseline never) | lost (baseline solves, mem never) | net |
|---|---|---|---|---|
| baseline | 70 | — | — | — |
| induced | 70 | 4 | 4 | **+0** |
| reselect | 71 | 4 | 3 | **+1** |
| paperlib | 69 | 4 | 5 | **−1** |
| **any memory (∪ of 3)** | **75** | **8** | **3** | **+5** |

**Findings.**
- **Memory genuinely unlocks ~4 puzzles per method that baseline never solves** in any of its
  5 runs — but it also **loses ~3–5** baseline solves. Per-method net ≈ 0 (+0/+1/−1). This is
  *why* the aggregate oracle∪5 looked flat (70/70/71/69): the solved **sets differ**, but
  they net out.
- **Two unlocks are robust** — solved by **all three** memory methods (`963f59bc`,
  `cb227835`) — these look like genuinely memory-enabled solves, not sampling luck.
- **Three puzzles are baseline-only**, solved by baseline but by *no* memory method ever
  (`103eff5b`, `351d6448`, `c62e2108`): memory's hints actively steer the model away from these.
- **Memory and baseline are complementary.** Pooling baseline + all memory methods reaches
  **75 distinct solved puzzles** vs baseline's 70: memory contributes 8 the baseline never
  gets; baseline contributes 3 no memory method gets.

**Interpretation.** Memory's value is **not** a higher ceiling — it explores a *different,
partly-complementary* region of the solution space. An ensemble of baseline + a memory method
covers more puzzles than either alone.

---

## 5. Oracle@2 — the value of a second independent sample (WIDTH ensembling)

For each condition, take all **C(5,2) = 10 pairs** of runs; each pair's oracle score = puzzles
solved (test-correct) by *either* run; average over the 10 pairs. This estimates the ceiling
of a 2-independent-sample ensemble.

| condition | oracle@1 (single mean) | oracle@2 (pair mean ± sd) [min,max] | oracle@5 (union) |
|---|---|---|---|
| baseline — no memory | 60.4 | **65.1 ± 2.33** [62, 69] | 70 |
| on-policy induced (55) | 61.4 | **66.5 ± 1.27** [64, 69] | 70 |
| on-policy + reselection | 60.4 | 65.8 ± 1.69 [63, 69] | 71 |
| paper lib (270, ref) | 59.4 | 65.0 ± 1.89 [63, 68] | 69 |

**Findings.**
- **A second independent sample buys ~+5 puzzles** (oracle@1 ~60 → oracle@2 ~65–66.5) — far
  more than memory's ~+1 single-run edge. **Width dominates** memory at this scale.
- The induced library is highest and lowest-variance at oracle@2 (66.5 ± 1.27); its lead over
  baseline at oracle@2 (+1.4) is marginally larger than its single-run lead (+1.0), but still
  within noise.
- Paper lib (270) is lowest at *both* oracle@1 (59.4) and oracle@2 (65.0): ensembling two of
  its (more variable) runs does not close the gap to the induced library.

---

## 6. Depth curve — memory vs baseline across retry depth (the DEPTH axis)

Cumulative strict-solved at each depth k (solved if *any* pass ≤ k is test-correct), mean ± sd
over the 5 samples. Width fixed at n=1. depth1 = pass-1; depth3 = full run.

| condition | depth 1 (pass-1) | depth 2 | depth 3 (full) | gain +pass2 | gain +pass3 |
|---|---|---|---|---|---|
| baseline — no memory | 46.2 ± 1.64 | 55.6 ± 1.52 | 60.4 ± 2.41 | +9.4 | +4.8 |
| on-policy induced (55) | 48.8 ± 2.49 | 57.2 ± 3.42 | 61.4 ± 1.14 | +8.4 | +4.2 |
| on-policy + reselection | 48.4 ± 4.45 | 57.0 ± 3.32 | 60.4 ± 2.30 | +8.6 | +3.4 |
| paper lib (270, ref) | 46.8 ± 5.12 | 55.6 ± 4.72 | 59.4 ± 2.97 | +8.8 | +3.8 |

**induced − baseline gap by depth: +2.6 (d1) → +1.6 (d2) → +1.0 (d3) — it SHRINKS with depth.**

**Findings.**
- **Retries help everyone a lot, and roughly equally:** ~+9 from the first retry, ~+4 from
  the second. The depth curves are near-parallel.
- **Memory's edge is front-loaded** — biggest at depth 1 (+2.6) and compresses as depth grows
  (+1.0 by depth 3). **Retries partially substitute for memory:** baseline with 2 retries
  (60.4) ≈ memory with retries (61.4). Train-feedback retries let the baseline rediscover much
  of what memory front-loads.
- **Reselection does not improve the retry slope** — even though "better retries" is its
  entire premise (it re-selects concepts on passes 2–3 using the prior attempt's code + a
  `<reflection>` step), its pass-2/pass-3 gains (+8.6/+3.4) are not above plain induced
  (+8.4/+4.2). It is not buying better depth scaling here.

**Interpretation.** Memory mostly helps the *first* attempt; depth and memory are partially
redundant (both surface the same fixes), so the gap narrows as retries accrue. For
compute-limited single-attempt use, memory's relative value is highest; with a retry budget,
plain retries close most of the gap.

---

## 7. Does this mean memory "doesn't scale with test-time compute"?

The gap shrinks as we add either kind of test-time compute:

| compute axis | gap (induced − baseline) | trend |
|---|---|---|
| **depth** (retries) | +2.6 (pass 1) → +1.6 (pass 2) → +1.0 (pass 3) | shrinks, stays positive |
| **width** (oracle@k) | +1.0 (k=1) → +1.4 (k=2) → 0.0 (k=5) | shrinks to a tie at the ceiling |

**This gap-closing is the expected behavior, not a failure mode.** Memory's value
proposition is *amortizing rediscovery*: it spares the model from re-deriving ideas it
already worked out on earlier puzzles. Test-time compute — extra retries, or extra
independent samples — is precisely the budget the model can spend to *rediscover those same
ideas on the fly*. So giving the no-memory baseline more exploration budget should, by
construction, let it close the gap. A memory advantage that *narrows* as you pour in
per-puzzle compute is exactly what the rediscovery account predicts. The interesting
quantity is not whether the gap shrinks (it must) but **how much compute it takes to erase
it** — here, roughly one extra independent sample or ~two retries.

**Why this is not the criticism "the method doesn't scale with test-time compute."**
1. **The advantage never inverts.** Across every compute level on both axes the gap is
   ≥ 0 — memory is never *worse* than baseline at higher compute. At worst it ties (oracle@5,
   70 = 70). There is no anti-scaling: more compute doesn't make memory a liability, it just
   makes its head-start less necessary.
2. **Per-puzzle compute is the wrong axis for memory's scaling.** Retry-rediscovery is
   *throwaway*: whatever the baseline figures out on passes 2–3 of puzzle *i* is discarded
   before puzzle *i+1*, so it re-pays the full rediscovery cost on every puzzle. Memory is
   *cumulative*: it pays the discovery cost once and reuses it. The axis along which memory is
   *supposed* to scale is therefore the **number of problems / size of the library**, not the
   per-puzzle attempt budget. Our width/depth sweeps deliberately hold the library fixed and
   scale only per-puzzle compute — the one axis where memory and rediscovery are
   substitutes — so a closing gap there says nothing about whether memory scales on its own
   axis.
3. **The front-loaded win is the economically relevant one.** The largest gap is at the
   smallest budget (pass 1: +2.6; single sample: +1.0). For any deployment where attempts are
   expensive or latency-bound — i.e. you *can't* afford many retries/samples per problem —
   memory delivers its biggest edge exactly where it matters, and does so with **lower
   variance** (strict sd 1.14 vs 2.41), which is itself a compute saving (fewer wasted runs).
4. **Memory and baseline remain complementary at scale.** Even pooling 5 samples, baseline +
   any-memory covers **75** distinct puzzles vs 70 for baseline alone (§4) — memory keeps
   adding solves the baseline never reaches no matter how many times it re-rolls, so the
   contribution is not fully absorbed by more sampling.

**Honest limit / forward-looking claim.** On *this* fixed 100-puzzle set with a fixed
55-concept library, memory does **not** raise the absolute oracle@5 ceiling (tied at 70): we
have **not** demonstrated "memory wins at large test-time compute" on the ceiling metric, and
we should not claim it. The stronger statement — that the early-attempt wins compound and
matter *more* on **future / harder problems** — is the natural consequence of the
amortization mechanism (a growing library should keep paying off on novel puzzles while the
baseline re-pays rediscovery each time), but it is a **hypothesis our current eval does not
test.** The clean way to settle it is to scale memory's *own* axis: grow the seed corpus /
library and measure whether the gap widens with problem count, and/or evaluate on harder
splits where per-puzzle rediscovery is more expensive. That is the experiment that would
convert "expected gap-closing under per-puzzle compute" into a positive "memory scales" result.

---

## 8. Synthesis — what the whole picture says

Decomposing the comparison along both compute axes plus capability/ensembling gives a
consistent story:

| lens | result | takeaway |
|---|---|---|
| single-run strict (n=5) | +1.0, p=0.44 | not significant |
| single-run pass-1 (n=5) | +2.6, p≈0.09 | marginal; memory's cleanest gain |
| variance | induced sd 1.14 vs 2.41 | memory → more consistent |
| union∪5 (capability) | induced 70 = baseline 70; any-memory 75 | unlocks ~4/method, loses ~3–5; complementary |
| width (oracle@2) | +5 from a 2nd sample vs +1 from memory | width ≫ memory |
| depth | edge +2.6 (d1) → +1.0 (d3) | memory front-loads; retries substitute |

**Bottom line.** On-policy, fully model-authored memory produces a **small,
not-statistically-significant single-run accuracy gain (~+1 strict / ~+2.6 pass-1), lower
run-to-run variance, and a real-but-churning unlocked-solves effect.** Its value is
**front-loaded (helps the first attempt most) and complementary (solves a different subset),
not a higher ceiling.** Both adding width (a 2nd independent sample, +5) and adding depth
(retries, +9/+4) outweigh memory at this scale. Critically, the **on-policy 55-concept
library matches or beats the paper's 270-concept human-derived library** (61.4 vs 59.4 strict
mean) at ~⅕ the size — the model can author a competitive concept library from its own solves.

**Caveats.**
- eval_100 (100 puzzles) + strict-solve is noisy (~±2.4); n=5 narrows but doesn't eliminate
  CIs. The strict gain would need more samples (or a paired per-puzzle McNemar) to firm up.
- "Unlocked / never-solved-in-5-runs" is a strong-but-finite bar; single-method unlocks could
  flip with more samples. The all-3-method unlocks (2) and baseline-only set (3) are the most
  robust capability signals.
- Reselection's null result is on a single mechanism design (inject prior code + reflect); an
  LLM-summarized exploration digest might do better but is untested.

---

## 9. Reproducibility — artifacts and commands

**Code (branch `onpolicy-concept-induction`).**
- Pipeline: `scripts/harvest_solves.py`, `src/mem2/concepts/induction.py`,
  `scripts/induce_library.py` (stages a/b/c/d).
- Reselection: `RESELECT_PROMPT_TEMPLATE` in `src/mem2/concepts/prompts/arc_select.py` +
  `ps_selector` flags (`use_reselection`, `reselect_max_attempts`, `reselect_max_chars`).
- Analysis: `scripts/extract_attempts.py`, `scripts/compute_oracle2.py`,
  `scripts/compare_union_solves.py`, `scripts/compute_depth_curve.py`.
- Provider: `deepseek-v4-flash` on `Provider.DEEPSEEK` + profile `llmplus_deepseek_v4_flash`.

**Configs.** `configs/experiments/eval100_{baseline,arcmemo,arcmemo_reselect,paperlib}.yaml`
plus `_rep2.._rep5` (5 independent samples each, `ignore_cache:true`).

**Data artifacts (under `outputs/_runs/`).**
- `eval100_variance_stats.json` — per-sample strict/pass-1, means, sds, Welch tests (also
  copied to `docs/onpolicy_variance_stats.json`).
- `eval100_oracle2_stats.json` / `.txt` — per-pair oracle@2.
- `eval100_union_diff.json` — unlocked/lost puzzle-ID lists.
- `eval100_depth_curve.json` — cumulative solves by depth.
- `eval100_all_attempt_records.jsonl` — **8956 rows**, one per puzzle × pass × branch ×
  thread × step (train/test correctness + completion) — the raw substrate for *any* future
  ensembling (pass@k, majority vote, oracle, per-retry curves).

**Re-run.**
```bash
# induce the library
python -m mem2.cli.run --config configs/experiments/onpolicy_solve_barc.yaml
python scripts/harvest_solves.py --run-dir outputs/_runs/onpolicy_solve_barc/<hash> --train-only
IND=outputs/_runs/onpolicy_solve_barc/<hash>/induction
python scripts/induce_library.py --solves $IND/solved_seeds.json --stage a
python scripts/induce_library.py --solves $IND/solved_seeds.json --stage b
python scripts/induce_library.py --stage c --out-dir $IND
python scripts/induce_library.py --stage d --out-dir $IND
# eval (5 samples/condition)
for c in baseline arcmemo arcmemo_reselect paperlib; do
  for r in "" _rep2 _rep3 _rep4 _rep5; do
    python -m mem2.cli.run --config configs/experiments/eval100_${c}${r}.yaml
  done
done
# analyses
python scripts/extract_attempts.py --glob 'eval100_*' --out outputs/_runs/eval100_all_attempt_records.jsonl
python scripts/compute_oracle2.py
python scripts/compare_union_solves.py
python scripts/compute_depth_curve.py
```
