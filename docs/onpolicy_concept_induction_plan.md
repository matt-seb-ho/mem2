# On-Policy Concept Induction with DeepSeek-V4-Flash — Plan

**Status:** DRAFT for review (no code launched yet)
**Author:** Claude (for Matthew Ho)
**Date:** 2026-05-31

---

## 1. Goal & Philosophy

Build an ArcMemo concept library that is **as on-policy as possible** with a single
model, `deepseek-v4-flash` (dsv4f), and **entirely model-authored** — no BARC human
annotations (`# concepts:` / `# description:`), no hand-written few-shot concept
examples (`op3a.yaml`, `example_concepts.yaml`). The model's own successful solutions
are the only ground truth; the model writes every pseudocode, tag, description, and
typed concept.

Two departures from the original ArcMemo offline pipeline:

1. **On-policy solutions.** Instead of abstracting from the 160 ground-truth BARC
   Python solutions, we have dsv4f *solve the same 160 seed puzzles from scratch*
   (vanilla, no memory) and abstract only from its **own correct solves**.
2. **Corpus-global induction (new).** Instead of one-puzzle-at-a-time accretion into a
   growing in-prompt repository, we induce the library at the **corpus level**:
   pseudocode → free-form tagging/description for every solve → a **vocabulary
   unification pass** across the whole corpus that discovers frequent concepts →
   synthesis of the final typed library from those unified clusters.

Then we run the **original ArcMemo solving pipeline** (concept selection → op3 hint →
python-transform-retry solve) on the **eval_100** validation subset using this
induced library, with dsv4f as the model throughout (selection + solving).

End state: a clean comparison of "model's own self-induced memory" vs. the
paper's BARC-derived `compressed_v1.json`, all on-policy with dsv4f.

---

## 2. End-to-End Data Flow

```
 [160 BARC seed puzzles]  (puzzle grids only, from data/arc_agi/training/<id>.json;
        │                   BARC .py solutions + annotations are NOT used)
        ▼
 PHASE 1: Vanilla solve (no memory, dsv4f)          ── runs first, in background ──
        │  outputs/_runs/onpolicy_solve_barc/<hash>/
        ▼
 PHASE 2: Harvest correct solves
        │  → solved_seeds.json  {uid: {code, train_ok, test_ok, prompt}}
        ▼
 PHASE 3: Corpus-global concept induction (dsv4f, NO human few-shots)
        │  Stage A  per-solve  pseudocode + summary        → induction/stageA_pseudocode.json
        │  Stage B  per-solve  free-form tags + NL desc    → induction/stageB_tags.json
        │  Stage C  corpus     unify/cluster vocabulary    → induction/stageC_vocab.json
        │  Stage D  per-cluster synthesize typed Concept   → induced_concepts_v1.json
        ▼
 PHASE 4: Original ArcMemo solve on eval_100 (dsv4f)
           memory_builder=arcmemo_ps(seed=induced_concepts_v1.json)
           retriever=ps_selector (inline LLM selection)  +  op3 hint  +  python_transform_retry
        │  outputs/_runs/onpolicy_eval100/<hash>/
        ▼
        summary.json  (strict_solved_puzzles / official_score) + cost report
```

---

## 3. Phase 0 — Provider & Cost Infrastructure (do first; blocks everything)

The official DeepSeek API (`https://api.deepseek.com/v1`) is already wired in the
vendored `llmplus` (`Provider.DEEPSEEK`, `env_key=DEEPSEEK_API_KEY`), but its model
registry only lists `deepseek-chat` / `deepseek-reasoner`, and mem2 has **no provider
profile** for the official DeepSeek backend (existing dsv4f configs route through
OpenRouter). Also: **token usage is tracked but cost is not.**

Work items:

1. **Register the model** — add `"deepseek-v4-flash": ModelMeta("deepseek-v4-flash")`
   (and optionally `deepseek-v4-pro`) to the `Provider.DEEPSEEK` block in
   `third_party/llm_wrapper/llmplus/model_registry.py`. Note `supports_multi=False`
   for DeepSeek → `n>1` is emulated by issuing N separate requests.
2. **Add a mem2 provider profile** `llmplus_deepseek_v4_flash` in
   `src/mem2/providers/profiles.py`:
   ```python
   ProviderProfile(
       profile_name="llmplus_deepseek_v4_flash",
       backend="llmplus", provider="deepseek", model="deepseek-v4-flash",
       system_prompt=<ArcMemo default solver system prompt>,
       default_max_concurrency=32, retry_attempts=5,
       extra_gen_defaults={"n": 1, "temperature": 0.3},
   )
   ```
3. **Usage tracking (tokens only — decision D0).** No dollar conversion. Add
   `scripts/usage_report.py` that aggregates `provider_usage.json` +
   per-iteration `token_usage.json` (`input_tokens`/`output_tokens`/`reasoning_tokens`/
   `requests`/`completions`, already emitted) into `usage_report.json` rolled up by
   stage (solve / pseudocode / tag / unify / synthesize / select / eval). Pricing hook
   left as a stub if we want to attach $/MTok later.
4. **Preflight.** Use/extend `scripts/preflight_model.py` + `scripts/real_api_smoketest.py`
   to fire one real dsv4f call through the official API, confirm auth from
   `~/mem2/.env`, and confirm usage accounting populates. **Gate: do not start Phase 1
   until this green-lights.**

---

## 4. Phase 1 — Vanilla Solve on BARC Seeds (kick off, runs in background)

Goal: maximize the pool of **correct, on-policy** dsv4f solutions over the 160 usable
seed puzzles, with **zero memory** and **no hints**.

- New config `configs/experiments/onpolicy_solve_barc.yaml` (`_base_: ../base.yaml`):
  - `memory_builder: none`, `memory_retriever: none`,
    `inference_engine.prompt_options.include_hint: false`.
  - `benchmark.data_root: data/arc_agi/training`,
    `benchmark.include_ids: <160 resolvable barc_seed ids>` (loaded from
    `splits.json["barc_seeds"]["ids"]`, dropping `common`, `template`,
    `25d487eb_Kevin`, `264363fd_Kevin` — see §7 note), `limit: 0`.
  - `provider.profile_name: llmplus_deepseek_v4_flash`, `dotenv_path: .env`.
  - `inference_engine.model: deepseek-v4-flash`, `gen_cfg`: `temperature 0.3`,
    `max_tokens 16384`, `batch_size 32`.
  - **Solve budget:** `max_passes: 3`, `retry_criterion: train`, train-based error
    feedback (mirrors the standard ArcMemo retry loop) to lift the solved-seed yield.
- Launch with `python -m mem2.cli.run --config .../onpolicy_solve_barc.yaml`,
  in the background; monitor token spend via the per-iteration `token_usage.json`.

**Why first:** it's the long pole and has no code dependency on Phase 3. We build
Phases 2–3 while it runs.

---

## 5. Phase 2 — Harvest Correct Solves

New `scripts/harvest_solves.py` (adapts `load_solved_problems()` logic from
`scripts/extract_concepts.py`):

- Join `attempts.jsonl` ⋈ `eval_records.jsonl` on `(problem_uid, attempt_idx)` from the
  Phase-1 run dir; pull the winning `completion` (Python code) per solved puzzle.
- Emit `induction/solved_seeds.json`: `{uid: {code, prompt, train_ok, test_ok,
  pass_idx}}`.
- **Correctness criterion (decision D1):** primary pool = solves correct on **all
  train pairs** (what the solver can actually verify, matches ArcMemo's notion of a
  "solved" puzzle); record `test_ok` alongside for analysis and an optional stricter
  pool. Report counts: `#train_ok`, `#train_ok & test_ok`.

---

## 6. Phase 3 — Corpus-Global Concept Induction (new system)

Implemented as `src/mem2/concepts/induction.py` (corpus-level, mirrors the *structure*
of `concepts/extraction.py` but adds the global unification stage) + a driver
`scripts/induce_library.py`. **All prompts are zero-shot — no human concept examples.**
All four stages run on dsv4f through the official API with full token/cost accounting.

- **Stage A — Pseudocode + summary (per solve).** Prompt dsv4f to rewrite each correct
  solution as abstraction-friendly pseudocode + a one-line transformation summary.
  Reuses the *intent* of arc_memo's `pseudocode_instr.txt` but **stripped of the
  few-shot `{examples}` and `{concepts}` blocks**. Output:
  `induction/stageA_pseudocode.json` `{uid: {pseudocode, summary}}`.
- **Stage B — Free-form tagging + descriptions (per solve).** Prompt dsv4f to emit, for
  each solve, a list of **candidate concept tags** (free vocabulary — the model invents
  the names) each with a short natural-language description and a `kind` guess
  (structure vs. routine). No global vocabulary yet — deliberately divergent. Output:
  `induction/stageB_tags.json` `{uid: [{tag, kind, description}]}`.
- **Stage C — Vocabulary unification (corpus-global; LLM-driven grouping).** The novel
  pass and the heart of the system. Decisions: **map-reduce + bounded loop (D3-ind)**
  with **LLM-driven grouping from scratch (D5)** — NO embedding/cosine pre-grouping.
  Even "what is similar / what is common" is a model judgment, consistent with the
  fully model-authored ethos. To keep each call tractable (a single flat "unify 600
  tags" dump is lossy/non-reproducible), grouping is done **hierarchically (map →
  reduce)**:
  1. **Map (LLM, per chunk):** shard the global tag+description set into bounded chunks;
     each call proposes candidate concept groups (canonical name, kind, member tags) for
     its chunk.
  2. **Reduce (LLM, over group *summaries* only):** reconcile the per-chunk group lists
     into one canonical vocabulary — merge synonyms, split conflations, rename. Input is
     compact (canonical names + glosses + counts), never the raw corpus dump.
  3. **Frequency = mechanical bookkeeping only:** after grouping, count how many puzzles
     map to each canonical concept (not used as a clustering signal). Drives the keep
     threshold and the critique loop.
  4. **Bounded reconciliation loop (light agency):** a critic call flags incoherent /
     duplicate / conflated concepts, frequent ideas with no concept, and singletons;
     only flagged items get another targeted grouping round. Repeat until stable or a
     round cap (loop-until-stable). This is the one place we inject agency — bounded, not
     open-ended.
  Output `induction/stageC_vocab.json`: `{canonical_name, kind, aliases, member_tags,
  member_uids, frequency, gloss}`. **Frequency threshold (D2):** keep concepts in
  `>=2` puzzles; singletons retained in an appendix tier.
- **Stage D — Typed library synthesis (per canonical concept).** For each canonical
  concept, give dsv4f its member solves' pseudocode + tag descriptions and synthesize a
  fully **typed `Concept`** matching the mem2/ArcMemo schema: `name, kind,
  routine_subtype, output_typing, parameters[{name,typing,description}], description,
  cues[], implementation[], used_in[]`. Output `induced_concepts_v1.json` in
  **`ConceptMemory` serialization** so it drops straight into `ArcMemoPsMemoryBuilder`.

Reuses where possible: `scripts/compress_concepts.py` (post-hoc concept compression)
and `scripts/compute_concept_frequencies.py` (mechanical frequency counts for Stage C.3).

---

## 7. Phase 4 — Original ArcMemo Solve on eval_100

New config `configs/experiments/onpolicy_eval100.yaml`, modeled on
`phase1_arc_base.yaml`, but on-policy and pointed at the induced library:

- `benchmark.data_root: data/arc_agi/evaluation`, `include_ids` = `splits.json["eval_100"]["ids"]` (100 puzzles).
- `memory_builder: arcmemo_ps`, `seed_memory_file: data/arc_agi/concept_memory/induced_concepts_v1.json`, `freeze_memory: true` (offline-built library; no online mutation during eval).
- **Retriever (decision D3): `ps_selector` in inline-LLM-selection mode** — this is the
  original ArcMemo mechanism (LLM picks relevant concepts per puzzle via the
  `arc_select` prompt), as opposed to `ps_topk` (embedding top-k). dsv4f is the selector
  → fully on-policy. `top_k`/render options mirror phase1 (`include_description: true`).

- **Reselection between iterations (decision D3, NEW work item).** Between attempt _k_
  and _k+1_ we want the selector to re-look at the concept library **informed by
  attempt _k_'s exploration** (its code + execution feedback), selecting a fresh concept
  set for _k+1_. Findings from the code:
  - The plumbing largely exists. The runner re-invokes the retriever every retry pass
    and already passes `previous_attempts=history` into `async_retrieve(...)`
    (`runner.py` ~665–673); `run.retrieval_rounds_per_pass` supports intra-pass rounds;
    `include_reselected_lessons: true` records reselected hints to a
    `reselect_concepts/` artifact dir each pass>0.
  - **The gap:** `ps_selector` inline-selection (Mode 3) builds its prompt as
    `select_template.format(concepts=…, puzzle=…)` and **ignores `previous_attempts`** —
    and the ported `arc_select` `SELECT_PROMPT_TEMPLATE` has no slot for prior attempts.
  - **Work:** add a reselection prompt variant with a `{prior_attempts}` block (prior
    code + feedback/errors), and have `ps_selector` Mode 3 inject `previous_attempts`
    when non-empty (i.e. on retry passes). Set `include_reselected_lessons: true` and
    confirm the reselected concept set flows into the `retry_attempt` hint. This is the
    one genuinely new mechanism vs. the paper's solve loop; scope it as a small,
    isolated change to `ps_selector` + a new template in `concepts/prompts/`.
- `inference_engine`: dsv4f, `prompt_options.include_hint: true`, `hint_template_key: op3`
  (the op3 hint template mem2 already ports), `max_passes: 3`, train-based retry — i.e.
  **the same solving mechanics as the paper.**
- Provider: `llmplus_deepseek_v4_flash` (official API), cost report emitted.

**Baseline for comparison:** the same config pointed at the paper's
`compressed_v1.json` (and a no-memory vanilla run on eval_100) so we can attribute any
delta to the self-induced library.

---

## 8. Files: new vs. modified

**New**
- `configs/experiments/onpolicy_solve_barc.yaml` (Phase 1)
- `configs/experiments/onpolicy_eval100.yaml` (Phase 4) + baseline variants
- `scripts/harvest_solves.py` (Phase 2)
- `src/mem2/concepts/induction.py` + `scripts/induce_library.py` (Phase 3)
- `scripts/compute_cost.py` + pricing constant (Phase 0)
- `induction/` output dir + `data/arc_agi/concept_memory/induced_concepts_v1.json`

**Modified**
- `third_party/llm_wrapper/llmplus/model_registry.py` (register `deepseek-v4-flash` on
  `Provider.DEEPSEEK`)
- `src/mem2/providers/profiles.py` (add `llmplus_deepseek_v4_flash` profile)

No changes to the orchestrator, evaluator, or the arcmemo_ps builder/retriever —
the induced library is schema-compatible by construction.

---

## 9. Decisions (RESOLVED 2026-05-31)

- **D0 — Usage tracking:** ✅ **Tokens only**, no $ conversion. (`usage_report.json`.)
- **D1 — Solve-harvest criterion:** ✅ **Train-correct** primary pool; record test-correctness alongside.
- **D2 — Stage C frequency cutoff:** keep `freq >= 2` (drop singletons; retain in an appendix tier). Proposed default — flag if you want all kept.
- **D3 — eval_100 retriever:** ✅ **`ps_selector` inline LLM selection** + **add reselection-between-iterations** (see §7 new work item: prior-attempt-informed reselection on retry passes).
- **D4 — Solve budget / sampling:** ✅ **`max_passes=3` + train retry, `n=1`.**
- **D3-ind — Phase 3 synthesis shape:** ✅ **Map-reduce + bounded reconciliation loop**
  (not single-dump, not fully open-ended agent).
- **D5 — Tag grouping signal:** ✅ **LLM groups from scratch** (no embedding/cosine
  pre-grouping); frequency is mechanical bookkeeping only.
- **Reselection prior-context (confirmed):** the reselection prompt receives prior-attempt
  context — implemented as an LLM-summarized "exploration digest" (what was tried, ideas
  raised, why it failed) distilled from attempt _k_'s completion + execution feedback,
  injected into the `{prior_attempts}` slot so reselection builds on prior work.

---

## 10. Risks & mitigations

- **Small solved pool.** dsv4f vanilla may solve well under 160 seeds → thin library.
  Mitigation: D4 sampling, train-correct criterion, report yield before committing to
  Phase 3.
- **Schema drift in Stage D.** Model-authored YAML may violate the `Concept` schema.
  Mitigation: validate/repair against the `Concept`/`ParameterSpec` dataclasses on
  ingest (reuse `concepts/extraction.py`'s parse+`write_concept` path).
- **Cost runaway.** 4 LLM stages over the corpus + per-puzzle selection on eval.
  Mitigation: per-stage token caps + `cost_report.json` checkpoints; Stage C embedding
  pre-grouping to shrink prompts.
- **`supports_multi=False`** on official DeepSeek → no server-side `n>1`; multi-sample
  is N requests (cost-linear). Reflected in D4.
- **Annotation leakage.** Must feed dsv4f only puzzle grids + its *own* solution code —
  never the BARC `.py` seeds or their `# concepts:`/`# description:` comments.
```
