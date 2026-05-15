# mem2 — Research Architecture

**What the method is: an online lifelong memory-augmented agent that adapts memory across tasks (cross-task lifelong) and re-attempts within a single task using a verifier-gated retry loop (within-task online).**

> **Status**: canonical, continuously refined — the method counterpart to `DESIGN.md`.
> - `ARCHITECTURE.md` (this doc) = what the *method* is (the research claim).
> - `DESIGN.md` = how the *codebase* is structured (the run system, file layout, structural enforcement).
> Iterate on this as the research direction lands. Historical evolution lives in `mem_devlog/`.

### Three-state labeling convention (read first)

Throughout this document, claims fall into three states that must be distinguished:

- **[CURRENT]** — implemented and exercised by today's mem2 runtime (`origin/main` as of the migration-history date in §11).
- **[TARGET]** — proposed structural design; not yet implemented but on the structural-hardening backlog (see DESIGN.md §8).
- **[HYPOTHESIS]** — a research direction that motivates the paper claim; the mechanism is described in an origin thread (`origins/threads/<thread>/`) and may have a Phase G ablation plan but is not yet present in runtime code.

When a section's tag is absent, treat the claim as **[CURRENT]**. Sections that mix states tag each claim inline.

---

## 0 — The core claim **[HYPOTHESIS, partial CURRENT]**

**A memory-augmented agent that improves over time at the task level (across episodes) and within a single attempt (across iterations) provides a better cost-accuracy frontier than either a stateless LLM or a single-shot retrieval-augmented baseline.**

Three dimensions of "online":

1. **Cross-task lifelong** — memory persists across episodes; concepts learned solving puzzle A help on puzzle B; failures are retained alongside successes. **[CURRENT]** — the runtime's `MemoryBuilder.update` + `consolidate` pipeline accumulates state across attempts (`src/mem2/orchestrator/runner.py:726-732`, `:949-950`).
2. **Within-task verifier-gated** — at iteration `t+1`, the agent sees what failed at iteration `t` and re-attempts. **[CURRENT, partial]** — the runtime fires a generic retry prompt that includes previous responses + execution errors + output mismatches + optionally reselected lessons (`src/mem2/branches/inference_engine/python_transform_retry.py:125-140`, `src/mem2/prompting/render.py:260-296`). The classify-the-failure-and-reformulate-retrieval mechanism is **[HYPOTHESIS]** — see §2.
3. **Substrate-aware retrieval** — the memory representation is tuned to the retrieval paper's native form (Zettelkasten notes for A-Mem, tree-position for RAPTOR/memtree, entity-passages for HippoRAG, etc.) rather than a flat concept list. **[CURRENT]** — the 16 per-port adapted-memory artifacts in `data/arc_agi/concept_memory/ports/` were built 2026-05-13 (commits `4138ec5` through `20a272c`); per-RN-007 audit, 9 of 16 ports are Faithful, 5 Partial-with-disclosed-gap, 2 Surface-port-only-disclosed.

The cost story riding underneath **[TARGET]**: hierarchical memory + a pre-retrieval gate + a multi-iter verifier-gated loop produce *more* small/cheap LLM calls (under a small/fast model) than a baseline that pays for one large/expensive call — but the *frontier* shifts: same accuracy at lower cost, or higher accuracy at the same cost. The cost-frontier evidence is not yet established empirically — Phase G headline-table protocol is required (see §6).

---

## 1 — The 6-axis ablation framework

The framework decomposes the design space into six orthogonal axes. An experiment varies exactly one axis at a time; baselines + a-handful-of-headline winners populate each axis. Phase G sweeps the full matrix at fixed seed × `n=50` for first-look, then `n=100 × 3 seeds × iters=3` for headline numbers.

| Axis | Question it answers | Baseline | Notable variants (today) |
|---|---|---|---|
| **1 — Retrieval** | Given a fixed flat memory, does a paper's *retrieval mechanism* change what the LLM gets? | `flat_topk` (top-k by cosine over concept descriptions) | `graphrag` (Leiden communities + hierarchical reports), `hipporag_ppr` (entity-graph + Personalized PageRank), `pathrag` (reliability-ordered relational paths), `raptor` (recursive tree descent), `hmem_hierarchical` (3-level category routing), `lightrag` (dense + sparse dual signal), `magma_multigraph` (typed multi-view), `colbert_rerank` (token-level rerank) |
| **2 — Reorganization** | Does the *memory builder* — what gets added/pruned/clustered as memory grows — change downstream retrieval signal? | `reorg_off` (memory never reorganized) | `accretive_prune` (capacity-pressure baseline, no LLM), `reorg_memtree`, `reorg_lilo`, `reorg_dreamcoder`, `reorg_lrll`, `reorg_amem` (Zettelkasten-style links), `reorg_memp` (procedural-card distillation), `reorg_sleepgate`, two MDL variants |
| **3 — Interactive** | Does *multi-round* query reformulation (with the agent observing a verifier signal between rounds) beat single-shot retrieval? | `one_shot` | `mediq_policy` (abstention + question-asking policy), `rrmc_multi_round` (multi-round retrieval + selector), `uot_entropy` (uncertainty-driven expansion) |
| **4 — Format** | Holding retrieval fixed, does the *prompt format* of retrieved concepts change LLM behavior? | `variant_minimal` | `variant_cue_heavy`, `variant_typed_only`, `variant_structured_routine`, `variant_free_text` (paragraph), `variant_parse` (per-kind to_string), `variant_dspy_opt`, `variant_gepa` |
| **5 — Metaedit** | Does an *LLM-driven* meta-edit on the memory (after consolidate) outperform hand-coded reorg policies? | `hand_coded_reorg` | `alma_style_metaedit`, `adas_style_search` (architecturally capped by consolidate-once — disclosed) |
| **6 — Initialization** | Does *seeded* memory (BARC concepts, HippoRAG-style corpus init) beat starting from an empty memory? | `empty_start` | `barc_seeded`, `barc_synthetic`, `corpus_hipporag_init` (external resources mostly blocked — disclosed) |

**Why these six.** Each axis isolates one degree of freedom of "memory-augmented agent" research:
- Axis 1 is the *retrieval-side* question (information surfaced).
- Axis 2 is the *builder-side* question (information stored).
- Axis 3 is the *interaction-side* question (how many turns the agent gets).
- Axis 4 is the *presentation-side* question (how surfaced information is rendered to the LLM).
- Axis 5 is the *self-improvement-side* question (does the agent rewrite its own memory).
- Axis 6 is the *initialization-side* question (does prior memory dominate accumulated memory).

The matrix is the structure of the paper. We do not run every (axis1 × axis2 × ...) cell — only one-axis-at-a-time vs the baseline, per axis. Cross-axis combination is reserved for a follow-up paper.

---

## 2 — Lead-novelty hypothesis: failure-typed within-task reformulation **[HYPOTHESIS — not yet implemented]**

After RN-007 audit + Phase G-lite v1 + the baselines mt16k mini-rerun cross-check, the v1 ranking (which had appeared to show axis-1 winners +19pp over baseline) **was a `max_tokens=2048` artifact**. At corrected `max_tokens=16384`, iters=1, the 8 baseline + winner conditions tested cluster at 47-51% accuracy — retrieval mechanism barely differentiates at single-shot. (Source: `case_studies/synthesis/2026-05-13_baselines_mt16k_results.md` when present; not yet committed as of this draft.)

The defensible novel claim **proposed for axis 3 (within-task loop)** — not implemented today:

```
PROPOSED within-task loop:
  iter 0: retrieve → attempt → verify (ARC training-grid execution)         [CURRENT — generic retry exists]
  iter 1: read failure type → reformulate retrieval query biased to failure-class → retry   [HYPOTHESIS]
  iter 2: same again, budget-bounded
```

The *failure type* would be a small discrete classification — `symmetry_mismatch`, `count_mismatch`, `color_mismatch`, `output_shape_mismatch`, `structural_mismatch`, `boundary_or_position_mismatch`, `unhandled_case`, `looks_correct_but_wrong_test_grid` — that drives retrieval reweighting via a `concept × failure_type` co-occurrence prior:

```
weight(c | query, failure_type) = cosine_sim(query, c) + λ · P(c_solved | failure_type)
```

**Current implementation reality:** none of this exists in runtime code as of `origin/main`. The hypothesis is captured in the origin thread `origins/threads/failure_typed_query/synthesis.md` (3 lines). The current retry path emits generic feedback (previous responses + execution errors + output mismatches + optionally reselected lessons; `src/mem2/branches/inference_engine/python_transform_retry.py:125-140`, `src/mem2/prompting/render.py:260-296`). There is no classifier, no failure-type label set in code, no `concept × failure_type` co-occurrence matrix.

**What's required to move this from HYPOTHESIS to CURRENT** (Phase G Track 2 work, per D-2026-05-13):
1. A `FailureTypologyClassifier` component (one LLM call per failure trace, returns one label from the 8-class set).
2. A `ports/<port>_memory_v1.json` schema extension persisting the `concept × failure_type` co-occurrence prior.
3. A retriever variant (axis 3) that reweights using the prior.
4. A three-arm ablation: `no-reflect` / `free-form-NL` (Reflexion-style) / `typed-failure`.

Closest prior art is Reflexion / ExpeL / ReasoningBank — all use **free-form NL reflection**. None use a discrete typology that drives retrieval-side reweighting. The structural change is small; the falsifiable comparison would be clean.

Substrate (per-port memory adaptation) and cost-frontier (hierarchical retrieval + pre-retrieval gate) ride underneath as supporting infrastructure, not as the headline.

---

## 3 — The cost story **[HYPOTHESIS — not yet empirically established]**

ArcMemo's wedge: hierarchical memory + cheap tree-walk retrieval → **same or better accuracy at lower total inference cost** as memory grows. **This is a research direction, not a current empirical claim.** The 16 per-port adapters + Phase G-lite v1 produced ranking data but no validated cost-frontier evidence.

The story has three components:

1. **Sublinear retrieval cost [HYPOTHESIS].** At a flat top-k = 3 over 270 concepts, retrieval is `O(270)` cosine sims (cheap). At memory scale ≥ 1000 concepts (post-cross-task accumulation), tree-walk retrieval is `O(log N)` (RAPTOR / memtree). The cost crossover is *theoretical* at today's 270-concept scale; whether it materializes at scale-1000 memory is open. Phase G headline run should grow memory across multi-pass iterations to test this.
2. **Pre-retrieval gate [HYPOTHESIS — not implemented].** Some questions don't need retrieval (Self-RAG / FLARE / Adaptive-RAG / MACLA prior art). The agent would decide per-question whether to fire retrieval at all, conditioned on its own uncertainty signal. **No pre-retrieval gate exists in mem2 today** — every retriever fires unconditionally per attempt. Adding the gate is a structural change to the runner + retriever Protocol, not just a parameter.
3. **Multi-iter retry as cheap calls [CURRENT runtime, HYPOTHESIS for cost-claim].** Each within-task iteration is one inference on the small/fast model (DeepSeek V4 Flash at ~$0.005/call). Three iters cost less than one expensive call on a frontier model. The verifier gate makes "did the cheap call work?" a deterministic check on ARC train-grid execution — no LLM judge needed. **The current Phase G-lite synthesis does not yet report per-condition spend** (every row's `Cost` column is `unknown`); the cost claim's evidence is therefore not in place even for the runtime path that exists.

**The cost-reporting contract [TARGET, not yet CURRENT]**: every results table SHOULD report accuracy + peak-tokens-per-problem + inference-time-per-problem. MEM1 (Wang+ 2024) Table 2 is the template. Reporting only accuracy loses the differentiator.

**Current artifact state:** `case_studies/synthesis/2026-05-13_phase_g_lite_results.md` reports accuracy + per-condition LLM call counts, but the `Cost` column is `unknown` per row (the provider metadata path that would populate it is not yet wired through). Peak-token-per-problem and inference-time-per-problem columns are absent. Before this becomes a paper-table contract, the harness needs to thread provider usage stats into `summary.json` and the synthesis renderer needs to surface those columns.

---

## 4 — Benchmark scope

- **ARC-AGI-1** (Chollet 2019 / ARC-AGI eval_100 split). The ArcMemo paper's split — apples-to-apples reproducibility. Establishes the mechanism. v1 SOTA is ~77-80% so headroom is narrow for frontier solvers, but the cost-frontier differentiator + the within-task loop ablations have room to land.
- **ARC-AGI-2** (Chollet 2025). The *primary results table*. SOTA ~24-29% means substantial headroom. Stable spec. This is where the headline numbers go.
- **ARC-AGI-3**: deferred to future work. Spec is in motion; appears interactive / agentic / skill-trace shape — may need a different memory primitive than concept-text. Discussing v3 as motivation for follow-up is acceptable; positioning a paper claim on v3 is not (the spec can move post-submission).

**Reviewer-defense line:** "v1 establishes mechanism; v2 carries the headline; v3 is the natural follow-up once the spec stabilizes."

---

## 5 — Substrate dependency chain

mem2's substrate is layered. The flat 270-concept base is the canonical input; everything else derives from it.

```
data/arc_agi/concept_memory/
├── compressed_v1.json                          ← THE FLAT BASE — 270 concepts, NEVER modified
│
├── shared/                                     ← shared substrates (built once, multi-port reuse)
│   ├── community_summaries_v1.json             ← Leiden communities over concept-entity graph
│   ├── entity_graph_v1.json                    ← LLM-extracted typed entities + co-mention edges (2569 entities × 4961 edges)
│   ├── hierarchical_reports_v1.json            ← 3-level hierarchical reports (33 reports × 3 levels)
│   ├── openie_facts_v1.json                    ← OpenIE-extracted S-P-O triples (995 triples)
│   ├── raptor_tree_v1.json                     ← recursive 2-level summary tree
│   ├── amem_link_graph_v1.json                 ← Zettelkasten-style inter-concept link graph
│   ├── lightrag_embed_v1.{npz,json}            ← dense embeddings for concepts + entities (1536-dim)
│   └── magma_typed_views_v1.json               ← typed multi-view (semantic / temporal / causal / entity)
│
└── ports/                                      ← per-port adapted memory (270 concepts × paper-native form)
    ├── hipporag_ppr_memory_v1.json             ← concepts as passages with entity-triples
    ├── hipporag2_memory_v1.json                ← concepts as filterable passages (Surface-tier per RN-007)
    ├── pathrag_memory_v1.json                  ← concepts with key entity-paths (Surface-tier)
    ├── graphrag_memory_v1.json                 ← concepts anchored to community reports (Partial-tier — test gap)
    ├── raptor_memory_v1.json                   ← concepts at tree leaves with path-to-root (Partial-tier)
    ├── amem_memory_v1.json                     ← concepts as atomic Zettelkasten notes
    ├── memtree_memory_v1.json                  ← concepts placed in 3-level hierarchy
    ├── hmem_memory_v1.json                     ← concepts with 3-level category routing
    ├── lightrag_memory_v1.json                 ← concepts with dual-graph membership
    ├── magma_memory_v1.json                    ← concepts with typed multi-view tags
    ├── mediq_memory_v1.json                    ← concepts with policy metadata (question-type)
    ├── rrmc_memory_v1.json                     ← concepts with multi-round selector metadata
    ├── uot_memory_v1.json                      ← concepts with entropy-reduction metadata
    ├── dreamcoder_memory_v1.json               ← concepts as program-card abstractions (Partial)
    ├── lilo_memory_v1.json                     ← concepts as routine-card abstractions (Partial)
    └── memp_memory_v1.json                     ← concepts as procedural-cards (Partial)
```

**The invariant chain:** `compressed_v1.json` is never written by any pipeline. `shared/` artifacts are built by per-substrate scripts under `scripts/prereq/shared/` and depend only on `compressed_v1.json`. `ports/<port>_memory_v1.json` artifacts are built by per-port adapters under `scripts/prereq/ports/<port>_adapter/` and depend on `compressed_v1.json` + the relevant `shared/` substrates. Retrievers under `src/mem2/branches/memory_retriever/<port>.py` prefer `ports/<port>_memory_v1.json` if present; fall back to `compressed_v1.json` if absent. Adaptation prompts that produced each `ports/` artifact are committed at `scripts/prereq/ports/<port>_adapter/prompt.md` (paper-informed, repo-versioned).

Per-port adaptation principle (Aaron 2026-05-13):

> "If we're doing hardcore retrieval, we need a hardcore memory. The concept memory is a flat list. Paper-port retrievers were designed around native structured memories. Retrieval-from-flat-list is not faithful to retrieval-from-paper-native-structure even if the retrieval logic looks right."

Design rationale + the 16-port adaptation table is in `mem_devlog/.copilot/decisions/D-2026-05-13_paper_port_faithful_memory_adaptation.md`.

---

## 6 — Validation flow **[TARGET — partially CURRENT]**

Experiments do not become claims without passing the following gates in order:

```
1. unit tests                       (per-port no-op replacement test + per-axis smoke)
   ↓
2. smoke sweep                      (45 conditions × N=3 problems × iters=1 × tracer-enabled)
   ↓
3. per-port case study              (manual eyeball of 2-3 problems per port — does retrieval bundle
                                     show the paper's native form?)
   ↓
4. adversarial review               (`/adversarial-review` Codex GPT-5.5 against the per-port
                                     adapters + retriever wiring + smoke sweep)
   ↓
5. Phase G first-look (mt=16384)    (45 conditions × n=50 × 2 seeds × iters=1 × cache=false ×
                                     max_workers=512)
   ↓
6. paired statistical comparison    (McNemar's test + bootstrap 95% CI per pair, Bonferroni-corrected
                                     over 990 pairs)
   ↓
7. Phase G headline (mt=16384)      (selected conditions × n=100 × 3 seeds × iters=3)
   ↓
8. final adversarial review         (against headline numbers + Report 4 narrative)
   ↓
9. paper claim / hub.md update      (only after step 8 clears)
```

**Today's state (2026-05-13), split by artifact run:**

- **Phase G-lite v1 artifact run (mt=2048)** — steps 1, 2, 3, 4, 5, 6 completed. Step 5 surfaced the `max_tokens=2048` artifact: the run's absolute scores are uniformly suppressed 5-6× vs historical Phase A protocol; the v1 ranking is now treated as a *protocol-bug-disclosure* artifact, not as the headline first-look. The paired-statistics artifact at `case_studies/synthesis/2026-05-13_phase_g_lite_paired_stats.md` IS Bonferroni-corrected over 990 pairs, but the underlying scores are at mt=2048 and so the paired-significance markers apply to the artifact run, not to a corrected-protocol headline.
- **Phase G-lite v2 artifact run (mt=16384)** — **not yet completed**. Baselines mt16k mini-rerun (8 conditions) confirmed the +31.5pp average magnitude shift; full 45-condition rerun gated on Phase G shape decision (T027 in `todo.jsonl`).
- **Step 7 headline run** — not yet started.

The validation flow is not a paper requirement — it's a *structural* defense against the failure mode that triggered Path C (paper ports claimed Faithful but mechanism never reached output at runtime). See `DESIGN.md` §5 for the no-op replacement test that enforces step 1, and `mem_devlog/.copilot/reviews/RN-007_codex_per_port_adapter_audit_2026-05-13.md` for the audit verdict on the current 16 ports.

**Gate enforcement is CURRENTLY convention only.** A human or copilot could update hub.md's State of Knowledge or set `parity_grade: faithful` in `configs/axes/<N>.yaml` without having actually run the gates. DESIGN.md §8.4 logs this as a backlog item: a pre-commit hook that requires an RN-NNN reference for any Faithful claim within 7 days.

---

## 7 — What this architecture does NOT claim

Explicit non-claims, so readers don't over-interpret:

- **Not "this retrieval method is SOTA."** At iters=1 with mt=16k, retrieval method choice barely differentiates (observed in baselines mt16k mini-rerun). The paper claim lives in axis 3 + the within-task loop, not axis 1 alone.
- **Not "memory architecture choice is irrelevant."** Multi-iter eval (iters=3) is the regime where memory differentiates; iters=1 is a diagnostic regime that proves retrievers engage, not that they are useful.
- **Not "this generalizes to all reasoning tasks."** The validation scope is ARC-AGI-1+2. Generalization to math / code / dialog is future work and explicitly out of scope for the paper.
- **Not "every port is Faithful to its paper."** Per RN-007: 9 Faithful + 5 Partial-with-disclosed-gap + 2 Surface-port-only-disclosed + 0 Wrong-undisclosed. Surface and Partial ports' substrate gaps are documented in their adapter READMEs.
- **Not "cache=false is the production protocol."** Phase G-lite ran cache=false to isolate ranking. Headline Phase G may use cache for cost reasons, with disclosure.

---

## 8 — Cross-references

- `DESIGN.md` — the codebase counterpart (how the structure enforces what this doc claims).
- `mem_devlog/.copilot/hub.md` — the live State of Knowledge (numbers update over time).
- `mem_devlog/.copilot/decisions/D-2026-05-13_paper_port_faithful_memory_adaptation.md` — the per-port adaptation principle + the 16-port table.
- `mem_devlog/.copilot/reviews/RN-007_codex_per_port_adapter_audit_2026-05-13.md` — the audit that grades each port.
- `origins/threads/` — per-research-thread origin notes (interactive_retrieval is Aaron's prior unpublished work; failure_typed_query is scholar's reframe).
- `case_studies/synthesis/2026-05-13_phase_g_lite_results.md` — first-look ranking at mt=2048 (preserved for v1-vs-v2 protocol-shift comparison).
- `case_studies/synthesis/2026-05-13_phase_g_lite_paired_stats.md` — paired McNemar + bootstrap CI over 990 condition pairs (Bonferroni-corrected). Stats apply to the mt=2048 artifact run, not to a corrected-protocol headline.
- `case_studies/synthesis/2026-05-13_baselines_mt16k_results.md` — magnitude-corrected baselines mini-rerun (8 conditions). **Status as of this doc draft: ping-back reported the file but it is not yet present in the worktree; the v1-vs-v2 magnitude shift was reported as +31.5pp average across the 8 conditions. Verify the file exists before citing as a canonical anchor.**
- `mem_devlog/deliverables/reports/04_2026-05-13_post_adapter_substrate_results.md` — Report 4 (scholar-drafted, in progress).

---

## 9 — Naming decisions (revertable)

- **"port"** — used throughout for "a paper's method adapted into mem2's contract." A port is graded Faithful / Partial-with-disclosed-gap / Surface-port-only-disclosed / Wrong-undisclosed (RN-007 grading rubric).
- **"adapted memory"** — `ports/<port>_memory_v1.json` is the *adapted* version of `compressed_v1.json` for a specific port. Not "preprocessed," not "transformed" — *adapted* signals the paper-native-form rewrite.
- **"shared substrate"** — `shared/` artifacts that multiple ports consume (entity graph, community summaries, etc.). Built once, indexed by version.
- **"thread"** — a research direction in `origins/threads/` (interactive_retrieval, hierarchical_memory, failure_typed_query, etc.). Encapsulates origin material + cross-references for one direction of inquiry.

---

## 10 — Open architectural questions

These shape the next decisions but are not yet settled:

1. **Phase G shape** — iters=1 (confirm null), iters=3 (chase real signal), or accept null + reframe Report 4? (Open: T027 in `todo.jsonl`.)
2. **magma_multigraph anomaly** — 3% at v1 (mt=2048) is well below floor. Still unexplained. May be wiring bug, may be substrate over-filtering, may be token-cap artifact that resolves at mt=16k. Investigate once Phase G shape decision is made.
3. **Cross-axis combination** — when does axis 1 × axis 3 (retrieval × interactive) yield benefits that single-axis doesn't surface? Reserved for follow-up.
4. **Memory scale for cost-claim binding** — at 270 concepts, flat top-k is already cheap; hierarchical retrieval's cost win is theoretical. Phase G should grow memory to 1000+ concepts (multi-episode lifecycle) for cost claim to bind empirically.
