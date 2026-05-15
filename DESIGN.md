# mem2 — Codebase Design

**How the code is structured so experiments cannot fail on the principle of control.**

> **Status**: canonical, continuously refined — the codebase-design counterpart to `ARCHITECTURE.md`.
> - `ARCHITECTURE.md` = what the *method* is (the research claim — online lifelong memory-augmented agent, 6-axis ablation framework, failure-typed reformulation as proposed lead novelty).
> - `DESIGN.md` (this doc) = how the *codebase* is structured (the run system, the contracts, file layout, enforcement).
> Iterate on this as design decisions land. Migration history + per-decision rationale live in `mem_devlog/.copilot/`.

### Three-state labeling convention (read first)

Throughout this document, claims fall into three states that must be distinguished — and the doc is structured so that mixing them is the canonical doc-bug.

- **[CURRENT]** — implemented and exercised by today's mem2 runtime (`origin/main` as of §9's migration date).
- **[TARGET]** — proposed structural design; not yet implemented. Lives in §8 (deferred / gaps).
- **[HISTORICAL]** — a past violation that motivated a rule. Documented here for context; not a current bug.

Sections may mix states. When they do, claims are tagged inline. When a section's overall tag is absent, treat as **[CURRENT]**.

---

## 0 — The core principle

**Experimental control is a property of the code structure, not of discipline.**

This is the rule mem2 *targets*. Today's codebase satisfies it partially: some bug classes are structurally prevented, others remain convention-only with documented gaps (§8). The doc is a target, with §8 logging the gap between target and current.

### What motivated the rule (history)

- **`max_tokens=2048` bug, Phase G-lite v1, 2026-05-13 [HISTORICAL]:** convention said "use `max_tokens=16384` for ARC, matching Phase A protocol"; the sweep harness defaulted to `2048`; nothing in code enforced the convention. ~$25 of 90-condition runs were artifactually suppressed before a sanity-check spotted the magnitude shift.
- **A-Mem heuristic fallback, 2026-05-13 [HISTORICAL]:** an earlier A-Mem adapter build silently substituted 40% of records when LLM extraction failed (RN-007 finding). The current adapter (`scripts/prereq/ports/amem_adapter/build_adapted_memory.py`) retries 3x then raises — the violation is fixed; the rule it motivated stays.
- **Cache-strip-empties [CURRENT VIOLATION]:** `third_party/llm_wrapper/llmplus/client.py:115` strips empty completions before caching → next call with same prompt hits a cache miss + silent retry → score inflates. Convention says "the cache is transparent"; the strip-before-store path violates that silently. **Still present as of `origin/main`** — see §8.2.
- **Surface-port mechanism gap [CURRENT, disclosed]:** hipporag2 + pathrag load adapted memory at runtime, and the adapted records are *partially* load-bearing (token-overlap-into-scoring for hipporag2; seed-text + render for pathrag), but the paper's full mechanism (BERT/DSPy filter loop for hipporag2; reliability-scored path selection from artifact paths for pathrag) is not implemented. RN-007 graded these Surface-port-only-disclosed; the substrate gap is in the adapter README.

### The target (this doc's claim)

mem2's structural answer: **make the recurring bug classes impossible by construction.** Memory builders, retrievers, and runner configs should be structurally constrained so the convention either holds or the system fails loud at launch. Discipline becomes a backstop, not a primary defense.

Today's reality is partial — §8 lists 6 structural items that are still convention-only.

### Borrowing acknowledgment

The principle is borrowed verbatim from triad's `DESIGN.md` §0 — a sibling project (RRMC → triad migration) that learned the same lesson on its own NPC bug and codified it. Bidirectional flow: triad borrowed mem2's `benchmarks/ + origins/` external-material layout; mem2 borrows triad's structural-control framing. mem2's per-condition sealed config (§2) is the triad-style "model gate fails loud at launch" — described here as **[TARGET]**, not yet implemented.

---

## 1 — The three-leg contract **[CURRENT — signatures verified against contracts.py]**

mem2's runtime is three Protocols + a runner that orchestrates them. The runner is the only component that imports all three. (Cross-leg import sealing is a convention, not yet structurally enforced — see §8.3.)

```
                   ┌────────────────────────────────┐
                   │   orchestrator/runner.py        │
                   │   the STATIC skeleton           │
                   │   — identical bytes per cond —  │
                   └──────┬─────────┬─────────┬──────┘
              imports all 3        │         │
        ┌──────────────┘           │         └──────────────┐
        ▼                          ▼                        ▼
  ┌──────────────────┐    ┌────────────────────┐    ┌────────────────────┐
  │ memory_builder/  │    │ memory_retriever/  │    │ inference_engine/  │
  │ ░ writes memory ░│    │ ▒ reads memory  ▒ │    │ ▓ runs LLM call  ▓ │
  └──────────────────┘    └────────────────────┘    └────────────────────┘
```

### 1.1 Memory builder — three-method Protocol

Signatures (from `src/mem2/core/contracts.py:34-49`):

```python
class MemoryBuilder(Protocol):
    name: str
    def initialize(self, ctx, problems) -> MemoryState: ...
    def update(self, ctx, memory, attempts, eval_records, feedback_records) -> MemoryState: ...
    def consolidate(self, ctx, memory) -> MemoryState: ...
```

The runner's call pattern (`src/mem2/orchestrator/runner.py:726-732`, `:884-950`):
- `initialize` once per run with the full problem set, returns initial `MemoryState`.
- `update` once per problem after attempts complete, with the attempt history + eval results + feedback. Returns a new `MemoryState`.
- `consolidate` **once per run, AFTER the entire pass loop ends** (`runner.py:884-948` loops over `pass_idx`; `runner.py:949-950` calls `consolidate(ctx, memory)` once after the loop). Per-pass consolidate is NOT the current pattern; if future intent is to consolidate per-pass, that becomes [TARGET].

**The builder never writes files.** It returns a `MemoryState`; the runner persists. This makes "builder wrote the wrong field" structurally hard — the runner's writer module is the only path to disk. **[CURRENT, convention-enforced]** — there is no static check that a builder doesn't open a file handle; CI lint is **[TARGET]** (see §8.3).

**The builder is forbidden by convention from importing memory_retriever modules.** Cross-leg communication flows through the in-memory `MemoryState` typed object. Static enforcement is **[TARGET]** — see §8.3.

> **Proposed hardening [TARGET]: `MemoryDiff` return type.** Today `update` and `consolidate` return a fresh `MemoryState` — the runner cannot tell which fields changed without diffing. A future signature `update(...) -> MemoryDiff` (where `MemoryDiff` is a typed delta: `concepts_added`, `concepts_modified`, `concepts_pruned`) would make the change-set explicit + enable per-builder audit. **Not implemented as of `origin/main`.** Logged in §8 as a structural-hardening item.

### 1.2 Memory retriever — `retrieve` + `async_retrieve` Protocol

Signatures (from `src/mem2/core/contracts.py:51-71`):

```python
class MemoryRetriever(Protocol):
    name: str
    def retrieve(self, ctx, memory, problem, previous_attempts) -> RetrievalBundle: ...
    async def async_retrieve(self, *, ctx, provider, memory, problem,
                              previous_attempts, selector_model="") -> RetrievalBundle: ...
```

Per-attempt: takes `MemoryState` + problem + prior attempts, returns a typed `RetrievalBundle` (retrieved items + bundle.metadata describing what fired). The bundle is the only thing the inference engine sees — the retriever's internal state never crosses into the LLM call.

**The retriever returns a typed object.** No raw LLM text, no untyped dict — `RetrievalBundle` with declared fields (defined at `src/mem2/core/entities.py`). Conversion from internal data structures to bundle happens *inside* the retriever; the runner sees only the typed object.

**The retriever LOADS adapted memory at runtime [CURRENT for 16 ports].** Per-port retrievers check for `data/arc_agi/concept_memory/ports/<port>_memory_v1.json` at init. If present, load it. If absent, fall back to `compressed_v1.json`. This contract is enforced by per-port unit tests (`tests/unit/test_<port>_adapted_memory.py`) — the test FAILS if the retriever uses `compressed_v1.json` when adapted memory exists.

**Failure handling is target-rule + current-violations.** §3 below covers the full picture. Summary: a failed `retrieve()` SHOULD raise. Current code mostly does — but inference-job failures elsewhere (in the runner orchestration) silently produce empty `attempts = []` and continue (see §3.2 and §8.2).

### 1.3 Inference engine — `initial_attempt` + `retry_attempt` Protocol

Signatures (from `src/mem2/core/contracts.py:110-135`):

```python
class InferenceEngine(Protocol):
    name: str
    model: str
    include_reselected_lessons: bool
    def set_retry_policy(self, policy) -> None: ...
    async def initial_attempt(self, ctx, provider, problem, retrieval,
                               trajectory_plan) -> list[AttemptRecord]: ...
    async def retry_attempt(self, ctx, provider, problem, retrieval,
                             attempt_history, feedback_history,
                             trajectory_plan) -> list[AttemptRecord]: ...
```

The actual LLM call sites. The engine has two methods: `initial_attempt` for iter-0; `retry_attempt` for iter-1+. Each returns a list of `AttemptRecord` objects (typed).

**The multi-iter loop is in the runner, not the engine [CURRENT].** The runner (`src/mem2/orchestrator/runner.py:884-948`) drives the `pass loop`: for each pass, it calls `initial_attempt` then optionally `retry_attempt` based on the configured `retry_policy`. `consolidate` is called once after the pass loop ends (`runner.py:949-950`).

**The engine is the only LLM-touching component in the within-task runtime path.** Builders and retrievers do not fire LLM calls during eval passes (they may at *build time* — `scripts/prereq/` — but never inside a pass). This isolates the LLM call site for cost tracking, retry policy, and the per-call tracer (see §6).

**Within-task failure classification + reformulation is not implemented.** The current `retry_attempt` renders a generic retry prompt (previous responses + execution errors + output mismatches + optionally reselected lessons; `src/mem2/branches/inference_engine/python_transform_retry.py:125-140`, `src/mem2/prompting/render.py:260-296`). There is no failure-type classifier, no discrete-label-set in code, no `concept × failure_type` co-occurrence prior. The proposed implementation is described in ARCHITECTURE.md §2 as **[HYPOTHESIS]** for Phase G Track 2.

---

## 2 — Per-condition sealed config

**Every condition declares its model + max_tokens + cache flag + every other LLM-affecting parameter explicitly. The runner fails loud at launch if any are ambiguous, missing, or default-inherited.**

This is the direct structural fix for today's `max_tokens=2048` bug.

### 2.1 The declaration contract

Every condition's YAML or programmatic config declares:

```yaml
# configs/conditions/<axis>/<condition>.yaml  (proposed canonical location)
condition_id: "graphrag"
axis: 1
memory_builder: arcmemo_ps          # or null for retrieve-only conditions
memory_retriever: graphrag
inference:
  model: deepseek/deepseek-v4-flash
  max_tokens: 16384                  # MUST be declared; no default leakage
  temperature: 0.3
  cache: false                       # MUST be declared; no default leakage
  retry_policy:
    max_retries: 3
    backoff_seconds: [2, 4, 8]
adapter:
  ports_memory: data/arc_agi/concept_memory/ports/graphrag_memory_v1.json
  shared_substrate: [community_summaries_v1, hierarchical_reports_v1]
parity_grade: faithful
parity_note: ""
```

### 2.2 The runner-level gate

At launch, the runner validates:

1. Every `condition_id` referenced by the sweep has a config file.
2. Every config file declares every field in the schema (no missing keys, no implicit defaults).
3. Every field that *would* default if omitted raises a `MissingDeclaration` error instead.
4. For cross-condition comparison runs, the runner verifies that *non-experimental-axis* fields are identical across conditions (e.g., when sweeping axis 1, the `inference.model` and `inference.max_tokens` MUST be the same string across all axis-1 conditions; otherwise the sweep is unfair by construction).

This makes today's bug structurally impossible: if `max_tokens` were missing from `graphrag.yaml`, the runner would refuse to launch. If `graphrag.yaml` declared `max_tokens: 2048` but `flat_topk.yaml` declared `max_tokens: 16384`, the cross-condition gate would refuse to launch.

### 2.3 The harness override surface

Today's `sweep_all_axes.py` accepts CLI flags (`--max-tokens`, `--cache`, `--iters`, `--seeds`). These OVERRIDE the per-condition config — but the override is *recorded* in the run's `meta.json` and *uniform* across all conditions in the sweep. The harness cannot silently apply different overrides per condition.

When in doubt, the *condition's declared value* wins. The CLI override is a sweep-wide knob, not a per-condition knob.

### 2.4 Status — what's structural today

**Today:** the config gate is **NOT yet implemented**. Conditions are defined in `configs/axes/<N>.yaml` as a list-of-dicts; the harness applies CLI defaults; the runner does not validate per-condition completeness.

**This is the highest-priority structural-hardening item.** Until it lands, today's bug class can recur. See §8 — deferred / gaps.

---

## 3 — No repair-parser rule **[TARGET — with current violations]**

**Target rule:** if a builder or retriever cannot construct a valid output, it RAISES. No `except: continue`, no fallback-to-zero, no heuristic-substitute-and-keep-going.

This is borrowed from triad `DESIGN.md` §3.1: *"A repair-parser is a silent-failure-masker."* In mem2 the rule is **partially in place** — some build scripts now satisfy it; others and one path in the runner still violate. §8.2 logs the residual violations as backlog.

### 3.1 What motivated the rule (HISTORICAL)

**A-Mem heuristic fallback (HISTORICAL — fixed):** An earlier A-Mem adapter build silently substituted heuristic links when LLM extraction failed:

```python
# WRONG — silently substitutes when LLM fails (this pattern is now REMOVED from amem_adapter)
try:
    links = await llm_extract_zettel_links(concept, related_concepts)
except Exception:
    links = deterministic_lexical_and_co_use_fallback(concept, related_concepts)
    rationale = "deterministic lexical and co-use fallback"
    confidence = 0.5
```

The 2026-05-13 audit (RN-007) found 110/270 concepts received the heuristic fallback in an early A-Mem rebuild — 40% of the adapter's output was heuristic-faked. **The current adapter (`scripts/prereq/ports/amem_adapter/build_adapted_memory.py:281-301`) retries 3x and raises `RuntimeError` after the third failure; the artifact is only assembled and written when all 270 results succeed (`:326-359`). The adapter README explicitly documents the no-heuristic-fallback contract (`README.md:45-58`).** The historical violation motivated the rule; it does not represent current behavior.

**The right pattern (which the current A-Mem adapter implements):**
```python
async def adapt_one(concept, dependencies):
    for attempt in range(3):
        try:
            return await llm_extract(concept, dependencies)
        except RetryableError:
            await asyncio.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"LLM failed 3x on concept {concept.id} — adapter aborting")
```

The whole adapter aborts. The artifact is not written. The sweep is not falsely greenlit. The human investigates.

### 3.2 Current violations [CURRENT — to remove]

**Cache-strip-empties** (`third_party/llm_wrapper/llmplus/client.py:115`):
```python
# WRONG — strips empty completions before storing
new_nonempty_responses = [r for r in fetched if r and r.strip() != ""]
self._resp_cache[cache_key] = cached + new_nonempty_responses
```

The empty completion is part of the cache's truth. Stripping it means the next call with the same prompt hits a "cache miss" on the previously-empty problem and silently re-fires the LLM. The score inflates by the retry-success rate. RN-003 root-caused this; the fix is to store the empty completion as a valid cache entry (or to never cache empties at all, but consistently — same on read and write). **Still in place as of `origin/main`.**

**Runner inference-job exception masking** (`src/mem2/orchestrator/runner.py:776-815`): when `_run_jobs_with_progress` returns an `Exception` for an inference job, the runner emits an error event, sets `attempts = []`, and still calls `_finalize_problem_result`. The run continues, the problem is recorded as "no attempt," the per-problem `correct` flag is False, and the aggregate is computed as if the problem genuinely scored 0. This is a continue-on-error path, not a stop-on-error path — the target rule says it should stop and let the human investigate.

**OpenIE shared-substrate partial-artifact write** (`scripts/prereq/shared/openie_facts/build_concept_facts_openie.py:364-423`): per-concept extraction exceptions are caught; a failure is recorded; zero cleaned facts are appended; the loop continues. The script then writes the artifact with `num_failures > 0` and returns success if there are any facts at all. The checked-in `data/arc_agi/concept_memory/shared/openie_facts_v1.json` records `num_failures: 6` with six named source concepts that produced no facts. The target rule says the build should raise + the artifact should not exist with `num_failures > 0`.

### 3.3 What "raise" means in practice [TARGET]

When the target rule is fully implemented:

- The runner catches the raise + records a structured error in `runs/<run_id>/errors/<attempt_idx>.json`.
- The run stops at the failed attempt (does not skip and continue).
- The human or copilot inspects the error + decides: fix the input, fix the parser, or document the limitation.
- No condition is allowed to ship with > 0 documented failures without an explicit `parity_note:` disclosing the failure rate.

The runner's inference-job exception path needs to be rewritten to satisfy this (see §8.2). The OpenIE builder needs to either raise on partial-failure or be rewritten to retry-3x-then-raise.

### 3.4 The exception: graceful retrieval fallback [CURRENT]

The *one* allowed fallback in retrievers is **`ports/<port>_memory_v1.json` not present → fall back to `compressed_v1.json`.** This is graceful, declared, and tested (per `tests/unit/test_<port>_adapted_memory.py`). It is the only structural deviation from the no-repair-parser rule.

The reason it is allowed: the absence-of-artifact case is *itself* the failure signal. The retriever explicitly checks `if artifact_path.exists()` and the user has explicit configuration: "I haven't built the adapter yet." Compare with the heuristic-fallback case, where the artifact exists but its content is heuristic-substituted *silently inside* the file.

---

## 4 — The substrate split (data writers sealed by directory)

`data/arc_agi/concept_memory/` is split into three roles, with structural rules about who writes what:

```
data/arc_agi/concept_memory/
├── compressed_v1.json     ← THE FLAT BASE — 270 concepts. NEVER WRITTEN by any pipeline.
│
├── shared/                ← shared substrates. Written ONLY by scripts/prereq/shared/<substrate>/.
│   ├── community_summaries_v1.json
│   ├── entity_graph_v1.json
│   ├── ... (see ARCHITECTURE.md §5)
│
└── ports/                 ← per-port adapted memory. Written ONLY by scripts/prereq/ports/<port>_adapter/.
    ├── graphrag_memory_v1.json
    ├── ... (16 ports — see ARCHITECTURE.md §5)
```

### 4.1 The write-path invariant

| Directory | Who writes | Who reads | Build-time vs runtime |
|---|---|---|---|
| `compressed_v1.json` | Imported from `arc_memo` repo. Treated as read-only. | Every retriever fallback; every adapter build script | Build-time (frozen) |
| `shared/*.json` | `scripts/prereq/shared/<substrate>/build_*.py` (one script per substrate) | Multi-port — `shared/community_summaries` consumed by graphrag + raptor; `shared/entity_graph` consumed by 5+ retrievers | Build-time (one-shot LLM batch) |
| `ports/<port>_memory_v1.json` | `scripts/prereq/ports/<port>_adapter/build_adapted_memory.py` (one script per port) | Exactly one retriever — `src/mem2/branches/memory_retriever/<port>.py` | Build-time (per-port LLM batch) |
| In-memory `MemoryState` | The runner (applies builder returns) | The retriever (consumes for next attempt) | Runtime (per-attempt) |

**The structural rule:** a retriever may only READ from `compressed_v1` + its own `ports/<port>_memory_v1` + relevant `shared/*` substrates. A retriever may not write to any of these. A builder returns updated `MemoryState` objects (or eventually `MemoryDiff` per §1.1 TARGET); the runner persists the in-memory state but does not write to `compressed_v1`, `shared/`, or `ports/`.

This makes the bug class **"the retriever modifies the adapted memory mid-run and a later condition reads the modified version"** structurally impossible.

### 4.2 The per-port adapter convention

Every adapter directory under `scripts/prereq/ports/<port>_adapter/` contains:

```
scripts/prereq/ports/<port>_adapter/
├── build_adapted_memory.py    ← the build script (uses LLM)
├── prompt.md                  ← the paper-informed conversion prompt (COMMITTED)
├── README.md                  ← paper citation + schema + substrate-gap disclosure
└── (no other files)
```

The `prompt.md` being committed is structural: it makes the "we re-ran the LLM and got different results" failure mode auditable. If the artifact is regenerated, the prompt that produced it is in git.

---

## 5 — The no-op replacement test

**Per-port: there is a unit test that FAILS if `retrieve()`'s body is replaced with `return memory` or `pass`.**

This is the structural invariant that distinguishes a Faithful port from a Surface port. The mechanism *must* affect the output.

### 5.1 The canonical test shape

```python
# tests/unit/test_<port>_adapted_memory.py

def test_<port>_adapted_scoring_changes_selection():
    """Replacing the adapted-scoring path with a no-op MUST change top-1 selection."""
    fixture_artifact = ...  # adapted memory with distinctive scoring signal
    
    bundle_with_adapted = run_<port>_with_artifact(fixture_artifact, query="...")
    bundle_no_adapted = run_<port>_without_artifact(query="...")
    
    assert bundle_with_adapted.metadata["scoring_mode"] == "<port>_adapted_memory"
    assert bundle_no_adapted.metadata["scoring_mode"] in ("<port>_fallback", "ps_topk")
    
    # CRITICAL: selection must differ. If they're the same, the adapted-scoring
    # code path is a no-op and the test fails.
    assert bundle_with_adapted.retrieved_items[0]["name"] != bundle_no_adapted.retrieved_items[0]["name"], \
        "adapted-scoring path produced same top-1 as fallback — scoring is no-op"
```

### 5.2 Today's coverage **[CURRENT]**

Per RN-007 + 2026-05-13 remediation:
- **9 Faithful ports**: pass the no-op replacement test cleanly. Adapted-scoring path is exercised by the test + would fail on a `return memory` no-op.
- **5 Partial-with-disclosed-gap ports**: pass a *weaker* version of the test (the fallback path is tested, not the primary; documented in adapter README). graphrag + raptor specifically had this gap flagged by RN-007 + were remediated 2026-05-13 with stronger tests, but the architectural cap (e.g., dreamcoder, lilo, memp where the substrate gap is fundamental) remains disclosed.
- **2 Surface-port-only-disclosed (hipporag2, pathrag)**: adapted memory is **partially load-bearing but Surface-tier relative to the paper.** hipporag2 loads adapted records and renders adapted hints (`src/mem2/branches/memory_retriever/hipporag2.py:107-124`); its template filter mixes tokens from adapted record text into candidate scoring (`:217-248`). pathrag uses adapted record text for seed-node scoring (`src/mem2/branches/memory_retriever/pathrag.py:115-128`) and renders adapted path summaries (`:166-188`). What's *missing* relative to the paper: hipporag2's default second-stage filter is template token overlap rather than the paper's BERT/DSPy loop; pathrag's runtime path selection enumerates `ConceptGraph` paths rather than using the artifact's reliability-scored paths as the primary path source. The substrate gaps are disclosed in each adapter's README "Substrate gap" section. These ports are explicitly *not* claimed Faithful.

The no-op replacement test is the structural defense against the bank-write-retriever-blind class of bug. Without it, a port can ship as "Faithful" while its mechanism never reaches the output.

---

## 6 — Per-puzzle per-call trace store

**Every LLM call writes its trace to disk immediately. Each call writes its own small directory; no shared-file lock; no shared-file rewrite per call.**

This is the contention-free incremental save pattern.

```
case_studies/runs/<run_id>/                          ← run_id = ISO_<benchmark>_<port>_n<N>_seed<S>_<label>
├── meta.json                                        ← reproducibility — port, seeds, model, mt, cache flag
├── summary.json                                     ← run-level — score, n_correct, n_total, per-problem
├── summary.md                                       ← human-readable rendered version
└── problems/
    └── <task_id>/
        └── iter_<N>/
            ├── prompt.txt                           ← the actual rendered LLM prompt
            ├── response.txt                         ← the LLM response (full text)
            ├── retrieval_bundle.json                ← bundle.metadata + retrieved items
            ├── eval.json                            ← scoring outcome
            ├── call_meta.json                       ← model, max_tokens, ignore_cache, latency, token_usage
            ├── parsed.json                          ← parsed Python transform code
            └── llm_calls/call_NNNN/                 ← per-call subdirs when iter has multiple inner calls
```

### 6.1 Why contention-free matters

A 90-condition × 50-puzzle × 3-iter sweep produces ~13,500 LLM calls. If each call wrote to a shared `transcript.json`, the file would be rewritten 13,500 times under concurrent workers — file-lock contention would dominate wall time, and a crash mid-write would corrupt the run.

The per-call-own-directory pattern means each worker writes to a unique path. No locks. No rewrites. A crash leaves everything up to the last completed call already persisted.

### 6.2 The tracer is opt-in

The runner only writes traces if `provider.trace_dir` is configured. When unset (default), zero file I/O for the tracer — no perf hit on production sweeps that don't need per-call traces.

For Phase G-lite + case studies, the tracer is on. For headline Phase G (where wall time matters and the per-call traces are 1.7GB / sweep), the tracer is on but written to a deletable scratch dir, with compact summaries persisted to `case_studies/synthesis/`.

### 6.3 Resume mode (deferred)

A future addition: `runner --resume <run_id>` reads the on-disk trace, skips completed (puzzle, iter) pairs, and continues. Two uses:

1. **Crash recovery** — read the trace, skip completed work, continue. The per-call-own-directory pattern is what makes this safe.
2. **Counterfactual investigation** — load a finished run, flip one knob (e.g., max_tokens, retriever, prompt template), re-run from the edit point. This is the "counterfactual" case-study mode.

Status: DEFERRED. The infrastructure is in place (incremental save + per-call dirs); the resume harness is not yet built. Likely the natural follow-up for the magma anomaly investigation.

---

## 7 — Module layout

```
mem2/
├── ARCHITECTURE.md  DESIGN.md  README.md  pyproject.toml
├── src/mem2/
│   ├── core/
│   │   ├── contracts.py             ← Protocols: MemoryBuilder, MemoryRetriever, InferenceEngine
│   │   ├── entities.py              ← typed records: MemoryState, RetrievalBundle, AttemptRecord, FeedbackRecord, EvalRecord (MemoryDiff = TARGET per §1.1)
│   │   ├── errors.py                ← fail-loud error types (MissingDeclaration, BuildFailed, …)
│   ├── branches/
│   │   ├── memory_builder/          ← 23+ builders (accretive_prune, reorg_*, variant_*, arcmemo_*, ...)
│   │   ├── memory_retriever/        ← 18+ retrievers (ps_topk, graphrag, raptor, hipporag_*, ...)
│   │   ├── inference_engine/        ← arc_python_transform, arc_python_transform_retry
│   │   ├── feedback_engine/
│   │   ├── task_adapter/
│   ├── concepts/                    ← Concept dataclass, ConceptGraph, MemoryState
│   ├── orchestrator/runner.py       ← the static skeleton; multi-round retrieval loop
│   ├── prompting/                   ← prompt templates (renderer for the LLM call)
│   ├── providers/                   ← LLM provider adapters (llmplus, OpenRouter)
│   ├── registry/                    ← name → class lookups for builders/retrievers
│   └── {retrieval, scoring, analysis, cli, io, utils}/
│
├── configs/                         ← declarative: axes/, conditions/, experiments/, parity/, ...
├── data/arc_agi/concept_memory/     ← compressed_v1 + shared/ + ports/ (§4)
├── scripts/prereq/
│   ├── shared/                      ← shared substrate builders
│   └── ports/                       ← per-port adapter builders (one dir per port)
├── case_studies/                    ← per-run traces + analysis modes + synthesis
│   ├── runs/<run_id>/               ← per-puzzle per-call trace store (§6)
│   ├── synthesis/                   ← cross-run aggregates + paired stats
│   ├── scripts/
│   ├── _tracer/                     ← opt-in TraceCollectingProviderClient
│   ├── by_method/                   ← per-port curated views (symlinks)
│   └── modes/                       ← 7 case-study modes (error/comparative/counterfactual/...)
├── benchmarks/                      ← cross-benchmark harness (arc_agi/ + _placeholder/)
├── analysis/                        ← failure_taxonomy/, memory_growth/, retrieval_telemetry/, provenance/
├── origins/                         ← research-thread origins + per-paper distillations + future/
├── outputs/                         ← run outputs (sweeps land here pre-aggregation)
├── literature/                      ← paper PDFs (read-only inputs)
├── third_party/                     ← reference repos (read-only for inspiration; vendored deps)
└── tests/{unit, smoke, sweeps}/
```

Tier 1 expansion landed 2026-05-13 (commits `b98176f`, `4c03848`, `9cf79b7`):
- `benchmarks/` — cross-benchmark harness placeholder + 9 future-benchmark READMEs
- `analysis/` — 4 sub-modules with stub extractors + cross-ref to case_studies/
- `origins/` — research-thread tracking + per-paper distillation home + external-repo notes

Per-port adapter convention landed 2026-05-13 (commits `4138ec5` through `20a272c`):
- `scripts/prereq/ports/<port>_adapter/` × 16 ports
- `data/arc_agi/concept_memory/ports/<port>_memory_v1.json` × 16 artifacts

---

## 8 — What is NOT yet structural (deferred / gaps)

The previous sections describe the *target* structure. These items are deferred — convention-only today, structural enforcement on the backlog.

### 8.1 Per-condition sealed config (§2)

**Not implemented.** This is the highest-priority structural-hardening item. Until landed, today's `max_tokens=2048` bug class can recur.

Recommended scope:
1. Define `configs/conditions/<axis>/<condition>.yaml` schema (or extend the existing `configs/axes/<N>.yaml` shape).
2. Implement `scripts/validate_conditions.py` that runs at sweep-launch and refuses if any condition omits required fields.
3. Implement the cross-condition uniformity check (non-experimental-axis fields must match across conditions).
4. Wire `sweep_all_axes.py` to invoke the validator before firing any worker.

### 8.2 Repair-parser removal (§3)

**Partially in place.** A-Mem adapter is now retry-3x-then-raise (the historical violation is fixed). Three current violations remain:

1. **`third_party/llm_wrapper/llmplus/client.py:115`** — cache-strip-empties path. RN-003 root-caused; not yet fixed. Either store empties in cache (canonical) or never cache empties consistently (same on read + write).
2. **`src/mem2/orchestrator/runner.py:776-815`** — inference-job exception masking. When a job throws, the runner sets `attempts = []` and continues. Should stop on error and surface to the human.
3. **`scripts/prereq/shared/openie_facts/build_concept_facts_openie.py:364-423`** — partial-artifact write. The script catches per-concept exceptions, appends zero facts, writes the artifact with `num_failures > 0`. The current `openie_facts_v1.json` has `num_failures: 6` and was greenlit. Should retry-3x-then-raise like the per-port adapters.

Recommended scope:
1. Patch `client.py:115` to consistent cache semantics on empties.
2. Refactor `runner.py` inference-job error path to raise (or at minimum record + halt the run, not silently continue).
3. Rewrite the OpenIE builder loop with retry-3x-then-raise; rebuild `openie_facts_v1.json` with `num_failures: 0` or document why the 6 failures are accepted.
4. Add CI lint: search for `except.*:.*continue|pass` patterns in `scripts/prereq/` + `src/mem2/orchestrator/`.

### 8.3 Role-sealing audit (§1)

**Convention only.** Today, nothing structurally prevents a memory_builder from importing a memory_retriever (or vice versa). If a cycle forms, the bug class "builder reads retriever's internal state and uses it to gate consolidate" becomes possible.

Recommended scope:
1. Add CI lint: `import` statements in `src/mem2/branches/memory_builder/<port>.py` may not reference `src/mem2/branches/memory_retriever/`. Inverse also.
2. Add a graph-cycle detector that fails CI if the directed import graph among the three legs forms a cycle.

### 8.4 Validation flow gate enforcement (`ARCHITECTURE.md` §6)

**Convention only.** The validation flow lists 9 gates from unit tests to final adversarial review. Today, a human can update `hub.md` claiming Faithful tier without having run RN-007-style adversarial review.

Recommended scope (light-weight):
1. Add a pre-commit hook on `mem_devlog/.copilot/hub.md` that checks: "if 'Faithful' appears in a diff, the diff must reference an RN-NNN note dated within the last 7 days."
2. The same gate for `parity_grade: faithful` in `configs/axes/<N>.yaml`.

### 8.5 Resume mode (§6.3)

**Not implemented.** The infrastructure (per-call incremental save) is in place; the resume harness is not built. Natural follow-up for the magma anomaly investigation (load the v1 magma run, flip max_tokens to 16k, see if anomaly resolves).

### 8.6 Reproducibility audit script

**Not implemented.** Currently the only way to verify a run's reproducibility is to manually re-run it. A structural answer:

1. Every run's `meta.json` records: code_commit, port, seed, max_tokens, cache flag, max_workers, deterministic seed for any RNG path.
2. A `scripts/verify_run_reproducibility.py <run_id>` script that re-fires the run with the same config + asserts the per-problem `correct/wrong` outcomes match the original.

The script is the structural defense against silent drift (e.g., a prompt-template change that changes outputs).

---

## 9 — Migration / restructure history

| Date | Migration | Notes |
|---|---|---|
| 2026-04-26 | Path-C rebuild (24 paper ports → 9 Faithful + 15 Partial / Surface / Wrong-disclosed) | RN-001, doc 74. The "no-op replacement test" invariant (§5) emerged from this rebuild. |
| 2026-05-12 | Per-port adapter principle (D-2026-05-13) | "Hardcore retrieval needs hardcore memory" — flat 270-concept base is insufficient when paper retrievers expect native structured memory. |
| 2026-05-13 | Repo restructure (3 commits: `d9ba7ad`, `ce8297e`, `78b82c1`) | 140 stale configs archived; outputs historical dirs archived; `shared/` + `ports/` directories established under `data/concept_memory/` and `scripts/prereq/`. |
| 2026-05-13 | 16 per-port adapters built (`scripts/prereq/ports/`) | 4500 LLM calls, 0 failures (per the no-heuristic-fallback rule), $1.82 spend. |
| 2026-05-13 | RN-007 audit + 3 remediation commits | hipporag2 + pathrag downgraded to Surface; graphrag + raptor tests strengthened; oe_topk first-call-empty documented as design. |
| 2026-05-13 | Tier 1 expansion (`benchmarks/`, `analysis/`, `origins/`) | Triad's external-material layout adopted (bidirectional flow). |
| 2026-05-13 | Case-studies infra (4 commits: `1b1b19c`, `0016c68`, `18408e4`, `6f4aa6d`) | 7 case-study modes + ARC grid rendering + HackMD synthesizer + PDF migration deferred. |
| 2026-05-13 | Phase G-lite v1 + baselines mt16k mini-rerun | v1 ran at `max_tokens=2048` (artifact); baselines mt16k confirmed +31.5pp magnitude shift. The reason for §0 + §2 (structural-control discipline). |

Detailed change history lives in `git log --oneline origin/main`; methodological evolution lives in `mem_devlog/.copilot/research_log.md`.

---

## 10 — Naming decisions (revertable)

- `ARCHITECTURE.md` — research claim doc (what the *method* is). Mirrors triad's naming.
- `DESIGN.md` — codebase structure doc (this file). Mirrors triad.
- **"port"** — a paper's method adapted into mem2's contract. Graded Faithful / Partial-with-disclosed-gap / Surface-port-only-disclosed / Wrong-undisclosed (RN-007 rubric).
- **"shared substrate"** vs **"adapted memory"** vs **"flat base"** — three roles, three directories (§4). Naming chosen for write-path clarity.
- **"per-condition sealed config"** — the §2 invariant. "Sealed" emphasizes the structural-rather-than-disciplinary nature.
- **`case_studies/`** vs **`analysis/`** — case_studies is per-run trace data + per-method curated views; analysis is cross-run modules (failure taxonomy, memory growth, telemetry, provenance). They cross-reference but live separately.
- **`origins/`** vs **`literature/`** — origins is broader (papers + Aaron's unpublished work + external repo notes + future candidates); literature is read-only PDF storage.

---

## 11 — Cross-references

- `ARCHITECTURE.md` — the method counterpart (what the structure here is serving).
- `mem_devlog/.copilot/decisions/D-2026-05-13_paper_port_faithful_memory_adaptation.md` — the per-port adaptation design (the principle §4 implements).
- `mem_devlog/.copilot/reviews/RN-007_codex_per_port_adapter_audit_2026-05-13.md` — the audit that grades each port against §1 + §5.
- `mem_devlog/docs/74_phase1_parity_audit_2026_04_26.md` — the audit that established the no-op replacement test (§5).
- Triad's `DESIGN.md` (at `../workstation_00_RRMC/triad/DESIGN.md`) — the sibling project this document borrows core principles from (§0, §1, §3).
- `origins/threads/` — per-research-thread origin notes (the *why* behind decisions; this doc is the *what*).
