# Component Reference

Complete documentation for all pipeline components. For quick parameter
reference, see `options.yaml`. For protocol definitions, see
`src/mem2/core/contracts.py`.

## How Components Work

A YAML config has three sections that work together:

```yaml
pipeline:
  inference_engine: math_ps_solve      # selects the CLASS by name
components:
  inference_engine:                     # provides CONSTRUCTOR ARGS for that class
    model: qwen/qwen-2.5-7b-instruct
    gen_cfg: {n: 1, temperature: 0.3}
```

`wiring.py` looks up the class in the registry, validates the kwargs against the
constructor signature via `inspect.signature`, and instantiates. Unknown params
raise `ConfigurationError` at startup — not at runtime.

### Validation layers

| Layer | What it catches | When |
|-------|----------------|------|
| Registry lookup | Unknown component name | Startup |
| `inspect.signature` | Wrong params for class | Startup |
| `DOMAIN_NAME` cross-check | ARC evaluator with Math IE | Startup |
| `SCHEMA_NAME` / `COMPATIBLE_SCHEMAS` | OE retriever with PS builder | Startup |

### Config inheritance and `null`

Configs inherit via `_base_:` and `deep_merge`. Parent keys bleed into children.
Use YAML `null` to neutralize inherited keys that don't apply:

```yaml
_base_: ../base.yaml                    # base has prompt_options for ARC
pipeline:
  inference_engine: math_ps_solve       # math doesn't accept prompt_options
components:
  inference_engine:
    prompt_options: null                 # neutralize — stripped before validation
```

`null` values are stripped before validation and before passing to the constructor.
A non-null value for a param the class doesn't accept is a `ConfigurationError`.

---

## Task Adapter

**Protocol**: `TaskAdapter` — `get_task_spec()`, `format_problem_sample()`.

Translates between the generic pipeline and domain-specific problem formats.
Returns a `TaskSpec` that tells the runner how to handle problems.

### arc_grid

For ARC-AGI grid transformation tasks.

| Param | Default | Description |
|-------|---------|-------------|
| `task_name` | "arc_grid" | Task identifier. |

---

### math_ps

For competition math problem-solving.

| Param | Default | Description |
|-------|---------|-------------|
| `task_name` | "math_ps" | Task identifier. |

---

### livecodebench

For LiveCodeBench code generation tasks.

| Param | Default | Description |
|-------|---------|-------------|
| `task_name` | "livecodebench" | Task identifier. |

---

## Benchmark

**Protocol**: `BenchmarkAdapter` — `load()`, `validate()`.

Loads problem sets from disk. Each implementation has `DOMAIN_NAME` which is
cross-checked against inference_engine, evaluator, and feedback_engine at startup.

### arc_agi

**Domain**: `arc`

Loads ARC-AGI grid puzzles from the official JSON format.

| Param | Default | Description |
|-------|---------|-------------|
| `data_root` | "data/arc_agi/training" | Path to ARC data directory. |
| `limit` | 5 | Max problems to load. 0 = no limit. |
| `include_ids` | null | List of UIDs to include. null = all. |

---

### competition_math_ps

**Domain**: `math`

Loads competition math problems (Number Theory, Counting & Probability).

| Param | Default | Description |
|-------|---------|-------------|
| `data_root` | "data/competition_math_nt_cp_l5" | Path to math data directory. |
| `split` | "train" | Data split: `train` or `test`. |
| `types` | null | Problem types, e.g. `["Number Theory"]`. null = all. |
| `levels` | null | Difficulty levels, e.g. `[5]`. null = all. |
| `limit` | 0 | Max problems. 0 = no limit. |
| `include_ids` | null | List of UIDs. null = all. |
| `require_integer_answer` | true | Only load problems with integer answers. |

---

### livecodebench

**Domain**: `code`

Loads LiveCodeBench code generation problems with stdin/stdout test cases.

| Param | Default | Description |
|-------|---------|-------------|
| `data_root` | "data/livecodebench_v56" | Path to LCB data directory. |
| `split` | "test" | Data split: `train` or `test`. |
| `difficulties` | null | Filter by difficulty. null = all. |
| `limit` | 0 | Max problems. 0 = no limit. |
| `include_ids` | null | List of UIDs. null = all. |

---

## Memory Builder

**Protocol**: `MemoryBuilder` — `initialize()`, `update()`, `consolidate()`.

Builds and maintains the memory state across the run. Called at startup
(`initialize`), after each problem (`update`), and at end of pass
(`consolidate`). Each builder declares `SCHEMA_NAME` which is checked against
the retriever's `COMPATIBLE_SCHEMAS` at startup.

### none — No-op memory builder

**Schema**: `none`

Returns an empty `MemoryState` and ignores all updates. Use for baseline
configs where no memory is involved.

No parameters.

**Pair with**: `none` retriever.

---

### arcmemo_oe — OE lesson-based memory

**Schema**: `arcmemo_oe`

Stores a flat list of lesson entries, each with `problem_uid`, `hint`,
`is_correct`, and `feedback`. Memory grows during the run as problems are
solved/failed.

**Lifecycle**:
- `initialize`: Loads seed lessons from JSON file
- `update`: Appends new entries from solved/failed problems (generic hints)
- `consolidate`: Trims to `max_entries`, keeping most recent

**Seed file formats** (all auto-detected):

| Format | Structure |
|--------|-----------|
| ArcMemo parsed lessons | `{uid: [{situation: "...", suggestion: "..."}]}` |
| Serialized MemoryState | `{payload: {entries: [{hint: "...", ...}]}}` |
| Plain hints | `{uid: "hint text"}` |

If the seed file contains UIDs matching the eval problem set, only those
entries are loaded (prevents cross-contamination from non-eval problems).

| Param | Default | Description |
|-------|---------|-------------|
| `max_entries` | 200 | Max entries in memory. Oldest trimmed when exceeded. |
| `seed_lessons_file` | null | Path to seed lessons JSON. |
| `seed_lessons_per_problem` | 5 | Max lessons per source problem. |

**Pair with**: `oe_topk` or `oe_selector` retrievers.

---

### arcmemo_ps — PS concept-based memory

**Schema**: `arcmemo_ps`

Stores a `ConceptMemory` object: rich typed concepts with kind, cues,
implementation patterns, parameters, and usage count. Concepts are organized
by category (domain-dependent).

Concepts are **static** — they come from the offline extraction pipeline and
don't change during the run. The `update` method only records solutions for
correctly solved problems (for potential future online extraction).

**Lifecycle**:
- `initialize`: Loads ConceptMemory from seed file
- `update`: Records solutions for correct problems (concepts unchanged)
- `consolidate`: No-op

**Offline extraction pipeline** (produces the seed file):

```
scripts/extract_concepts.py    → extracted.json    (solution → concepts)
scripts/compress_concepts.py   → compressed.json   (deduplicate cues/impl)
```

Always use the compressed output as the seed file. Uncompressed concepts hurt
performance (devlog 17: -9% to -15%).

| Param | Default | Description |
|-------|---------|-------------|
| `seed_memory_file` | null | Path to compressed ConceptMemory JSON. |
| `seed_annotations_file` | null | Alternative: raw annotations (use seed_memory_file instead). |
| `domain` | "arc" | `arc`, `math`, or `code`. Controls concept categories. |
| `max_concepts` | 0 | Max concepts to load. 0 = no limit. |

**Domain categories**:

| Domain | Categories |
|--------|-----------|
| arc | structure, routine |
| math | theorem, technique, definition |
| code | algorithm, pattern, data_structure |

**Pair with**: `ps_selector` retriever.

---

## Memory Retriever

**Protocol**: `MemoryRetriever` — `retrieve()`, `async_retrieve()`.

Retrieves relevant memory for a given problem. Called once per problem per
pass. Returns a `RetrievalBundle`:
- `hint_text`: Rendered text injected into the solver prompt (or null)
- `retrieved_items`: Structured items for analysis/debugging
- `metadata`: Selection mode, scores, errors

Each retriever declares `COMPATIBLE_SCHEMAS` which is checked against the
builder's `SCHEMA_NAME` at startup.

### none — No-op retriever

**Compatible schemas**: `none`, `arcmemo_oe`, `arcmemo_ps`

Always returns `hint_text=None` with no retrieved items. Use for baseline
configs where no memory retrieval is involved.

No parameters.

**Pair with**: any builder.

---

### oe_topk — Recency-based retrieval

**Compatible schemas**: `arcmemo_oe`, `none`

Returns the most recent `top_k` entries from memory. If entries exist for the
current problem UID, scopes to those; otherwise uses the global pool.

No LLM calls. No offline preparation needed.

| Param | Default | Description |
|-------|---------|-------------|
| `top_k` | 2 | Number of entries to retrieve. |

**Pair with**: `arcmemo_oe` builder.

---

### ps_selector — Concept selection retriever

**Compatible schemas**: `arcmemo_ps`

Three selection modes (priority order):

#### A. Precomputed names mode (recommended for experiments)

Set `selected_concepts_file` to `selected_concepts.json` from
`scripts/select_concepts.py`. The retriever loads concept names per problem
and renders them at runtime through the filter → route → render pipeline.
This means `render_mode`, `max_frequency`, `routing_strategy`, etc. are
all active.

#### B. Precomputed rendered mode (legacy baselines)

Set `prompt_info_file` to `prompt_info.json` from `scripts/select_concepts.py`.
The retriever returns pre-rendered hint text directly, **bypassing** the
filter/route/render pipeline. Use for backward-compatible baselines.

#### C. Inline LLM mode (legacy)

When neither precomputed file is set, falls back to calling an LLM at runtime
to select concepts. Slower, non-deterministic. If `use_llm_selector=false`,
returns all concepts (not recommended — causes -9% to -15% degradation).

**Offline selection pipeline** (produces both files):

```
scripts/select_concepts.py \
  --concept-memory data/.../compressed.json \
  --problems data/.../problems.json \
  --domain code \
  --model qwen/qwen3.5-397b-a17b \
  --output-dir data/.../selection_v1 \
  --max-tokens 16384
```

Outputs:
- `selected_concepts.json` — `{uid: ["concept_name", ...]}` (for mode A)
- `prompt_info.json` — `{uid: {hint: "rendered text"}}` (for mode B)
- `completions.json` — raw LLM outputs
- `parse_errors.json` — failed selections

**Filtering and routing** are delegated to format-independent stages
(`ConceptFilter`, `RetrievalRouter` in `src/mem2/retrieval/`) that can be
reused by any retriever regardless of memory format.

| Param | Default | Description |
|-------|---------|-------------|
| `domain` | "arc" | Must match builder's domain. Controls prompt templates. |
| `selected_concepts_file` | "" | Path to precomputed concept names JSON (mode A, preferred). |
| `prompt_info_file` | "" | Path to precomputed rendered hints JSON (mode B, legacy). |
| `use_llm_selector` | true | Enable inline LLM selection (mode C, only when no precomputed files). |
| `selector_model` | "" | Model for inline selection. Inherits inference model if empty. |
| `selector_gen_cfg` | null | Gen config for selector. Default: `{n:1, temperature:0.0, max_tokens:1024}`. Use `max_tokens: 16384`+ for thinking models. |
| `hint_template_key` | "op3" | Hint template variant (legacy inline mode). |
| `top_k` | 10 | Max concepts in inline LLM mode. |
| `render_mode` | "full" | Hint rendering verbosity: `full` (cues + impl + params), `cues_only` (cues, no impl/params), `name_only` (names + descriptions only). |
| `max_frequency` | 0.0 | Drop concepts selected in more than this fraction of problems. 0 = disabled. Requires `concept_frequency_file`. |
| `max_concepts_per_problem` | 0 | Cap the number of selected concepts per problem. 0 = no limit. |
| `routing_strategy` | "none" | Per-problem routing gate: `none` (always include hints), `selection_confidence` (skip if all selected concepts are high-frequency), `hint_length` (skip if hint exceeds `routing_max_hint_chars`). |
| `routing_max_hint_chars` | 0 | Max hint chars for `hint_length` routing strategy. 0 = disabled. |
| `concept_frequency_file` | "" | Path to JSON mapping concept names to selection fractions. Produced by `scripts/compute_concept_frequencies.py`. |

**Prompt templates by domain** (in `src/mem2/concepts/prompts/`):

| Domain | Select template | Hint template |
|--------|----------------|---------------|
| arc | `arc_select.py` | `arc_hints.py` |
| math | `math_select.py` | `math_hints.py` |
| code | `code_select.py` | `code_hints.py` |

**Pair with**: `arcmemo_ps` builder.

---

### oe_selector — OE LLM lesson selection

**Compatible schemas**: `arcmemo_oe`, `none`

Full-featured selector matching original arc_memo behavior. Candidates come
from a lesson bank file. Selection uses an LLM to pick the most relevant
lessons based on a problem description query.

**Selection flow**:
1. Load candidates from `lesson_file` (or fall back to memory entries)
2. Build a problem description query from `description_file` (or synthesize
   from grid metadata: shapes, colors)
3. Send numbered lesson list + query to selector LLM
4. Parse YAML response for lesson indices
5. If LLM fails: fall back to token-overlap ranking

**Retry reselection**: When `include_prev_attempt=true`, the selector prompt
on retry includes the previous attempt's completion. This lets the LLM pick
different lessons based on what went wrong.

| Param | Default | Description |
|-------|---------|-------------|
| `top_k` | 5 | Number of lessons to select. |
| `lesson_file` | null | Path to lesson bank JSON. |
| `description_file` | null | Path to puzzle descriptions JSON. |
| `description_variant_key` | "gpt41_img" | Preferred description variant. |
| `include_prev_attempt` | true | Include prev attempt on retry for reselection. |
| `fallback_to_memory_entries` | true | Use memory entries if no lesson_file. |
| `use_llm_selector` | true | Enable LLM selection. False = token overlap only. |
| `selector_model` | "" | Model for selection. Inherits inference model if empty. |
| `selector_gen_cfg` | null | Default: `{n:1, temperature:0.0, max_tokens:512}`. |
| `max_candidates_for_prompt` | 200 | Max lessons in selector prompt. |

**Pair with**: `arcmemo_oe` builder.

---

## Builder + Retriever Compatibility

| Builder | Retriever | Use Case |
|---------|-----------|----------|
| `none` | `none` | Baseline — no memory involved |
| `arcmemo_oe` | `oe_topk` | Simple recency-based hints |
| `arcmemo_oe` | `oe_selector` | ARC with lesson bank + LLM selection |
| `arcmemo_ps` | `ps_selector` | Any domain with offline concept extraction |
| `arcmemo_oe` | `none` | Build memory but don't use it (debugging) |
| `arcmemo_ps` | `none` | Build memory but don't use it (debugging) |

Mismatched combinations (e.g., `arcmemo_ps` + `oe_topk`) raise
`ConfigurationError` at startup. Builders declare `SCHEMA_NAME`, retrievers
declare `COMPATIBLE_SCHEMAS`.

---

## Trajectory Policy

**Protocol**: `TrajectoryPolicy` — `plan_initial()`, `plan_retry()`.

Controls how many retry paths to explore per problem. Domain-agnostic — no
`DOMAIN_NAME`, works with any benchmark.

### single_path

Linear retry: try once, then retry up to `retry_paths` times on failure.

| Param | Default | Description |
|-------|---------|-------------|
| `retry_paths` | 1 | Number of retry attempts per problem. |

---

## Provider

**Protocol**: `ProviderClient` — `async_generate()`, `async_batch_generate()`,
`get_usage_snapshot()`.

Handles LLM API communication. All provider names map to either
`MockProviderClient` or `LLMPlusProviderClient` — different names select
different API profiles, not different classes.

Providers are wired through `_build_provider` (separate from `_build_component`)
and use profile-based configuration.

### mock

In-memory provider for testing. Returns grid-echo or canned responses.

| Param | Default | Description |
|-------|---------|-------------|
| `profile_name` | "mock" | Must match registry name. |

---

### llmplus_openrouter

Routes to OpenRouter API. Supports all models available on OpenRouter.

| Param | Default | Description |
|-------|---------|-------------|
| `profile_name` | "llmplus_openrouter" | Must match registry name. |
| `dotenv_path` | ".env.example" | Path to env file with `OPENROUTER_API_KEY`. |
| `default_max_concurrency` | 32 | Max concurrent API requests. |

---

### llmplus_openai

Routes to OpenAI API directly.

| Param | Default | Description |
|-------|---------|-------------|
| `profile_name` | "llmplus_openai" | Must match registry name. |
| `dotenv_path` | ".env.example" | Path to env file with `OPENAI_API_KEY`. |
| `default_max_concurrency` | 32 | Max concurrent API requests. |

---

### llmplus_xai

Routes to xAI API (Grok models).

| Param | Default | Description |
|-------|---------|-------------|
| `profile_name` | "llmplus_xai" | Must match registry name. |
| `dotenv_path` | ".env.example" | Path to env file with `XAI_API_KEY`. |
| `default_max_concurrency` | 32 | Max concurrent API requests. |

---

### llmplus_arcmemo_gpt41

Legacy profile for GPT-4.1 via OpenAI. Same class as `llmplus_openai`.

---

### llmplus_openrouter_gemini25_flash_lite

OpenRouter profile targeting Gemini 2.5 Flash Lite specifically. Same class
as `llmplus_openrouter`.

---

## Inference Engine

**Protocol**: `InferenceEngine` — `set_retry_policy()`, `initial_attempt()`,
`retry_attempt()`.

The core solver. Takes a problem + optional hints, calls the LLM, returns
candidate solutions. Each implementation has `DOMAIN_NAME` which is
cross-checked against benchmark, evaluator, and feedback_engine at startup.

### python_transform_retry — ARC solver

**Domain**: `arc`

Generates Python transformation code for ARC grid puzzles. Supports
configurable prompt templates, hint injection, and multi-pass retry with
error feedback.

| Param | Default | Description |
|-------|---------|-------------|
| `model` | "" | Model identifier (e.g., `qwen/qwen3-30b`). |
| `gen_cfg` | null | Generation config: `{n, temperature, max_tokens, top_p, seed, batch_size, ignore_cache, expand_multi}`. |
| `prompt_options` | null | ARC-specific prompt config (see below). |
| `error_feedback` | "all" | Which errors to include on retry: `first` or `all`. |
| `num_feedback_passes` | 1 | How many previous attempts to show. -1 = all. |
| `include_past_outcomes` | true | Include pass/fail status of previous attempts. |
| `include_reselected_lessons` | false | Re-retrieve memory on retry attempts. |

**`prompt_options` sub-keys** (only for this engine):

| Key | Default | Description |
|-----|---------|-------------|
| `include_hint` | false | Inject hint_text from retriever into prompt. |
| `hint_template_key` | "selected" | Hint template variant. |
| `require_hint_citations` | false | Ask model to cite which hints it used. |
| `instruction_key` | "default" | Instruction template variant. |
| `system_prompt_key` | "default" | System prompt variant: `default` or `arcmemo`. |

---

### math_ps_solve — Math solver

**Domain**: `math`

Generates `solve()` functions for competition math problems. Output is
executed and the return value compared to the ground truth integer answer.

| Param | Default | Description |
|-------|---------|-------------|
| `model` | "" | Model identifier. |
| `gen_cfg` | null | Generation config. |
| `error_feedback` | "all" | `first` or `all`. |
| `num_feedback_passes` | 1 | -1 = all. |
| `include_past_outcomes` | true | Include pass/fail of previous attempts. |
| `include_reselected_lessons` | false | Re-retrieve memory on retry. |

Does **not** accept `prompt_options`. Passing it raises `ConfigurationError`.
Use `prompt_options: null` in configs that inherit from base.yaml.

---

### lcb_solve — LiveCodeBench solver

**Domain**: `code`

Generates complete programs for LiveCodeBench problems. Output is executed
as a subprocess with stdin/stdout test cases.

| Param | Default | Description |
|-------|---------|-------------|
| `model` | "" | Model identifier. |
| `gen_cfg` | null | Generation config. |
| `error_feedback` | "all" | `first` or `all`. |
| `num_feedback_passes` | 1 | -1 = all. |
| `include_past_outcomes` | true | Include pass/fail of previous attempts. |
| `include_reselected_lessons` | false | Re-retrieve memory on retry. |

Does **not** accept `prompt_options`. Passing it raises `ConfigurationError`.
Use `prompt_options: null` in configs that inherit from base.yaml.

---

## Feedback Engine

**Protocol**: `FeedbackEngine` — `generate()`.

Compares model output against ground truth and produces feedback for retry
attempts. Each implementation has `DOMAIN_NAME`.

### gt_check — ARC ground truth feedback

**Domain**: `arc`

Compares generated grids against expected grids. Produces per-pair feedback
showing which training pairs passed/failed, with expected vs actual output.

| Param | Default | Description |
|-------|---------|-------------|
| `positive_msg` | "Correct" | Message for correct solutions. |
| `negative_msg` | "Incorrect" | Message for incorrect solutions. |

---

### math_ps_gt — Math ground truth feedback

**Domain**: `math`

Compares `solve()` return value against expected integer answer. Feedback
includes execution errors, wrong answers, or timeout info.

| Param | Default | Description |
|-------|---------|-------------|
| `positive_msg` | "Correct" | Message for correct solutions. |
| `negative_msg` | "Incorrect" | Message for incorrect solutions. |

---

### lcb_gt — LCB ground truth feedback

**Domain**: `code`

Runs generated code against stdin/stdout test cases via subprocess. Feedback
includes which test cases passed/failed, with actual vs expected output.

| Param | Default | Description |
|-------|---------|-------------|
| `positive_msg` | "All test cases passed" | Message for correct solutions. |
| `negative_msg` | "Some test cases failed" | Message for incorrect solutions. |

---

## Evaluator

**Protocol**: `Evaluator` — `evaluate()`, `aggregate()`.

Executes candidate solutions and scores them. Each implementation has
`DOMAIN_NAME`.

### arc_exec — ARC grid evaluator

**Domain**: `arc`

Executes Python code against test grids. Compares output grids to expected.

| Param | Default | Description |
|-------|---------|-------------|
| `require_all_tests` | true | Require all test pairs correct to count as solved. |
| `timeout_s` | 2.0 | Execution timeout per solution in seconds. |

`require_all_tests` is ARC-only. Passing it to math/lcb evaluators raises
`ConfigurationError`. Use `require_all_tests: null` in inherited configs.

---

### math_ps_exec — Math evaluator

**Domain**: `math`

Executes `solve()` function and compares return value to expected integer.

| Param | Default | Description |
|-------|---------|-------------|
| `timeout_s` | 10.0 | Execution timeout per solution in seconds. |

---

### lcb_exec — LCB evaluator

**Domain**: `code`

Runs generated code as subprocess with stdin, compares stdout to expected.

| Param | Default | Description |
|-------|---------|-------------|
| `timeout_s` | 30.0 | Execution timeout per solution in seconds. |

---

## Artifact Sink

**Protocol**: `ArtifactSink` — `write_stage_artifact()`, `write_run_summary()`.

Serializes run outputs to storage. Domain-agnostic.

### json_local

Writes JSON files to the local filesystem under `outputs/_runs/{run_type}/{run_id}/`.

No parameters.

---

## Domain Triples

Benchmark, inference engine, evaluator, and feedback engine must all share the
same domain. Valid combinations:

| Domain | Benchmark | Inference Engine | Evaluator | Feedback Engine |
|--------|-----------|-----------------|-----------|----------------|
| `arc` | `arc_agi` | `python_transform_retry` | `arc_exec` | `gt_check` |
| `math` | `competition_math_ps` | `math_ps_solve` | `math_ps_exec` | `math_ps_gt` |
| `code` | `livecodebench` | `lcb_solve` | `lcb_exec` | `lcb_gt` |

Cross-domain combinations (e.g., `arc_agi` + `math_ps_solve`) raise
`ConfigurationError` at startup via `DOMAIN_NAME` validation.

Task adapter, memory builder/retriever, trajectory policy, provider, and
artifact sink are domain-agnostic — they work with any domain triple.

---

## Adding a New Component

1. Write a class satisfying the Protocol (see `src/mem2/core/contracts.py`)
2. Register it by name in `src/mem2/registry/{component_type}.py`
3. Set `pipeline.{component_type}: your_name` in your config
4. Set `components.{component_type}: {your_params}` in your config

The constructor IS the config schema. `wiring.py` validates params via
`inspect.signature` — no separate schema to maintain. Just define your
`__init__` with explicit keyword arguments (no `**kwargs`).

For domain-specific components, add `DOMAIN_NAME = "your_domain"` as a
class attribute. For memory builders, add `SCHEMA_NAME`. For memory
retrievers, add `COMPATIBLE_SCHEMAS`.

---

## Legacy Aliases

Some components have legacy registry names for backward compatibility:

| Alias | Maps to |
|-------|---------|
| `arc_grid_v1` | `arc_grid` (ArcGridTaskAdapter) |
| `arc_agi_v1` | `arc_agi` (ArcAgiBenchmarkAdapter) |
| `concept_ps`, `arcmemo_ps_v1` | `arcmemo_ps` (ArcMemoPsMemoryBuilder) |
| `lesson_topk`, `lesson_topk_v1` | `oe_topk` (OeTopKRetriever) |
| `lesson_selector`, `arcmemo_selector`, `arcmemo_selector_v1` | `oe_selector` (OeSelectorRetriever) |
| `concept_selector` | `ps_selector` (PsSelectorRetriever) |
| `python_transform_retry_v1` | `python_transform_retry` (PythonTransformRetryIE) |
| `gt_check_v1` | `gt_check` (GroundTruthFeedbackEngine) |
| `arc_exec_v1` | `arc_exec` (ArcExecutionEvaluator) |
| `single_path_v1` | `single_path` (SinglePathTrajectoryPolicy) |
| `json_local_v1` | `json_local` (JsonLocalArtifactSink) |

Use canonical names in new configs.
