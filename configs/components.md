# Component Reference

Detailed documentation for each pipeline component. For quick parameter
reference, see `options.yaml`.

## Memory Builder

Builds and maintains the memory state across the run. Called at startup
(`initialize`), after each problem (`update`), and at end of pass
(`consolidate`).

### none — No-op memory builder

Returns an empty `MemoryState` and ignores all updates. Use for baseline
configs where no memory is involved.

No parameters.

**Pair with**: `none` retriever.

---

### arcmemo_oe — OE lesson-based memory

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

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `max_entries` | 200 | Max entries in memory. Oldest trimmed when exceeded. |
| `seed_lessons_file` | null | Path to seed lessons JSON. |
| `seed_lessons_per_problem` | 5 | Max lessons per source problem. |

**Pair with**: `oe_topk` or `oe_selector` retrievers.

---

### arcmemo_ps — PS concept-based memory

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

**Parameters**:

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

Retrieves relevant memory for a given problem. Called once per problem per
pass. Returns a `RetrievalBundle`:
- `hint_text`: Rendered text injected into the solver prompt (or null)
- `retrieved_items`: Structured items for analysis/debugging
- `metadata`: Selection mode, scores, errors

### none — No-op retriever

Always returns `hint_text=None` with no retrieved items. Use for baseline
configs where no memory retrieval is involved.

No parameters.

**Pair with**: `none` builder.

---

### oe_topk — Recency-based retrieval

Returns the most recent `top_k` entries from memory. If entries exist for the
current problem UID, scopes to those; otherwise uses the global pool.

No LLM calls. No offline preparation needed.

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `top_k` | 2 | Number of entries to retrieve. |

**Pair with**: `arcmemo_oe` builder.

---

### ps_selector — Concept selection retriever

Two modes of operation:

#### A. Precomputed mode (recommended)

Set `prompt_info_file` to a JSON produced by `scripts/select_concepts.py`.
The retriever returns the pre-rendered hint string directly. No LLM calls at
runtime. Fast, deterministic, debuggable.

**Offline selection pipeline** (produces prompt_info.json):

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
- `prompt_info.json` — `{uid: {hint: "rendered text"}}` (used by retriever)
- `selected_concepts.json` — `{uid: ["concept_name", ...]}` (for analysis)
- `completions.json` — raw LLM outputs
- `parse_errors.json` — failed selections

#### B. Inline LLM mode (legacy)

When `prompt_info_file` is not set, falls back to calling an LLM at runtime
to select concepts. Slower, non-deterministic. If `use_llm_selector=false`,
returns all concepts (not recommended — causes -9% to -15% degradation).

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `domain` | "arc" | Must match builder's domain. Controls prompt templates. |
| `prompt_info_file` | "" | Path to precomputed hints JSON. When set, precomputed mode. |
| `use_llm_selector` | true | Enable inline LLM selection (only when no prompt_info_file). |
| `selector_model` | "" | Model for inline selection. Inherits inference model if empty. |
| `selector_gen_cfg` | null | Gen config for selector. Default: `{n:1, temperature:0.0, max_tokens:1024}`. Use `max_tokens: 16384`+ for thinking models. |
| `hint_template_key` | "op3" | Hint template variant (legacy inline mode). |
| `top_k` | 10 | Max concepts in inline LLM mode. |

**Prompt templates by domain** (in `src/mem2/concepts/prompts/`):

| Domain | Select template | Hint template |
|--------|----------------|---------------|
| arc | `arc_select.py` | `arc_hints.py` |
| math | `math_select.py` | `math_hints.py` |
| code | `code_select.py` | `code_hints.py` |

**Pair with**: `arcmemo_ps` builder.

---

### oe_selector — OE LLM lesson selection

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

**Parameters**:

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

## Valid Builder + Retriever Combinations

| Builder | Retriever | Use Case |
|---------|-----------|----------|
| `none` | `none` | Baseline — no memory involved |
| `arcmemo_oe` | `oe_topk` | Simple recency-based hints |
| `arcmemo_oe` | `oe_selector` | ARC with lesson bank + LLM selection |
| `arcmemo_ps` | `ps_selector` | Any domain with offline concept extraction |

Mismatched combinations (e.g., `arcmemo_ps` + `oe_topk`) will not crash
but will produce empty or nonsensical retrievals because the memory payload
schemas are different.
