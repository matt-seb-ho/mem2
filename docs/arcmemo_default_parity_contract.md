# ArcMemo Default Parity Contract (ARC)

This document freezes the inference-critical defaults that `mem2` must match for ArcMemo migration fidelity in ARC default mode.

## Source References

- `/root/workspace/arc_memo/configs/default.yaml`
- `/root/workspace/arc_memo/configs/generation/gen_default.yaml`
- `/root/workspace/arc_memo/configs/model/gpt41.yaml`
- `/root/workspace/arc_memo/concept_mem/evaluation/prompts.py`
- `/root/workspace/arc_memo/concept_mem/evaluation/retry_policy.py`
- `/root/workspace/arc_memo/concept_mem/evaluation/score_tree.py`

## Inference-critical defaults

- execution mode:
  - `run.execution_mode: arc_batch` (ArcMemo iteration-batched scheduling behavior)
- provider profile: `llmplus_arcmemo_gpt41`
- model: `gpt-4.1-2025-04-14`
- generation:
  - `n: 1`
  - `temperature: 0.3`
  - `max_tokens: 1024`
  - `top_p: 1.0`
  - `batch_size: 16`
  - `seed: 88`
  - `ignore_cache: false`
  - `expand_multi: null`
- prompt defaults:
  - `instruction_key: default`
  - `system_prompt_key: default`
  - `hint_template_key: selected`
  - `include_hint: false`
  - `require_hint_citations: false`
- retry policy:
  - `max_passes: 3`
  - `criterion: train`
  - `error_feedback: all`
  - `num_feedback_passes: 1`
  - `include_past_outcomes: true`
- retry prompt:
  - `include_reselected_lessons: false` (ArcMemo default `reselect_concepts=false`)
- scoring:
  - official score semantics: test-pair solved by any attempt, puzzle score is mean over pairs, report sum over puzzles.
  - strict score semantics (migration default): include train + test, select last step, aggregate per puzzle with `any`.

## Compatibility modes

- **strict compatibility** (`run.strict_arcmemo_compat=true`)
  - pins ArcMemo-default provider + model (`gpt-4.1-2025-04-14`) and all inference-critical settings.
- **logic compatibility** (`run.strict_arcmemo_logic_compat=true`)
  - enforces ArcMemo mechanics (prompt/retry/eval/orchestration-critical settings) while allowing provider/model overrides.

## Enforcements in mem2

- strict guard path:
  - `run.strict_arcmemo_compat: true`
  - checker module: `src/mem2/core/arcmemo_compat.py`
- strict experiment preset:
  - `configs/experiments/arcmemo_arc_strict.yaml`
- parity harness scripts:
  - `scripts/parity/run_arc_default_parity_lock.py`
  - `scripts/parity/freeze_parity_baseline.py`
