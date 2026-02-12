from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from mem2.core.entities import AttemptRecord, MemoryState, ProblemSpec, RetrievalBundle, RunContext


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

_TOPK_CONCEPT_PROMPT = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons and a puzzle description.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Puzzle Description
{description}
"""

_RESELECT_PROMPT = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons and a previous attempt at solving the puzzle.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Previous Attempt
{completion}
"""

_RESELECT_PROMPT_WITH_DESC = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons, a visual description of the puzzle, and a previous attempt at solving the puzzle.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Puzzle Description
{description}

### Previous Attempt
{completion}
"""


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _extract_first_yaml_block(text: str) -> str | None:
    m = _YAML_BLOCK_RE.search(text)
    if not m:
        return None
    return m.group(1)


def _flatten_lesson_bank(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # ArcMemo parsed lessons: {uid: [ {situation, suggestion}, ... ] }
    if isinstance(payload, dict) and "payload" not in payload:
        for uid, lesson_list in payload.items():
            if not isinstance(lesson_list, list):
                continue
            for idx, lesson in enumerate(lesson_list):
                if not isinstance(lesson, dict):
                    continue
                situation = str(lesson.get("situation", "")).strip()
                suggestion = str(lesson.get("suggestion", "")).strip()
                if not situation and not suggestion:
                    continue
                hint_parts = []
                if situation:
                    hint_parts.append(f"- situation: {situation}")
                if suggestion:
                    hint_parts.append(f"  suggestion: {suggestion}")
                rows.append(
                    {
                        "source_uid": str(uid),
                        "lesson_idx": idx,
                        "situation": situation,
                        "suggestion": suggestion,
                        "hint": "\n".join(hint_parts),
                    }
                )
        return rows

    # MemoryState-like format: {"payload": {"entries": [...]}}
    if isinstance(payload, dict):
        entries = payload.get("payload", {}).get("entries", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        hint = str(entry.get("hint", "")).strip()
        if not hint:
            continue
        rows.append(
            {
                "source_uid": str(entry.get("problem_uid", "")),
                "lesson_idx": idx,
                "situation": str(entry.get("situation", "")).strip(),
                "suggestion": str(entry.get("suggestion", "")).strip(),
                "hint": hint,
            }
        )
    return rows


class ArcMemoStyleSelectorRetriever:
    """
    ArcMemo-style selector:
    - Candidate lessons come from a large lesson bank (parsed lessons).
    - Selection is query-based (problem description + optional previous completion),
      returning top-k hints formatted as situation/suggestion bullets.
    """

    name = "arcmemo_selector"

    def __init__(
        self,
        top_k: int = 5,
        lesson_file: str | None = None,
        description_file: str | None = None,
        description_variant_key: str | None = "gpt41_img",
        include_prev_attempt: bool = True,
        fallback_to_memory_entries: bool = True,
        use_llm_selector: bool = True,
        selector_model: str = "",
        selector_gen_cfg: dict[str, Any] | None = None,
        max_candidates_for_prompt: int = 200,
    ):
        self.top_k = int(top_k)
        self.include_prev_attempt = bool(include_prev_attempt)
        self.fallback_to_memory_entries = bool(fallback_to_memory_entries)
        self.use_llm_selector = bool(use_llm_selector)
        self.selector_model = str(selector_model or "")
        self.selector_gen_cfg = dict(selector_gen_cfg or {"n": 1, "temperature": 0.0, "max_tokens": 512})
        self.max_candidates_for_prompt = int(max_candidates_for_prompt)

        self.lesson_file = lesson_file
        self.description_file = description_file
        self.description_variant_key = description_variant_key
        self._lesson_rows: list[dict[str, Any]] = []
        self._descriptions: dict[str, str] = {}

        if lesson_file:
            lesson_path = Path(lesson_file).expanduser()
            if not lesson_path.is_absolute():
                lesson_path = Path.cwd() / lesson_path
            if lesson_path.exists():
                lesson_payload = json.loads(lesson_path.read_text())
                self._lesson_rows = _flatten_lesson_bank(lesson_payload)

        if description_file:
            description_path = Path(description_file).expanduser()
            if not description_path.is_absolute():
                description_path = Path.cwd() / description_path
            if description_path.exists():
                payload = json.loads(description_path.read_text())
                if isinstance(payload, dict):
                    descriptions: dict[str, str] = {}
                    for uid, desc in payload.items():
                        text = self._extract_description_text(desc)
                        if text:
                            descriptions[str(uid)] = text
                    self._descriptions = descriptions

        for row in self._lesson_rows:
            text = f"{row.get('situation', '')}\n{row.get('suggestion', '')}\n{row.get('hint', '')}"
            row["_tokens"] = _tokenize(text)

    def _extract_description_text(self, value: Any) -> str | None:
        if isinstance(value, str):
            txt = value.strip()
            return txt or None
        if not isinstance(value, dict):
            return None

        # nested format: {variant_key: {"description": "..."}}
        if self.description_variant_key and self.description_variant_key in value:
            preferred = value[self.description_variant_key]
            if isinstance(preferred, dict):
                desc = preferred.get("description")
                if isinstance(desc, str) and desc.strip():
                    return desc.strip()

        # fallback: first sub-entry that carries a description field
        for sub in value.values():
            if isinstance(sub, dict):
                desc = sub.get("description")
                if isinstance(desc, str) and desc.strip():
                    return desc.strip()
        return None

    @staticmethod
    def _synthesize_problem_description(problem: ProblemSpec) -> str:
        chunks: list[str] = [f"puzzle_id={problem.uid}"]
        for idx, pair in enumerate(problem.train_pairs[:4], start=1):
            in_grid = pair.get("input", [])
            out_grid = pair.get("output", [])
            in_h = len(in_grid)
            in_w = len(in_grid[0]) if in_grid and isinstance(in_grid[0], list) else 0
            out_h = len(out_grid)
            out_w = len(out_grid[0]) if out_grid and isinstance(out_grid[0], list) else 0
            in_colors = sorted({int(v) for row in in_grid for v in row}) if in_grid else []
            out_colors = sorted({int(v) for row in out_grid for v in row}) if out_grid else []
            chunks.append(
                f"train_{idx}: in_shape={in_h}x{in_w} out_shape={out_h}x{out_w} "
                f"in_colors={in_colors} out_colors={out_colors}"
            )
        return " | ".join(chunks)

    def _build_query_text(
        self,
        problem: ProblemSpec,
    ) -> tuple[str, str]:
        if problem.uid in self._descriptions:
            base = self._descriptions[problem.uid]
            source = "description_file"
        else:
            base = self._synthesize_problem_description(problem)
            source = "synthesized"
        return base, source

    def _candidate_rows(self, memory: MemoryState) -> tuple[list[dict[str, Any]], str]:
        if self._lesson_rows:
            return self._lesson_rows, "lesson_file"
        if not self.fallback_to_memory_entries:
            return [], "none"
        mem_rows = _flatten_lesson_bank(memory.payload.get("entries", []))
        for row in mem_rows:
            text = f"{row.get('situation', '')}\n{row.get('suggestion', '')}\n{row.get('hint', '')}"
            row["_tokens"] = _tokenize(text)
        return mem_rows, "memory_entries"

    @staticmethod
    def _choose_working_rows(
        candidates: list[dict[str, Any]],
        problem_uid: str,
    ) -> tuple[list[dict[str, Any]], str]:
        _ = problem_uid
        return candidates, "full_bank"

    @staticmethod
    def _rank_overlap(q_tokens: set[str], rows: list[dict[str, Any]]) -> list[tuple[float, int, dict[str, Any]]]:
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            l_tokens = row.get("_tokens", set())
            if not isinstance(l_tokens, set):
                l_tokens = set()
            if not q_tokens or not l_tokens:
                score = 0.0
            else:
                overlap = len(q_tokens & l_tokens)
                score = overlap / max(1.0, len(l_tokens) ** 0.5)
            scored.append((float(score), idx, row))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    @staticmethod
    def _build_concept_list(rows: list[dict[str, Any]]) -> tuple[str, dict[int, dict[str, Any]]]:
        mapping: dict[int, dict[str, Any]] = {}
        entries: list[str] = []
        for i, row in enumerate(rows, start=1):
            situation = str(row.get("situation", "")).strip()
            suggestion = str(row.get("suggestion", "")).strip()
            lesson_entry = [f"lesson {i}."]
            if situation:
                lesson_entry.append(f"- situation: {situation}")
            if suggestion:
                lesson_entry.append(f"- suggestion: {suggestion}")
            if not situation and not suggestion:
                hint = str(row.get("hint", "")).strip()
                if hint:
                    lesson_entry.append(f"- hint: {hint}")
            entries.append("\n".join(lesson_entry))
            mapping[i] = row
        return "\n".join(entries), mapping

    def _parse_top_k_selection(
        self, completion_text: str, valid_ids: set[int]
    ) -> tuple[list[int], str | None]:
        if not completion_text.strip():
            return [], "empty_completion"
        yaml_text = _extract_first_yaml_block(completion_text) or completion_text
        try:
            parsed = yaml.safe_load(yaml_text)
        except Exception as exc:
            return [], f"yaml_parse_error: {exc}"

        if isinstance(parsed, list):
            raw = parsed
        elif isinstance(parsed, dict):
            raw = list(parsed.keys())
        else:
            return [], f"unsupported_yaml_type: {type(parsed).__name__}"

        selected: list[int] = []
        for item in raw:
            try:
                idx = int(str(item).strip())
            except Exception:
                continue
            if idx in valid_ids and idx not in selected:
                selected.append(idx)
            if len(selected) >= self.top_k:
                break
        if not selected:
            return [], "no_valid_indices"
        return selected, None

    @staticmethod
    def _rows_to_bundle(
        *,
        problem_uid: str,
        chosen_rows: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> RetrievalBundle:
        hint_lines = [str(row.get("hint", "")).strip() for row in chosen_rows if row.get("hint")]
        hint_text = "\n".join(x for x in hint_lines if x) or None
        retrieved_items = []
        for row in chosen_rows:
            retrieved_items.append(
                {
                    "source_uid": row.get("source_uid"),
                    "lesson_idx": row.get("lesson_idx"),
                    "situation": row.get("situation"),
                    "suggestion": row.get("suggestion"),
                    "hint": row.get("hint"),
                    "score": row.get("_score", 0.0),
                }
            )
        return RetrievalBundle(
            problem_uid=problem_uid,
            hint_text=hint_text,
            retrieved_items=retrieved_items,
            metadata=metadata,
        )

    def _build_selector_prompt(
        self,
        *,
        top_k: int,
        concept_list: str,
        query_text: str,
        query_source: str,
        previous_attempts: list[AttemptRecord],
    ) -> str:
        if self.include_prev_attempt and previous_attempts:
            completion = str(previous_attempts[-1].completion or "").strip()
            if completion:
                if query_source == "description_file":
                    return _RESELECT_PROMPT_WITH_DESC.format(
                        top_k=top_k,
                        concept_list=concept_list,
                        description=query_text,
                        completion=completion,
                    )
                return _RESELECT_PROMPT.format(
                    top_k=top_k,
                    concept_list=concept_list,
                    completion=completion,
                )
        return _TOPK_CONCEPT_PROMPT.format(
            top_k=top_k,
            concept_list=concept_list,
            description=query_text,
        )

    def _fallback_bundle(
        self,
        *,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        candidates: list[dict[str, Any]],
        candidate_source: str,
        parse_error: str | None = None,
        selector_prompt: str | None = None,
        selector_completion: str | None = None,
    ) -> RetrievalBundle:
        query_text, query_source = self._build_query_text(problem)
        q_tokens = _tokenize(query_text)
        working_rows, selector_mode = self._choose_working_rows(candidates, problem.uid)
        scored = self._rank_overlap(q_tokens, working_rows)
        chosen_rows = []
        for score, _, row in scored[: self.top_k]:
            row_copy = dict(row)
            row_copy["_score"] = score
            chosen_rows.append(row_copy)
        metadata = {
            "top_k": self.top_k,
            "selector_mode": f"{selector_mode}_fallback",
            "query_source": query_source,
            "candidate_source": candidate_source,
            "candidate_count": len(working_rows),
            "history_attempts": len(previous_attempts),
            "selector_parsing_error": parse_error,
            "selector_prompt": selector_prompt,
            "selector_completion": selector_completion,
            "selector_selected_uids": [
                [row.get("source_uid"), int(row.get("lesson_idx", 0))] for row in chosen_rows
            ],
        }
        return self._rows_to_bundle(problem_uid=problem.uid, chosen_rows=chosen_rows, metadata=metadata)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        candidates, candidate_source = self._candidate_rows(memory)
        if not candidates or self.top_k <= 0:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={
                    "top_k": self.top_k,
                    "candidate_count": len(candidates),
                    "candidate_source": candidate_source,
                    "history_attempts": len(previous_attempts),
                    "selector_mode": "empty",
                },
            )
        return self._fallback_bundle(
            problem=problem,
            previous_attempts=previous_attempts,
            candidates=candidates,
            candidate_source=candidate_source,
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        candidates, candidate_source = self._candidate_rows(memory)
        if not candidates or self.top_k <= 0:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={
                    "top_k": self.top_k,
                    "candidate_count": len(candidates),
                    "candidate_source": candidate_source,
                    "history_attempts": len(previous_attempts),
                    "selector_mode": "empty",
                },
            )

        if not self.use_llm_selector:
            return self._fallback_bundle(
                problem=problem,
                previous_attempts=previous_attempts,
                candidates=candidates,
                candidate_source=candidate_source,
            )

        query_text, query_source = self._build_query_text(problem)
        working_rows, selector_mode = self._choose_working_rows(candidates, problem.uid)
        prompt_rows = list(working_rows)
        concept_list, concept_map = self._build_concept_list(prompt_rows)
        selector_prompt = self._build_selector_prompt(
            top_k=self.top_k,
            concept_list=concept_list,
            query_text=query_text,
            query_source=query_source,
            previous_attempts=previous_attempts,
        )

        model_name = self.selector_model or selector_model
        if not model_name:
            return self._fallback_bundle(
                problem=problem,
                previous_attempts=previous_attempts,
                candidates=candidates,
                candidate_source=candidate_source,
                parse_error="selector_model_missing",
                selector_prompt=selector_prompt,
            )

        try:
            completions = await provider.async_generate(
                prompt=selector_prompt,
                model=model_name,
                gen_cfg=self.selector_gen_cfg,
            )
            selector_completion = str(completions[0]) if completions else ""
        except Exception as exc:
            return self._fallback_bundle(
                problem=problem,
                previous_attempts=previous_attempts,
                candidates=candidates,
                candidate_source=candidate_source,
                parse_error=f"selector_call_error: {type(exc).__name__}: {exc}",
                selector_prompt=selector_prompt,
            )

        selected_ids, parse_error = self._parse_top_k_selection(
            selector_completion, valid_ids=set(concept_map.keys())
        )
        if not selected_ids:
            return self._fallback_bundle(
                problem=problem,
                previous_attempts=previous_attempts,
                candidates=candidates,
                candidate_source=candidate_source,
                parse_error=parse_error,
                selector_prompt=selector_prompt,
                selector_completion=selector_completion,
            )

        chosen_rows = []
        for idx in selected_ids[: self.top_k]:
            row = dict(concept_map[idx])
            row["_score"] = 1.0
            chosen_rows.append(row)

        metadata = {
            "top_k": self.top_k,
            "selector_mode": f"{selector_mode}_llm",
            "query_source": query_source,
            "candidate_source": candidate_source,
            "candidate_count": len(working_rows),
            "prompt_candidate_count": len(prompt_rows),
            "history_attempts": len(previous_attempts),
            "selector_model": model_name,
            "selector_prompt": selector_prompt,
            "selector_completion": selector_completion,
            "selector_parsing_error": parse_error,
            "selector_selected_uids": [
                [row.get("source_uid"), int(row.get("lesson_idx", 0))] for row in chosen_rows
            ],
        }
        return self._rows_to_bundle(problem_uid=problem.uid, chosen_rows=chosen_rows, metadata=metadata)
