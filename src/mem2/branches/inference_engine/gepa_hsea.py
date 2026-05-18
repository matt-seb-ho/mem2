from __future__ import annotations

import ast
import asyncio
import logging
from typing import Any

import numpy as np

from mem2.core.entities import (
    AttemptRecord,
    FeedbackRecord,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
    TrajectoryPlan,
)
from mem2.core.retry_policy import ArcMemoRetryPolicy
from mem2.prompting.render import format_grid_numpy, prompt_fingerprint
from mem2.utils.code_execution import execute_transform, extract_python_block

logger = logging.getLogger(__name__)

# HSEA runs several sequential LLM stages per puzzle. At sweep scale the provider
# semaphore can queue tail-stage calls behind earlier puzzles, so this guard must
# cover queue wait plus the provider's own per-request timeout/retry budget.
_LLM_CALL_TIMEOUT_S = 1800.0
_DEFAULT_HINT_CHAR_LIMIT = 1600
_TRACE_GRID_CHAR_LIMIT = 500


async def _async_generate_bounded(provider, prompt: str, model: str, gen_cfg: dict) -> list[str]:
    try:
        return await asyncio.wait_for(
            provider.async_generate(prompt=prompt, model=model, gen_cfg=gen_cfg),
            timeout=_LLM_CALL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        n = int(gen_cfg.get("n", 1) or 1)
        logger.warning(
            "LLM call exceeded %.0fs wall-clock - returning %d empty completions.",
            _LLM_CALL_TIMEOUT_S,
            n,
        )
        return [""] * n


def _truncate_text(text: str, limit: int) -> str:
    compact = text.strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _render_train_pairs(problem: ProblemSpec) -> str:
    blocks: list[str] = []
    for idx, pair in enumerate(problem.train_pairs, start=1):
        blocks.extend(
            [
                f"Pair {idx}",
                "Input:",
                format_grid_numpy(pair["input"]),
                "Output:",
                format_grid_numpy(pair["output"]),
                "",
            ]
        )
    return "\n".join(blocks).strip()


def make_hypothesize_prompt(
    problem: ProblemSpec,
    hint_text: str | None = None,
    *,
    hint_char_limit: int = _DEFAULT_HINT_CHAR_LIMIT,
) -> str:
    blocks = [
        "You are solving an ARC-AGI puzzle. Below are training input/output pairs.",
        "",
        _render_train_pairs(problem),
    ]
    if hint_text:
        blocks.extend(
            [
                "",
                "Optional concept hint:",
                _truncate_text(hint_text, hint_char_limit),
            ]
        )
    blocks.extend(
        [
            "",
            "First, reason step-by-step about the transformation pattern in one short paragraph.",
            'Then state the rule on a final line prefixed exactly with "Rule:".',
        ]
    )
    return "\n".join(blocks)


def make_synthesize_prompt(problem: ProblemSpec, rule: str) -> str:
    return "\n".join(
        [
            "You are implementing an ARC-AGI transformation in Python.",
            "",
            "Training pairs:",
            _render_train_pairs(problem),
            "",
            "Rule (natural language):",
            rule.strip(),
            "",
            "Write a Python function `transform_matrix(grid)` where `grid` is `List[List[int]]`.",
            "Return only one ```python fenced code block.",
        ]
    )


def make_debug_prompt(problem: ProblemSpec, current_code: str, execution_trace: str) -> str:
    return "\n".join(
        [
            "The previous attempt failed on some training pairs.",
            "",
            "Training pairs:",
            _render_train_pairs(problem),
            "",
            "Previous code:",
            "```python",
            current_code.strip(),
            "```",
            "",
            "Execution trace per training pair:",
            execution_trace.strip(),
            "",
            "Diagnose the failure mode in 1-2 sentences. Then output an improved",
            "`transform_matrix(grid)` function in one ```python fenced block.",
        ]
    )


def _parse_rule(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("Rule:"):
            return line.removeprefix("Rule:").strip()
    return _truncate_text(text, 1200)


def _defined_functions(code: str) -> set[str]:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _extract_code_body(completion: str) -> str:
    code, parsing_error = extract_python_block(completion)
    if parsing_error:
        return completion.strip()
    return (code or "").strip()


def _normalize_code_body(code: str) -> str:
    code = code.strip()
    if not code:
        return code
    if "List[" in code and "from typing import" not in code and "import typing" not in code:
        code = "from typing import List\n\n" + code

    functions = _defined_functions(code)
    if "transform" in functions or "transform_matrix" not in functions:
        return code

    wrapper = """

def transform(input_grid):
    grid = input_grid.tolist() if hasattr(input_grid, "tolist") else input_grid
    return np.array(transform_matrix(grid), dtype=int)
""".rstrip()
    return code.rstrip() + wrapper


def _fenced_code(code: str) -> str:
    return "```python\n" + code.strip() + "\n```"


def _code_from_completion(completion: str) -> tuple[str, str]:
    body = _normalize_code_body(_extract_code_body(completion))
    return body, _fenced_code(body)


def _format_grid_short(value: Any) -> str:
    if value is None:
        return "None"
    try:
        rendered = format_grid_numpy(value)
    except Exception:
        rendered = str(value)
    return _truncate_text(rendered, _TRACE_GRID_CHAR_LIMIT)


def _execute_train_pairs(
    problem: ProblemSpec,
    code_body: str,
    *,
    timeout_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not code_body.strip():
        return [
            {
                "pair_idx": idx,
                "passed": False,
                "error": "empty code",
                "output": None,
                "expected": pair.get("output"),
            }
            for idx, pair in enumerate(problem.train_pairs, start=1)
        ]

    for idx, pair in enumerate(problem.train_pairs, start=1):
        result = execute_transform(code_body, pair["input"], timeout_s=timeout_s)
        if result.get("status") != "ok":
            rows.append(
                {
                    "pair_idx": idx,
                    "passed": False,
                    "error": result.get("error") or "execution failed",
                    "output": None,
                    "expected": pair.get("output"),
                }
            )
            continue

        output = result.get("output")
        expected = pair.get("output")
        try:
            passed = bool(
                np.array_equal(np.array(output, dtype=int), np.array(expected, dtype=int))
            )
        except Exception:
            passed = False
        rows.append(
            {
                "pair_idx": idx,
                "passed": passed,
                "error": None if passed else "output mismatch",
                "output": output,
                "expected": expected,
            }
        )
    return rows


def _all_train_passed(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(bool(row.get("passed")) for row in rows)


def _render_execution_trace(rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in rows:
        status = "PASS" if row.get("passed") else "FAIL"
        blocks.append(f"Pair {row.get('pair_idx')}: {status}")
        if row.get("passed"):
            continue
        if row.get("error"):
            blocks.append(f"  error: {_truncate_text(str(row['error']), 500)}")
        blocks.append(f"  expected: {_format_grid_short(row.get('expected'))}")
        blocks.append(f"  observed: {_format_grid_short(row.get('output'))}")
    return "\n".join(blocks)


class GepaHseaInferenceEngine:
    name = "gepa_hsea"
    DOMAIN_NAME = "arc"

    def __init__(
        self,
        model: str = "",
        gen_cfg: dict | None = None,
        prompt_options: dict | None = None,
        error_feedback: str = "all",
        num_feedback_passes: int = 1,
        include_past_outcomes: bool = True,
        include_reselected_lessons: bool = False,
        max_retries: int = 3,
        max_hypothesize_attempts: int = 1,
        execution_timeout_s: float = 2.0,
        hint_char_limit: int = _DEFAULT_HINT_CHAR_LIMIT,
        hypothesize_max_tokens: int = 1024,
        synthesize_max_tokens: int = 4096,
        debug_max_tokens: int = 4096,
    ):
        self.model = model
        self.gen_cfg = gen_cfg or {"n": 1, "temperature": 0.2}
        self.prompt_options = prompt_options or {}
        self.error_feedback = str(error_feedback)
        self.num_feedback_passes = int(num_feedback_passes)
        self.include_past_outcomes = bool(include_past_outcomes)
        self.include_reselected_lessons = bool(include_reselected_lessons)
        self.max_retries = max(0, int(max_retries))
        self.max_hypothesize_attempts = max(1, int(max_hypothesize_attempts))
        self.execution_timeout_s = float(execution_timeout_s)
        self.hint_char_limit = int(hint_char_limit)
        self.stage_max_tokens = {
            "hypothesize": max(1, int(hypothesize_max_tokens)),
            "synthesize": max(1, int(synthesize_max_tokens)),
            "debug": max(1, int(debug_max_tokens)),
        }

    def set_retry_policy(self, policy: ArcMemoRetryPolicy) -> None:
        self.error_feedback = policy.error_feedback
        self.num_feedback_passes = policy.num_feedback_passes
        self.include_past_outcomes = policy.include_past_outcomes

    async def _generate_one(self, provider, prompt: str, *, stage: str) -> str:
        cfg = dict(self.gen_cfg)
        cfg["n"] = 1
        cap = self.stage_max_tokens.get(stage)
        if cap is not None:
            requested = int(cfg.get("max_tokens", cap) or cap)
            cfg["max_tokens"] = min(requested, cap)
        completions = await _async_generate_bounded(provider, prompt, self.model, cfg)
        return str(completions[0] if completions else "")

    async def _hypothesize(self, provider, problem: ProblemSpec, hint_text: str | None) -> tuple[str, str]:
        prompt = make_hypothesize_prompt(
            problem,
            hint_text,
            hint_char_limit=self.hint_char_limit,
        )
        last_text = ""
        for _ in range(self.max_hypothesize_attempts):
            last_text = await self._generate_one(provider, prompt, stage="hypothesize")
            rule = _parse_rule(last_text)
            if rule:
                return rule, prompt
        return _parse_rule(last_text), prompt

    async def _synthesize(self, provider, problem: ProblemSpec, rule: str) -> tuple[str, str, str]:
        prompt = make_synthesize_prompt(problem, rule)
        completion = await self._generate_one(provider, prompt, stage="synthesize")
        code_body, fenced = _code_from_completion(completion)
        return code_body, fenced, prompt

    async def _debug_once(
        self,
        provider,
        problem: ProblemSpec,
        current_code: str,
        trace_text: str,
    ) -> tuple[str, str, str]:
        prompt = make_debug_prompt(problem, current_code, trace_text)
        completion = await self._generate_one(provider, prompt, stage="debug")
        code_body, fenced = _code_from_completion(completion)
        return code_body, fenced, prompt

    async def _run_hsea(
        self,
        *,
        provider,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        trajectory_plan: TrajectoryPlan,
        previous_code: str | None = None,
        pass_kind: str,
    ) -> AttemptRecord:
        stage_order: list[str] = []
        prompt_fingerprints: dict[str, str] = {}
        rule = ""
        final_prompt = ""
        debug_iterations = 0

        if previous_code is None:
            stage_order.append("hypothesize")
            rule, hypothesize_prompt = await self._hypothesize(
                provider,
                problem,
                retrieval.hint_text if retrieval else None,
            )
            prompt_fingerprints["hypothesize"] = prompt_fingerprint(hypothesize_prompt)

            stage_order.append("synthesize")
            code_body, final_completion, synthesize_prompt = await self._synthesize(
                provider,
                problem,
                rule,
            )
            prompt_fingerprints["synthesize"] = prompt_fingerprint(synthesize_prompt)
            final_prompt = synthesize_prompt
        else:
            code_body = _normalize_code_body(previous_code)
            final_completion = _fenced_code(code_body)

        stage_order.append("execute")
        rows = _execute_train_pairs(problem, code_body, timeout_s=self.execution_timeout_s)
        trace_text = _render_execution_trace(rows)
        gate_hit = _all_train_passed(rows)

        while not gate_hit and debug_iterations < self.max_retries:
            stage_order.append("debug")
            code_body, final_completion, debug_prompt = await self._debug_once(
                provider,
                problem,
                code_body,
                trace_text,
            )
            debug_iterations += 1
            prompt_fingerprints[f"debug_{debug_iterations}"] = prompt_fingerprint(debug_prompt)
            final_prompt = debug_prompt

            stage_order.append("execute")
            rows = _execute_train_pairs(problem, code_body, timeout_s=self.execution_timeout_s)
            trace_text = _render_execution_trace(rows)
            gate_hit = _all_train_passed(rows)

        stage_order.append("answer")
        return AttemptRecord(
            problem_uid=problem.uid,
            pass_idx=0,
            branch_id=self.name,
            completion=final_completion,
            prompt=final_prompt,
            metadata={
                "strategy": trajectory_plan.strategy,
                "path_idx": 0,
                "requested_num_paths": trajectory_plan.num_paths,
                "pass_kind": pass_kind,
                "gepa_hsea_stage_order": stage_order,
                "gepa_hsea_rule": rule,
                "gepa_hsea_prompt_fingerprints": prompt_fingerprints,
                "gepa_hsea_all_train_pass_gate_hit": gate_hit,
                "gepa_hsea_debug_iterations": debug_iterations,
                "gepa_hsea_max_retries": self.max_retries,
                "gepa_hsea_execution_trace": trace_text,
                "gepa_hsea_train_passes": [
                    bool(row.get("passed")) for row in rows
                ],
                "gepa_hsea_hint_present": bool(retrieval and retrieval.hint_text),
            },
        )

    async def initial_attempt(
        self,
        ctx: RunContext,
        provider,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        trajectory_plan: TrajectoryPlan,
        preset_completions: list[str] | None = None,
    ) -> list[AttemptRecord]:
        if preset_completions:
            code_body, fenced = _code_from_completion(str(preset_completions[0]))
            rows = _execute_train_pairs(problem, code_body, timeout_s=self.execution_timeout_s)
            return [
                AttemptRecord(
                    problem_uid=problem.uid,
                    pass_idx=0,
                    branch_id=self.name,
                    completion=fenced,
                    prompt="",
                    metadata={
                        "strategy": trajectory_plan.strategy,
                        "path_idx": 0,
                        "preset_completion": True,
                        "gepa_hsea_stage_order": ["execute", "answer"],
                        "gepa_hsea_all_train_pass_gate_hit": _all_train_passed(rows),
                        "gepa_hsea_debug_iterations": 0,
                        "gepa_hsea_execution_trace": _render_execution_trace(rows),
                    },
                )
            ]
        attempt = await self._run_hsea(
            provider=provider,
            problem=problem,
            retrieval=retrieval,
            trajectory_plan=trajectory_plan,
            previous_code=None,
            pass_kind="initial",
        )
        return [attempt]

    async def retry_attempt(
        self,
        ctx: RunContext,
        provider,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        attempt_history: list[AttemptRecord],
        feedback_history: list[FeedbackRecord],
        trajectory_plan: TrajectoryPlan,
    ) -> list[AttemptRecord]:
        previous_code = None
        if attempt_history:
            previous_code = _extract_code_body(attempt_history[-1].completion)
        attempt = await self._run_hsea(
            provider=provider,
            problem=problem,
            retrieval=retrieval,
            trajectory_plan=trajectory_plan,
            previous_code=previous_code,
            pass_kind="outer_retry" if previous_code else "initial",
        )
        return [attempt]
