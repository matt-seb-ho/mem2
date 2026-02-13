from __future__ import annotations

from mem2.core.entities import (
    AttemptRecord,
    FeedbackRecord,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
    TrajectoryPlan,
)
from mem2.core.retry_policy import ArcMemoRetryPolicy
from mem2.prompting.render import prompt_fingerprint


MATH_PS_SYSTEM = (
    "You are an expert competition math solver. "
    "You solve problems by writing Python code."
)

MATH_PS_INITIAL_TEMPLATE = """### Problem
{problem_text}

### Instructions
Write a Python function `solve()` that computes and returns the integer answer to the problem above.
- Your function should have signature: `def solve() -> int:`
- You may use the standard library (math, itertools, functools, collections, fractions).
- Write your final code in a markdown python code block (```python ... ```).
- Return the answer as an integer from the `solve()` function.
"""

MATH_PS_HINT_TEMPLATE = """### Hints
Here are some lessons from previously solved math problems that may be relevant:
{hints}
"""

MATH_PS_RETRY_TEMPLATE = """### Your Previous Response(s) and Outcomes
{history}

### New Instructions
Please reflect on the above issues (code errors, wrong answer, or timeout) and revise your approach.
- Consider whether your mathematical reasoning is correct.
- Consider whether your code correctly implements your reasoning.
- Return a corrected `solve()` function in a markdown python code block.
"""


class MathPsSolveInferenceEngine:
    """Inference engine for math-PS problems.

    Generates prompts asking the model to write a ``solve()`` function
    that returns an integer answer.
    """

    name = "math_ps_solve"

    def __init__(
        self,
        model: str = "",
        gen_cfg: dict | None = None,
        include_reselected_lessons: bool = False,
        error_feedback: str = "all",
        num_feedback_passes: int = 1,
        include_past_outcomes: bool = True,
        **kwargs,  # absorb extra keys from config inheritance (e.g. prompt_options)
    ):
        self.model = model
        self.gen_cfg = gen_cfg or {"n": 1, "temperature": 0.2}
        self.include_reselected_lessons = bool(include_reselected_lessons)
        self.error_feedback = str(error_feedback)
        self.num_feedback_passes = int(num_feedback_passes)
        self.include_past_outcomes = bool(include_past_outcomes)

    def set_retry_policy(self, policy: ArcMemoRetryPolicy) -> None:
        self.error_feedback = policy.error_feedback
        self.num_feedback_passes = policy.num_feedback_passes
        self.include_past_outcomes = policy.include_past_outcomes

    def _make_initial_prompt(
        self,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
    ) -> str:
        problem_text = problem.metadata.get("problem_text", "")
        prompt = MATH_PS_INITIAL_TEMPLATE.format(problem_text=problem_text)

        if retrieval and retrieval.hint_text:
            prompt += "\n" + MATH_PS_HINT_TEMPLATE.format(hints=retrieval.hint_text)

        return prompt

    def _make_retry_prompt(
        self,
        initial_prompt: str,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
        new_concepts: str | None = None,
    ) -> str:
        if self.num_feedback_passes == -1:
            attempt_slice = attempts
            offset = 0
        else:
            attempt_slice = attempts[-self.num_feedback_passes:]
            offset = max(0, len(attempts) - len(attempt_slice))

        blocks: list[str] = []
        for local_idx, att in enumerate(attempt_slice, start=offset + 1):
            idx0 = local_idx - 1
            block = [f"#### Attempt {local_idx}", att.completion or ""]
            if idx0 < len(feedback):
                include_outcome = self.include_past_outcomes or (idx0 == len(attempts) - 1)
                if include_outcome:
                    fb = feedback[idx0]
                    md = fb.metadata or {}
                    errors = list(md.get("errors", []))
                    mismatches = list(md.get("mismatches", []))
                    if self.error_feedback == "first":
                        errors = errors[:1]
                        mismatches = mismatches[:1]
                    if errors:
                        block.append("**Execution / Parsing Errors**")
                        block.extend(f"- {e}" for e in errors)
                    if mismatches:
                        block.append("**Wrong Answer**")
                        for m in mismatches:
                            block.append(
                                f"- Your code returned {m.get('output')}, "
                                f"expected {m.get('expected')}"
                            )
                    if not errors and not mismatches and fb.content:
                        block.append(fb.content)
            blocks.append("\n".join(block))

        history = "\n\n---\n\n".join(blocks) if blocks else "No previous attempts."

        components = [initial_prompt, ""]
        components.append(MATH_PS_RETRY_TEMPLATE.format(history=history))

        if new_concepts:
            components.append(
                "\n### Reselected Lessons\n"
                "Here are reselected lessons that may help:\n"
                f"{new_concepts}"
            )
        return "\n".join(components)

    async def initial_attempt(
        self,
        ctx: RunContext,
        provider,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        trajectory_plan: TrajectoryPlan,
        preset_completions: list[str] | None = None,
    ) -> list[AttemptRecord]:
        prompt = self._make_initial_prompt(problem, retrieval)
        cfg = dict(self.gen_cfg)
        cfg["n"] = trajectory_plan.num_paths
        if preset_completions is None:
            completions = await provider.async_generate(
                prompt=prompt, model=self.model, gen_cfg=cfg
            )
        else:
            completions = [str(x) for x in preset_completions][:trajectory_plan.num_paths]

        return [
            AttemptRecord(
                problem_uid=problem.uid,
                pass_idx=0,
                branch_id=self.name,
                completion=txt,
                prompt=prompt,
                metadata={
                    "strategy": trajectory_plan.strategy,
                    "path_idx": i,
                    "initial_prompt": prompt,
                    "initial_prompt_fingerprint": prompt_fingerprint(prompt),
                },
            )
            for i, txt in enumerate(completions)
        ]

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
        initial_prompt = ""
        if attempt_history:
            initial_prompt = str(
                attempt_history[0].metadata.get("initial_prompt", attempt_history[0].prompt)
            )
        else:
            initial_prompt = self._make_initial_prompt(problem, retrieval)

        prompt = self._make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=attempt_history,
            feedback=feedback_history,
            new_concepts=(
                retrieval.hint_text
                if self.include_reselected_lessons and retrieval and retrieval.hint_text
                else None
            ),
        )
        cfg = dict(self.gen_cfg)
        cfg["n"] = trajectory_plan.num_paths
        completions = await provider.async_generate(
            prompt=prompt, model=self.model, gen_cfg=cfg
        )
        return [
            AttemptRecord(
                problem_uid=problem.uid,
                pass_idx=0,
                branch_id=self.name,
                completion=txt,
                prompt=prompt,
                metadata={
                    "strategy": trajectory_plan.strategy,
                    "path_idx": i,
                    "retry": True,
                    "history_len": len(attempt_history),
                    "initial_prompt": initial_prompt,
                    "retry_prompt_fingerprint": prompt_fingerprint(prompt),
                },
            )
            for i, txt in enumerate(completions)
        ]
