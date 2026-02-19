"""Inference engine for LiveCodeBench problems.

Generates prompts asking the model to write stdin/stdout Python code.
"""

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

LCB_SYSTEM = (
    "You are an expert competitive programmer. "
    "You solve problems by writing Python code that reads from stdin and writes to stdout."
)

LCB_INITIAL_TEMPLATE = """### Problem
{question_content}

{test_cases_section}

### Instructions
Write a complete Python program that solves the problem above.
- Your program should read input from stdin using `input()` and print the answer to stdout.
- Write your final code in a markdown python code block (```python ... ```).
- Make sure your code handles all edge cases.
- Do not include any example usage or test code outside the main solution.
{starter_code_section}"""

LCB_HINT_TEMPLATE = """### Hints
Here are some lessons from previously solved coding problems that may be relevant:
{hints}
"""

LCB_RETRY_TEMPLATE = """### Your Previous Response(s) and Outcomes
{history}

### New Instructions
Please reflect on the above issues (code errors, wrong output, or timeout) and revise your approach.
- Consider whether your algorithm is correct.
- Consider edge cases and input parsing.
- Return a corrected solution in a markdown python code block.
"""


def _format_test_cases(problem: ProblemSpec) -> str:
    """Format public test cases for the prompt."""
    public_tests = problem.metadata.get("public_test_cases", [])
    if not public_tests:
        return ""

    lines = ["### Example Test Cases"]
    for i, tc in enumerate(public_tests, start=1):
        inp = tc.get("input", "").strip()
        out = tc.get("expected_output", "").strip()
        lines.append(f"**Example {i}:**")
        lines.append(f"Input:\n```\n{inp}\n```")
        lines.append(f"Expected Output:\n```\n{out}\n```")
        lines.append("")
    return "\n".join(lines)


class LcbSolveInferenceEngine:
    """Inference engine for LiveCodeBench problems."""

    name = "lcb_solve"

    def __init__(
        self,
        model: str = "",
        gen_cfg: dict | None = None,
        include_reselected_lessons: bool = False,
        error_feedback: str = "all",
        num_feedback_passes: int = 1,
        include_past_outcomes: bool = True,
        **kwargs,
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
        question_content = problem.metadata.get("question_content", "")
        test_cases_section = _format_test_cases(problem)

        starter_code = problem.metadata.get("starter_code", "").strip()
        starter_code_section = ""
        if starter_code:
            starter_code_section = f"- Use the following starter code:\n```python\n{starter_code}\n```"

        prompt = LCB_INITIAL_TEMPLATE.format(
            question_content=question_content,
            test_cases_section=test_cases_section,
            starter_code_section=starter_code_section,
        )

        if retrieval and retrieval.hint_text:
            prompt += "\n" + LCB_HINT_TEMPLATE.format(hints=retrieval.hint_text)

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
                    test_failures = list(md.get("test_failures", []))
                    if errors:
                        block.append("**Execution / Parsing Errors**")
                        block.extend(f"- {e}" for e in errors)
                    if test_failures:
                        block.append("**Failed Test Cases**")
                        for tf in test_failures:
                            block.append(
                                f"- Test {tf.get('test_idx', '?')}: "
                                f"expected {tf.get('expected', '?')!r}, "
                                f"got {tf.get('actual', '?')!r}"
                            )
                    if not errors and not test_failures and fb.content:
                        block.append(fb.content)
            blocks.append("\n".join(block))

        history = "\n\n---\n\n".join(blocks) if blocks else "No previous attempts."

        components = [initial_prompt, ""]
        components.append(LCB_RETRY_TEMPLATE.format(history=history))

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
