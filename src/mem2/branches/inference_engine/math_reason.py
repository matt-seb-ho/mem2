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


MATH_REASON_SYSTEM = (
    "You are an expert competition math solver. "
    "You solve problems by writing clear mathematical reasoning."
)

MATH_REASON_INITIAL_TEMPLATE = """\
### Problem
{problem_text}

### Instructions
Solve the problem above using clear mathematical reasoning.
- Show your work step by step, naming any theorems or techniques you use.
- You may use intermediate calculations, but focus on mathematical reasoning, not code.
- Present your final answer as an integer inside \\boxed{{}}, e.g. \\boxed{{42}}.
"""

MATH_REASON_HINT_TEMPLATE = """\
### Hints
Here are some techniques from previously solved math problems that may be relevant:
{hints}
"""

MATH_REASON_RETRY_TEMPLATE = """\
### Your Previous Response(s) and Outcomes
{history}

### New Instructions
Please reflect on the above issues and revise your approach.
- Consider whether your mathematical reasoning is correct.
- Check for arithmetic or counting errors.
- Try an alternative approach if needed.
- Present your final answer as an integer inside \\boxed{{}}.
"""


class MathReasonInferenceEngine:
    """Inference engine for math problems using mathematical reasoning.

    Generates prompts asking the model to reason mathematically and
    present the answer in \\boxed{{N}} format, instead of writing code.
    """

    name = "math_reason"
    DOMAIN_NAME = "math"

    def __init__(
        self,
        model: str = "",
        gen_cfg: dict | None = None,
        include_reselected_lessons: bool = False,
        include_initial_hints: bool = True,
        error_feedback: str = "all",
        num_feedback_passes: int = 1,
        include_past_outcomes: bool = True,
    ):
        self.model = model
        self.gen_cfg = gen_cfg or {"n": 1, "temperature": 0.2}
        self.include_reselected_lessons = bool(include_reselected_lessons)
        self.include_initial_hints = bool(include_initial_hints)
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
        prompt = MATH_REASON_INITIAL_TEMPLATE.format(problem_text=problem_text)

        if self.include_initial_hints and retrieval and retrieval.hint_text:
            prompt += "\n" + MATH_REASON_HINT_TEMPLATE.format(hints=retrieval.hint_text)

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
                    content = fb.content or ""
                    if content and content != "Correct":
                        block.append(f"**Outcome**: {content}")
            blocks.append("\n".join(block))

        history = "\n\n---\n\n".join(blocks) if blocks else "No previous attempts."

        components = [initial_prompt, ""]
        components.append(MATH_REASON_RETRY_TEMPLATE.format(history=history))

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
