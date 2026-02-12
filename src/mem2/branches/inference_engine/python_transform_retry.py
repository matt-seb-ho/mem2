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
from mem2.prompting.options import ArcMemoPromptOptions
from mem2.prompting.render import make_initial_prompt, make_retry_prompt, prompt_fingerprint


class PythonTransformRetryInferenceEngine:
    name = "python_transform_retry"

    def __init__(
        self,
        model: str = "",
        gen_cfg: dict | None = None,
        prompt_options: dict | None = None,
        error_feedback: str = "all",
        num_feedback_passes: int = 1,
        include_past_outcomes: bool = True,
        include_reselected_lessons: bool = False,
    ):
        self.model = model
        self.gen_cfg = gen_cfg or {"n": 1, "temperature": 0.2}
        self.prompt_options = ArcMemoPromptOptions(**(prompt_options or {}))
        self.error_feedback = str(error_feedback)
        self.num_feedback_passes = int(num_feedback_passes)
        self.include_past_outcomes = bool(include_past_outcomes)
        self.include_reselected_lessons = bool(include_reselected_lessons)

    def set_retry_policy(self, policy: ArcMemoRetryPolicy) -> None:
        self.error_feedback = policy.error_feedback
        self.num_feedback_passes = policy.num_feedback_passes
        self.include_past_outcomes = policy.include_past_outcomes

    async def initial_attempt(
        self,
        ctx: RunContext,
        provider,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        trajectory_plan: TrajectoryPlan,
        preset_completions: list[str] | None = None,
    ) -> list[AttemptRecord]:
        prompt = make_initial_prompt(problem, retrieval, options=self.prompt_options)
        cfg = dict(self.gen_cfg)
        cfg["n"] = trajectory_plan.num_paths
        if preset_completions is None:
            completions = await provider.async_generate(prompt=prompt, model=self.model, gen_cfg=cfg)
        else:
            completions = [str(x) for x in preset_completions][: trajectory_plan.num_paths]
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
            initial_prompt = make_initial_prompt(problem, retrieval, options=self.prompt_options)

        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=attempt_history,
            feedback=feedback_history,
            error_feedback=self.error_feedback,
            num_feedback_passes=self.num_feedback_passes,
            include_past_outcomes=self.include_past_outcomes,
            new_concepts=(
                retrieval.hint_text
                if self.include_reselected_lessons and retrieval and retrieval.hint_text
                else None
            ),
        )
        cfg = dict(self.gen_cfg)
        cfg["n"] = trajectory_plan.num_paths
        completions = await provider.async_generate(prompt=prompt, model=self.model, gen_cfg=cfg)
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
