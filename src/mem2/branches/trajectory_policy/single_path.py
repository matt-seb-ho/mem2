from __future__ import annotations

from mem2.core.entities import AttemptRecord, FeedbackRecord, ProblemSpec, RunContext, TrajectoryPlan


class SinglePathTrajectoryPolicy:
    name = "single_path"

    def __init__(self, retry_paths: int = 1):
        self.retry_paths = int(retry_paths)

    def plan_initial(self, ctx: RunContext, problem: ProblemSpec) -> TrajectoryPlan:
        return TrajectoryPlan(num_paths=1, strategy="single_path", metadata={"phase": "initial"})

    def plan_retry(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
    ) -> TrajectoryPlan:
        return TrajectoryPlan(
            num_paths=max(1, self.retry_paths),
            strategy="single_path_retry",
            metadata={"phase": "retry", "attempts": len(attempts), "feedback": len(feedback)},
        )
