from __future__ import annotations

from mem2.core.entities import ProblemSpec, RunContext, TaskSpec


class ArcGridTaskAdapter:
    name = "arc_grid"

    def __init__(self, task_name: str = "arc_grid"):
        self.task_name = task_name

    def get_task_spec(self, ctx: RunContext) -> TaskSpec:
        return TaskSpec(
            task_name=self.task_name,
            task_description=(
                "Solve ARC grid transformation tasks. Output predicted test grid as JSON list-of-lists."
            ),
            sample_format={
                "input": "list[list[int]]",
                "output": "list[list[int]]",
            },
            feedback_mode="gt",
            metadata={"adapter": self.name},
        )

    def format_problem_sample(self, problem: ProblemSpec) -> dict:
        return {
            "uid": problem.uid,
            "train": problem.train_pairs,
            "test": problem.test_pairs,
        }
