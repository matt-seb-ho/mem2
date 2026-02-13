from __future__ import annotations

from mem2.core.entities import ProblemSpec, RunContext, TaskSpec


class MathPsTaskAdapter:
    """Task adapter for math problems solved via Python code (PS format).

    Problems are competition math (Number Theory, Counting & Probability, etc.)
    where the model writes a Python function ``solve()`` that returns the answer.
    """

    name = "math_ps"

    def __init__(self, task_name: str = "math_ps"):
        self.task_name = task_name

    def get_task_spec(self, ctx: RunContext) -> TaskSpec:
        return TaskSpec(
            task_name=self.task_name,
            task_description=(
                "Solve competition math problems by writing a Python function "
                "solve() that returns the integer answer."
            ),
            sample_format={
                "input": "problem statement (str)",
                "output": "integer answer",
            },
            feedback_mode="gt",
            metadata={"adapter": self.name},
        )

    def format_problem_sample(self, problem: ProblemSpec) -> dict:
        return {
            "uid": problem.uid,
            "problem": problem.metadata.get("problem_text", ""),
            "answer": problem.metadata.get("answer"),
            "type": problem.metadata.get("math_type", ""),
            "level": problem.metadata.get("level", ""),
        }
