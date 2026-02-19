"""Task adapter for LiveCodeBench problems."""

from __future__ import annotations

from mem2.core.entities import ProblemSpec, RunContext, TaskSpec


class LiveCodeBenchTaskAdapter:
    """Task adapter for LiveCodeBench code generation problems.

    Problems are competitive programming problems where the model writes
    a Python solution that reads from stdin and writes to stdout.
    """

    name = "livecodebench"

    def __init__(self, task_name: str = "livecodebench"):
        self.task_name = task_name

    def get_task_spec(self, ctx: RunContext) -> TaskSpec:
        return TaskSpec(
            task_name=self.task_name,
            task_description=(
                "Solve competitive programming problems by writing Python code "
                "that reads from stdin and writes to stdout."
            ),
            sample_format={
                "input": "problem statement (str) with test cases",
                "output": "stdout for each test case",
            },
            feedback_mode="gt",
            metadata={"adapter": self.name},
        )

    def format_problem_sample(self, problem: ProblemSpec) -> dict:
        return {
            "uid": problem.uid,
            "problem": problem.metadata.get("question_content", ""),
            "difficulty": problem.metadata.get("difficulty", ""),
            "public_tests": len(problem.metadata.get("public_test_cases", [])),
            "private_tests": len(problem.metadata.get("private_test_cases", [])),
        }
