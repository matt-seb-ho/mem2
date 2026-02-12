from __future__ import annotations

from typing import Any, Protocol

from .entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
    TaskSpec,
    TrajectoryPlan,
)


class TaskAdapter(Protocol):
    name: str

    def get_task_spec(self, ctx: RunContext) -> TaskSpec: ...

    def format_problem_sample(self, problem: ProblemSpec) -> dict[str, Any]: ...


class BenchmarkAdapter(Protocol):
    name: str

    def load(self, ctx: RunContext) -> dict[str, ProblemSpec]: ...

    def validate(self, problems: dict[str, ProblemSpec]) -> None: ...


class MemoryBuilder(Protocol):
    name: str

    def initialize(self, ctx: RunContext, problems: dict[str, ProblemSpec]) -> MemoryState: ...

    def reflect(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
    ) -> list[dict[str, Any]]: ...

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState: ...

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState: ...


class MemoryRetriever(Protocol):
    name: str

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle: ...


class TrajectoryPolicy(Protocol):
    name: str

    def plan_initial(self, ctx: RunContext, problem: ProblemSpec) -> TrajectoryPlan: ...

    def plan_retry(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
    ) -> TrajectoryPlan: ...


class ProviderClient(Protocol):
    name: str
    version: str
    supports_multi_completion: bool

    async def async_generate(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[str]: ...

    async def async_batch_generate(
        self,
        prompts: list[str | list[dict[str, Any]]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[list[str | None]]: ...

    def get_usage_snapshot(self) -> dict[str, Any]: ...


class InferenceEngine(Protocol):
    name: str

    async def initial_attempt(
        self,
        ctx: RunContext,
        provider: ProviderClient,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        trajectory_plan: TrajectoryPlan,
    ) -> list[AttemptRecord]: ...

    async def retry_attempt(
        self,
        ctx: RunContext,
        provider: ProviderClient,
        problem: ProblemSpec,
        retrieval: RetrievalBundle | None,
        attempt_history: list[AttemptRecord],
        feedback_history: list[FeedbackRecord],
        trajectory_plan: TrajectoryPlan,
    ) -> list[AttemptRecord]: ...


class FeedbackEngine(Protocol):
    name: str

    async def generate(
        self,
        ctx: RunContext,
        provider: ProviderClient | None,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord] | None,
    ) -> list[FeedbackRecord]: ...


class Evaluator(Protocol):
    name: str

    def evaluate(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
    ) -> list[EvalRecord]: ...

    def aggregate(self, ctx: RunContext, records: list[EvalRecord]) -> dict[str, Any]: ...


class ArtifactSink(Protocol):
    name: str

    def write_stage_artifact(self, ctx: RunContext, stage: str, data: Any) -> str: ...

    def write_run_summary(self, ctx: RunContext, summary: dict[str, Any]) -> str: ...
