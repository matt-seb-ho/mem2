from __future__ import annotations

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)


class NoneMemoryBuilder:
    name = "none"
    SCHEMA_NAME = "none"

    def initialize(self, ctx: RunContext, problems: dict[str, ProblemSpec]) -> MemoryState:
        return MemoryState(schema_name="none", schema_version="v1", payload={})

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        return memory
