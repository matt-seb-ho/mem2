from __future__ import annotations

from typing import Any

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

    def initialize(self, ctx: RunContext, problems: dict[str, ProblemSpec]) -> MemoryState:
        return MemoryState(schema_name="none", schema_version="v1", payload={})

    def reflect(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
    ) -> list[dict[str, Any]]:
        return []

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
