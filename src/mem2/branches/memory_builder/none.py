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

    def __init__(self, **kwargs) -> None:
        # Accept and ignore any kwargs (e.g. seed_memory_file, domain, max_concepts
        # inherited from base experiment builder_cfg) so this no-op builder can be
        # used as a true bare baseline without requiring the base config to be
        # restructured around it.
        del kwargs

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
