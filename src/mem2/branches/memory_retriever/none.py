from __future__ import annotations

from mem2.core.entities import AttemptRecord, MemoryState, ProblemSpec, RetrievalBundle, RunContext


class NoneMemoryRetriever:
    name = "none"
    COMPATIBLE_SCHEMAS = {"none", "arcmemo_oe", "arcmemo_ps"}

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=None,
            retrieved_items=[],
            metadata={"selector_mode": "none"},
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        return self.retrieve(ctx, memory, problem, previous_attempts)
