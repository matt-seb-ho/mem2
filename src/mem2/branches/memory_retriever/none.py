from __future__ import annotations

from mem2.core.entities import AttemptRecord, MemoryState, ProblemSpec, RetrievalBundle, RunContext


class NoneMemoryRetriever:
    name = "none"
    COMPATIBLE_SCHEMAS = {"none", "arcmemo_oe", "arcmemo_ps"}

    def __init__(self, **kwargs) -> None:
        # Accept and ignore any kwargs (e.g. top_k, domain inherited from base
        # experiment retriever_cfg) so this no-op retriever can be used as a true
        # bare baseline without requiring the base config to be restructured.
        del kwargs

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
