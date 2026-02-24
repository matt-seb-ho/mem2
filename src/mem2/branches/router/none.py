"""Pass-through router — returns the retrieval bundle unchanged."""
from __future__ import annotations

from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext


class NoneRouter:
    name = "none"

    async def route(
        self,
        *,
        ctx: RunContext,
        provider: object,
        problem: ProblemSpec,
        retrieval: RetrievalBundle,
    ) -> RetrievalBundle:
        return retrieval
