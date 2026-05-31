"""ACE playbook retriever."""
from __future__ import annotations

from mem2.branches.memory_builder.ace import parse_playbook_line
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


class AceSelectorRetriever:
    name = "ace"
    COMPATIBLE_SCHEMAS = {"ace"}

    def __init__(self, max_bullets: int = 50):
        self.max_bullets = int(max_bullets)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        playbook = memory.payload.get("playbook_text", "") if memory else ""
        lines = playbook.strip().split("\n") if playbook else []
        parsed = [parse_playbook_line(line) for line in lines]
        parsed = [p for p in parsed if p is not None]
        parsed.sort(key=lambda p: (-int(p["helpful"]), int(p["harmful"]), str(p["id"])))
        top = parsed[: self.max_bullets]
        hint = "\n".join(str(p["raw_line"]) for p in top) if top else None
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint,
            retrieved_items=[{"bullet_id": str(p["id"])} for p in top],
            metadata={
                "selector_mode": "ace_playbook",
                "bullet_count": len(parsed),
                "selected_count": len(top),
            },
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
