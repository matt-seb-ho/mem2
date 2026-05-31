from __future__ import annotations

from pathlib import Path

from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "third_party/dynamic-cheatsheet").exists():
            return parent
    raise FileNotFoundError("Could not locate third_party/dynamic-cheatsheet")


def _read_upstream_prompt(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


GENERATOR_PROMPT = _read_upstream_prompt(
    "third_party/dynamic-cheatsheet/prompts/generator_prompt.txt"
)


class DcSelectorRetriever:
    name = "dc"
    COMPATIBLE_SCHEMAS = {"dc"}

    def __init__(self):
        self._generator_prompt = GENERATOR_PROMPT

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        cheatsheet = ""
        if memory is not None:
            cheatsheet = str(memory.payload.get("cheatsheet", "") or "")
        hint_text = self._generator_prompt.replace("[[CHEATSHEET]]", cheatsheet)
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text,
            retrieved_items=[],
            metadata={"retriever_mode": "dc_cumulative"},
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
