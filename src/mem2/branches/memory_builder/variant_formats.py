"""Format-variant builders for axis D.

Five new variants in addition to ``arcmemo_oe`` and ``arcmemo_ps``:

  - minimal           : concept name + short description
  - typed_only        : name + kind + description (no cues, no impl, no params)
  - cue_heavy         : name + cues (description hidden)
  - free_text         : descriptions concatenated as a paragraph
  - structured_routine: full structured render with kind/subtype/parameters

All variants share the ``arcmemo_ps`` schema so existing retrievers remain
compatible. The chosen variant is stamped into ``metadata.variant`` at init
time; the retriever reads it when rendering.
"""
from __future__ import annotations

from dataclasses import asdict

from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.concepts.memory import ConceptMemory, ProblemSolution
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)

VARIANTS = {
    "minimal",
    "typed_only",
    "cue_heavy",
    "free_text",
    "structured_routine",
}

RENDER_FLAGS = {
    "minimal":            dict(skip_cues=True,  skip_implementation=True,  skip_parameters=True,  include_description=True),
    "typed_only":         dict(skip_cues=True,  skip_implementation=True,  skip_parameters=True,  include_description=True,  skip_kind=False),
    "cue_heavy":          dict(skip_cues=False, skip_implementation=True,  skip_parameters=True,  include_description=False),
    "free_text":          dict(skip_cues=True,  skip_implementation=True,  skip_parameters=True,  include_description=True),
    "structured_routine": dict(skip_cues=False, skip_implementation=False, skip_parameters=False, include_description=True, skip_kind=False, skip_routine_subtype=False),
}


class VariantFormatBuilder:
    """Thin wrapper over ``ArcMemoPsMemoryBuilder`` that stamps a format variant.

    All logic is delegated to the PS builder; the variant flag lives in
    ``memory.metadata["variant"]``. Retrievers (including the plain
    ``ps_selector``) can read the variant and pick the matching render-flag
    set from ``RENDER_FLAGS``.
    """

    name = "variant_format"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        variant: str = "minimal",
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 0,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Valid: {sorted(VARIANTS)}"
            )
        self.variant = variant
        self._inner = ArcMemoPsMemoryBuilder(
            seed_memory_file=seed_memory_file,
            seed_annotations_file=seed_annotations_file,
            domain=domain,
            max_concepts=max_concepts,
        )

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        state = self._inner.initialize(ctx, problems)
        state.metadata["variant"] = self.variant
        state.metadata["render_flags"] = dict(RENDER_FLAGS[self.variant])
        return state

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        solutions = memory.payload.get("solutions", {})
        for i, att in enumerate(attempts):
            is_correct = eval_records[i].is_correct if i < len(eval_records) else False
            if is_correct:
                solutions[att.problem_uid] = asdict(
                    ProblemSolution(
                        problem_id=att.problem_uid,
                        solution=(att.completion or "")[:2000],
                    )
                )
        memory.payload["solutions"] = solutions
        memory.metadata.setdefault("variant", self.variant)
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        return self._inner.consolidate(ctx, memory)
