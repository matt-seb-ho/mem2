"""ConceptPsMemoryBuilder: MemoryBuilder using ConceptMemory.

Loads a seed ConceptMemory from a JSON file or annotations file,
serializes into MemoryState.payload for use by ConceptSelectorRetriever.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mem2.concepts.memory import ConceptMemory, ProblemSolution
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)


class ConceptPsMemoryBuilder:
    """Memory builder that stores rich typed concepts via ConceptMemory.

    The builder loads a pre-built ConceptMemory from a seed file and
    serializes it into MemoryState.payload. The concept extraction
    workflow is offline (matching arc_memo's actual batch workflow),
    so ``update()`` only records solutions for correctly solved problems.
    """

    name = "concept_ps"

    def __init__(
        self,
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 0,
        **kwargs,  # absorb extra keys from config inheritance (e.g. max_entries)
    ):
        self.seed_memory_file = seed_memory_file
        self.seed_annotations_file = seed_annotations_file
        self.domain = domain
        self.max_concepts = int(max_concepts)

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    def _load_seed_memory(self) -> ConceptMemory:
        mem = ConceptMemory()

        if self.seed_memory_file:
            path = self._resolve_path(self.seed_memory_file)
            if path.exists():
                mem.load_from_file(path)
                return mem

        if self.seed_annotations_file:
            path = self._resolve_path(self.seed_annotations_file)
            if path.exists():
                annotations = json.loads(path.read_text())
                mem.initialize_from_annotations(annotations)
                return mem

        return mem

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        concept_mem = self._load_seed_memory()
        payload = concept_mem.to_payload()

        return MemoryState(
            schema_name="concept_ps",
            schema_version="v1",
            payload=payload,
            metadata={
                "initialized_problem_count": len(problems),
                "seed_memory_file": self.seed_memory_file,
                "seed_annotations_file": self.seed_annotations_file,
                "domain": self.domain,
                "concept_count": len(concept_mem.concepts),
                "solution_count": len(concept_mem.solutions),
            },
        )

    def reflect(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
    ) -> list[dict]:
        items = []
        for idx, att in enumerate(attempts):
            fb_txt = feedback[idx].content if idx < len(feedback) else ""
            items.append(
                {
                    "problem_uid": problem.uid,
                    "attempt_idx": idx,
                    "attempt_preview": att.completion[:200],
                    "feedback": fb_txt[:200],
                }
            )
        return items

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        # For correct solutions: store ProblemSolution in solutions dict
        solutions = memory.payload.get("solutions", {})
        for i, att in enumerate(attempts):
            is_correct = eval_records[i].is_correct if i < len(eval_records) else False
            if is_correct:
                solutions[att.problem_uid] = asdict(
                    ProblemSolution(
                        problem_id=att.problem_uid,
                        solution=att.completion[:2000],
                        summary=None,
                        pseudocode=None,
                    )
                )
        memory.payload["solutions"] = solutions
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        # Re-serialize (no-op for concept memory, concepts are stable)
        return memory
