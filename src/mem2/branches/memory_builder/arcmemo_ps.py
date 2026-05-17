"""ArcMemoPsMemoryBuilder: MemoryBuilder using ConceptMemory (PS formulation).

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


class ArcMemoPsMemoryBuilder:
    """Memory builder for ArcMemo-PS (Program Synthesis) formulation.

    Stores rich typed concepts via ConceptMemory. The builder loads a
    pre-built ConceptMemory from a seed file and serializes it into
    MemoryState.payload. The concept extraction workflow is offline
    (matching arc_memo's actual batch workflow), so ``update()`` only
    records solutions for correctly solved problems.
    """

    name = "arcmemo_ps"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        seed_memory_file: str | None = None,
        seed_annotations_file: str | None = None,
        domain: str = "arc",
        max_concepts: int = 0,
        freeze_memory: bool = False,
    ):
        self.seed_memory_file = seed_memory_file
        self.seed_annotations_file = seed_annotations_file
        self.domain = domain
        self.max_concepts = int(max_concepts)
        self._frozen = bool(freeze_memory)

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
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("payload"), dict):
                    return ConceptMemory.from_payload(raw["payload"])
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

        # Preserve extra payload keys from the seed file beyond the
        # ConceptMemory canonical {concepts, solutions, custom_types,
        # categories} schema. Round-2 retrievers depend on reorg-history,
        # memtree_hierarchy, and other builder-written sidecars that the
        # seed converter at scripts/sweeps/two_round_axis_2/run.py:287
        # preserves via dict(payload) — but ConceptMemory.load_from_file
        # only reads the canonical schema. Without this merge, those
        # sidecars get stripped at R2 init and the matching-retriever
        # pairing receives no reorg state. (Audit finding 2026-05-12.)
        if self.seed_memory_file:
            seed_path = self._resolve_path(self.seed_memory_file)
            if seed_path.exists():
                try:
                    seed_raw = json.loads(seed_path.read_text())
                    if isinstance(seed_raw, dict) and isinstance(seed_raw.get("payload"), dict):
                        seed_raw = seed_raw["payload"]
                    canonical = {"concepts", "solutions", "custom_types", "categories"}
                    for k, v in seed_raw.items():
                        if k not in canonical and k not in payload:
                            payload[k] = v
                except Exception:
                    pass  # silent: seed file may be in a non-JSON-toplevel-dict shape

        return MemoryState(
            schema_name="arcmemo_ps",
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

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        if self._frozen:
            return memory
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
        if self._frozen:
            return memory
        # Re-serialize (no-op for concept memory, concepts are stable)
        return memory
