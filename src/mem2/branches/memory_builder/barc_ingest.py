"""BARCIngestMemoryBuilder: bulk-ingest BARC seeds into concept memory.

Axis E: empty-start (accretive baseline) vs BARC-seeded-start.

BARC seeds (``arc_memo/data/dataset/src/BARC/seeds/``) are Python files with
a standardized header::

    # concepts:
    # <comma-separated concept names>
    #
    # description:
    # <free-text description>
    #
    def main(input_grid): ...
    def generate_input(): ...

The ingester parses these, creates one ``Concept`` per listed concept name,
and populates ``used_in`` with the seed filename (task ID). When multiple
seeds mention the same concept, ``ConceptMemory.update()`` merges cues /
descriptions / used_in automatically.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict
from pathlib import Path

from mem2.concepts.memory import ConceptMemory, ProblemSolution
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(
    r"#\s*concepts:\s*(.*?)"
    r"#\s*description:\s*(.*?)"
    r"(?:def\s+main|\n\n[^#])",
    re.DOTALL,
)


def _parse_header(text: str) -> tuple[list[str], str] | None:
    m = _HEADER_RE.search(text)
    if not m:
        return None
    concept_block = m.group(1)
    desc_block = m.group(2)
    concepts: list[str] = []
    for line in concept_block.splitlines():
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        for raw in line.split(","):
            c = raw.strip()
            if c:
                concepts.append(c)
    description = " ".join(
        line.strip().lstrip("#").strip()
        for line in desc_block.splitlines()
        if line.strip().lstrip("#").strip()
    )
    return concepts, description


def ingest_barc_dir(barc_dir: Path, mem: ConceptMemory) -> dict:
    """Ingest every ``.py`` file under ``barc_dir``. Returns stats dict."""
    if not barc_dir.exists():
        return {"seeds_found": 0, "seeds_parsed": 0, "concepts_written": 0}
    seed_files = sorted(p for p in barc_dir.iterdir() if p.suffix == ".py")
    parsed = 0
    concepts_written = 0
    for seed in seed_files:
        try:
            text = seed.read_text(errors="ignore")
        except OSError:
            continue
        hdr = _parse_header(text)
        if not hdr:
            continue
        parsed += 1
        concepts, description = hdr
        task_id = seed.stem
        mem.solutions[task_id] = ProblemSolution(
            problem_id=task_id,
            solution=text[:2000],
            summary=description[:500] or None,
        )
        for concept_name in concepts:
            # write/merge a routine concept
            mem.write_concept(task_id, {
                "concept": concept_name,
                "kind": "routine",
                "routine_subtype": "grid manipulation",
                "description": description[:300] or None,
                "cues": [description[:160]] if description else [],
            })
            concepts_written += 1
    return {
        "seeds_found": len(seed_files),
        "seeds_parsed": parsed,
        "concepts_written": concepts_written,
        "unique_concepts": len(mem.concepts),
    }


class BARCIngestMemoryBuilder:
    """Memory builder that pre-seeds ConceptMemory with BARC concepts.

    Subsequent ``update()`` records solved-problem ``ProblemSolution`` entries
    (same semantics as ``ArcMemoPsMemoryBuilder.update``). ``consolidate`` is
    a no-op — use ``arcmemo_reorg`` if you want reorg on top.
    """

    name = "barc_ingest"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        barc_dir: str = "../arc_memo/data/dataset/src/BARC/seeds",
        domain: str = "arc",
    ) -> None:
        self.barc_dir = Path(barc_dir).expanduser()
        self.domain = domain

    def _resolve_dir(self) -> Path:
        if self.barc_dir.is_absolute():
            return self.barc_dir
        return (Path.cwd() / self.barc_dir).resolve()

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        mem = ConceptMemory()
        stats = ingest_barc_dir(self._resolve_dir(), mem)
        payload = mem.to_payload()
        return MemoryState(
            schema_name="arcmemo_ps",
            schema_version="v1",
            payload=payload,
            metadata={
                "initialized_problem_count": len(problems),
                "barc_dir": str(self._resolve_dir()),
                "domain": self.domain,
                **stats,
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
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        return memory
