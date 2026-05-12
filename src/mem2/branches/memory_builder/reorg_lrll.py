"""LRLL experience-filtered wake-sleep fragment extraction — axis A.5.

Port of LRLL (Tziafas & Kasaei, 2024; arxiv 2406.18746).

Paper: literature/2406.18746.pdf
Repo:  https://gtziafas.github.io/LRLL_project/ (project page; no public code repo).

LRLL is a robotics-focused wake-sleep lifelong-learning system. The
paper-distinctive bits (simulator self-verification, vision-language
skills, CaP-style code generation) DO NOT map to mem2's declarative
ARC-concept memory — a naive port would collapse into A.2 (DreamCoder
wake-sleep) or A.4 (LILO LLM abstractions).

What IS portable — the **success-filtered experience distillation** idea:
    - Wake phase: accumulate experiences (concepts + outcomes).
    - Sleep phase: abstract ONLY from experiences that succeeded. Failures
      stay in raw memory but do not contribute new skills.
    - Mapping: on consolidate, filter concepts to those with success_rate
      ≥ `success_threshold` AND hit count ≥ `min_hits`; do DreamCoder-style
      fragment extraction on this SUCCESS SUBSET.

Specifically ported (from §III Method):
    - The wake/sleep separation: wake = accumulation (no-op in consolidate),
      sleep = abstraction (our consolidate body).
    - The self-verification filter: LRLL uses simulator-verified skills
      only; we substitute "only abstract from concepts that appeared in
      successful trajectories" as the nearest mem2 analog.
    - The skill-library-append pattern: new abstractions get written to
      the library alongside originals (like DreamCoder), not replacing them.

Deliberate simplifications:
    - No simulator, no vision, no CaP code generation — LLM-optional
      `_meta_edit_provider` can still propose new-skill names for the
      filtered fragments.
    - The wake/sleep cycle aligns to mem2's every_k consolidate step.
    - Fragment extraction reuses A.2 DreamCoder's pairwise-co-occurrence
      logic, differing ONLY in the input-concept filter.

A.5 vs A.2 / A.4 / A.7:
    - A.2 DreamCoder: fragment extraction on ALL concepts.
    - A.4 LILO: LLM abstractions on ALL concepts.
    - A.7 Memp: prunes low-success concepts AFTER the fact.
    - A.5 LRLL (this module): filter input to SUCCESS subset BEFORE
      abstraction — the "self-verification" mechanic from the paper,
      adapted to our no-simulator setting.
"""
from __future__ import annotations

import logging
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext
from mem2.scoring.mdl import MDLScorer

logger = logging.getLogger(__name__)


class LRLLWakeSleepBuilder(ArcMemoReorgMemoryBuilder):
    """Experience-filtered wake-sleep extraction."""

    name = "reorg_lrll"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        success_threshold: float = 0.5,
        min_hits: int = 2,
        min_shared_lines: int = 2,
        min_fragment_frequency: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.success_threshold = float(success_threshold)
        self.min_hits = int(min_hits)
        # Reuse DreamCoder's parameters for the inner sleep-phase extraction.
        self._dreamcoder = DreamCoderReorgBuilder(
            min_shared_lines=min_shared_lines,
            min_fragment_frequency=min_fragment_frequency,
            trigger=self.trigger,
            every_k=self.every_k,
        )

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        outcomes_by_pid = self._build_outcomes_map(reorg)
        mem = ConceptMemory.from_payload(memory.payload)

        # Wake → Sleep filter: keep only concepts with success_rate ≥ threshold
        # AND hit count ≥ min_hits. The filtered subset is what we abstract
        # from — unsuccessful concepts aren't distilled into skills.
        filtered_names: set[str] = set()
        success_stats: list[dict[str, Any]] = []
        for name, c in mem.concepts.items():
            used_in = list(getattr(c, "used_in", []) or [])
            hit = len(used_in)
            success = sum(
                1 for pid in used_in
                if outcomes_by_pid.get(str(pid), 0.0) > 0.0
            )
            rate = success / max(hit, 1) if hit > 0 else 0.0
            if hit >= self.min_hits and rate >= self.success_threshold:
                filtered_names.add(name)
                success_stats.append({
                    "name": name, "hit": hit, "success": success,
                    "rate": round(rate, 3),
                })

        if not filtered_names:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "lrll_skipped",
                "reason": f"no concepts met success_threshold={self.success_threshold} + min_hits={self.min_hits}",
                "total_concepts": len(mem.concepts),
            })
            return memory

        # Build a SLICE memory containing only the filtered concepts.
        sliced = ConceptMemory()
        for name in filtered_names:
            c = mem.concepts[name]
            sliced.concepts[name] = c
            if name not in sliced.categories[c.kind]:
                sliced.categories[c.kind].append(name)

        # Sleep phase: run DreamCoder fragment extraction on the SLICE.
        slice_state = MemoryState(
            schema_name="arcmemo_ps",
            schema_version="v1",
            payload={**sliced.to_payload(), "reorg": {"step": reorg.get("step", 0), "history": []}},
        )
        distilled = self._dreamcoder.consolidate(ctx, slice_state)
        distilled_mem = ConceptMemory.from_payload(distilled.payload)

        # Append distilled skills to the ORIGINAL memory (library-append, like
        # the paper). Do not modify the filtered-out concepts.
        appended: list[str] = []
        for name, c in distilled_mem.concepts.items():
            if name in filtered_names:
                continue
            if name not in mem.concepts:
                mem.concepts[name] = c
                if name not in mem.categories[c.kind]:
                    mem.categories[c.kind].append(name)
                appended.append(name)

        if not appended:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "lrll_skipped",
                "reason": "sleep-phase distillation produced no new skills from filtered subset",
                "filtered_count": len(filtered_names),
            })
            return memory

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "lrll_experience_filtered_distillation",
            "filtered_input_count": len(filtered_names),
            "new_skills_count": len(appended),
            "new_skills": appended,
            "success_threshold": self.success_threshold,
            "min_hits": self.min_hits,
            "top_5_filtered_by_rate": sorted(
                success_stats, key=lambda s: (-s["rate"], -s["hit"]),
            )[:5],
        })
        new_payload["reorg"] = reorg
        memory.payload = new_payload
        return memory

    # ----------------------------------------------------------------- #
    def _build_outcomes_map(self, reorg: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for o in reorg.get("outcomes", []) or []:
            pid = str(o.get("problem_id", "")).strip()
            if not pid:
                continue
            try:
                out[pid] = float(o.get("score", 0.0))
            except (TypeError, ValueError):
                continue
        return out
