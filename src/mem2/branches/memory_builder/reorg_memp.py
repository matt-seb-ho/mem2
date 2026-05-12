"""Memp procedural-memory distillation with hit/success pruning — axis A.7.

Port of Memp (Fang, Liang, Wang et al., 2025; arxiv 2508.06433).

Paper: literature/2508.06433.pdf
Repo:  third_party/memp/ (entry: ProcedureMem/memory.py::Memory.update + process_trajectory_item)

Specifically ported:
    - The *hit/success quality tracking* per memory item from `Memory.update`:
      every time a concept is retrieved (`hit += 1`), and when the trajectory
      it contributed to succeeds (`success += 1`). The tracker mirrors Memp's
      per-document metadata.
    - The *performance-based pruning* rule: delete any memory where
      `hit >= min_hits` and `success / hit < prune_threshold`. Memp's default
      is `min_hits=3` and `threshold=0.5`.
    - Optional (LLM-mode): the `process_trajectory_item_reflect` adjustment
      when a workflow fails — rewrite the concept's description to repair it.

Deliberate simplifications (LLM-optional):
    - We read the outcome signal from `memory.payload["reorg"]["outcomes"]`
      (a list of {problem_id, score} dicts maintained by the driver) and
      cross-reference against each concept's `used_in` problem list. No LLM
      needed for the pruning pass.
    - Workflow distillation ("round" and "direct" build policies from
      Memp.build) is NOT ported — those require LLM summarization of long
      trajectories. Axis A.7 focuses on the distinctive Memp mechanic:
      **evidence-based deletion**, which no other axis A candidate does.
    - The reflect/adjust path is stubbed to append a `[memp-adjust-pending]`
      marker to the description when no LLM provider is wired; the full
      rewrite activates only with a provider.

A.7 vs A.11 / A.1:
    - A.11 `accretive_prune`: prune by count cap (size-based). No notion of
      memory quality.
    - A.1 `arcmemo_reorg`: cluster & merge. No deletion.
    - A.7 Memp (this module): prune by PERFORMANCE (hit-weighted success rate).
      Retains concepts that are used AND work; deletes concepts that are used
      but regularly don't contribute to success.
"""
from __future__ import annotations

import logging
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext

logger = logging.getLogger(__name__)


class MempProceduralMemoryBuilder(ArcMemoReorgMemoryBuilder):
    """Performance-based memory pruning with hit/success tracking."""

    name = "reorg_memp"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        min_hits: int = 3,
        prune_threshold: float = 0.5,
        reflect_on_failure: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_hits = int(min_hits)
        self.prune_threshold = float(prune_threshold)
        self.reflect_on_failure = bool(reflect_on_failure)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        outcomes_by_pid: dict[str, float] = {}
        for o in reorg.get("outcomes", []) or []:
            pid = str(o.get("problem_id", "")).strip()
            if pid:
                try:
                    outcomes_by_pid[pid] = float(o.get("score", 0.0))
                except (TypeError, ValueError):
                    continue

        mem = ConceptMemory.from_payload(memory.payload)
        provider = self._resolve_provider(ctx)

        pruned: list[dict[str, Any]] = []
        reflected: list[dict[str, Any]] = []
        tracked: list[dict[str, Any]] = []

        for name in list(mem.concepts.keys()):
            c = mem.concepts[name]
            used_in = list(getattr(c, "used_in", []) or [])
            if not used_in:
                continue

            hit = len(used_in)
            success = sum(
                1 for pid in used_in
                if outcomes_by_pid.get(str(pid), 0.0) > 0.0
            )
            if hit < self.min_hits:
                tracked.append({"name": name, "hit": hit, "success": success,
                                 "status": "under_min_hits"})
                continue

            rate = success / max(hit, 1)
            tracked.append({"name": name, "hit": hit, "success": success,
                             "success_rate": round(rate, 3)})

            if rate < self.prune_threshold:
                # Memp rule: delete under-performing memory.
                kind = c.kind
                del mem.concepts[name]
                if name in mem.categories.get(kind, []):
                    mem.categories[kind].remove(name)
                pruned.append({
                    "name": name, "hit": hit, "success": success,
                    "success_rate": round(rate, 3),
                })
            elif rate < 1.0 and self.reflect_on_failure:
                # Memp reflect: failing some of the time → adjust.
                if provider is not None:
                    new_desc = self._reflect_via_llm(provider, c, outcomes_by_pid, used_in)
                    if new_desc:
                        c.description = new_desc
                        reflected.append({"name": name, "method": "llm_rewrite"})
                else:
                    marker = "[memp-adjust-pending]"
                    if marker not in (c.description or ""):
                        c.description = (c.description or "") + " " + marker
                        reflected.append({"name": name, "method": "template_marker"})

        if not pruned and not reflected:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "memp_skipped",
                "reason": "no concept met min_hits + prune_threshold criteria",
                "tracked": tracked[:10],  # cap logged tracking
            })
            return memory

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "memp_procedural_pruning",
            "pruned_count": len(pruned),
            "reflected_count": len(reflected),
            "tracked_count": len(tracked),
            "pruned": pruned,
            "reflected": reflected,
            "used_llm": provider is not None,
        })
        new_payload["reorg"] = reorg
        memory.payload = new_payload
        return memory

    # ----------------------------------------------------------------- #
    def _resolve_provider(self, ctx: RunContext):
        try:
            return (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            return None

    def _reflect_via_llm(
        self, provider, concept, outcomes_by_pid: dict[str, float],
        used_in: list[str],
    ) -> str | None:
        failing = [pid for pid in used_in if outcomes_by_pid.get(str(pid), 0.0) <= 0.0]
        passing = [pid for pid in used_in if outcomes_by_pid.get(str(pid), 0.0) > 0.0]
        prompt = (
            "You are reflecting on a partially-failing procedural memory. "
            "Rewrite its description to better capture when it works and when "
            "it doesn't. Keep it under 500 characters.\n\n"
            f"Concept name: {concept.name}\n"
            f"Current description: {concept.description}\n"
            f"Passing problems ({len(passing)}): {passing[:5]}\n"
            f"Failing problems ({len(failing)}): {failing[:5]}\n\n"
            "Rewritten description:"
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            if completions and isinstance(completions[0], str):
                return completions[0].strip()[:500]
        except Exception as exc:  # pragma: no cover
            logger.warning("memp reflect LLM call failed on %s: %s", concept.name, exc)
        return None
