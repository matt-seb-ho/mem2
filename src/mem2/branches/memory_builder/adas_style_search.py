"""ADAS Meta Agent Search meta-edit builder — axis F.3.

Port of ADAS (Hu, Lu, Clune ICLR'25 / NeurIPS'24 Outstanding-Paper Workshop).

Paper: literature/2408.08435.pdf
Repo:  third_party/adas/ (entry: _arc/search.py + arc_prompt.py — GPT-based meta-loop)

Specifically ported:
    - The *iterative* meta-search pattern with reflexion: LLM proposes an
      edit, outcome is evaluated, reflexion feedback updates the prompt,
      loop. Distinct from F.2 ALMA's single-shot edit proposal.

Deliberate simplifications (LLM-optional):
    - Like F.2 `alma_style_metaedit`, the LLM provider is read from
      `ctx.config["_meta_edit_provider"]`. If absent, fall back to F.2
      hand-coded behavior (which itself falls back to A.1 novel reorg).
    - The ADAS-distinctive bit: a *reflexion buffer* accumulates prior
      proposals + outcomes, and each LLM call is fed the full history.
      Without an LLM provider, this degenerates to "attempt one hand-
      coded reorg" — honest behavior vs a silent "works" that doesn't
      exercise the meta-search.

F.3 vs F.2 (the axis-F ablation):
    - F.2: single LLM-proposed edit plan + MDL gate. One-shot.
    - F.3: iterative (up to `max_reflexion_rounds`) with accumulated
      reflexion history; each round's feedback shapes the next proposal.
      Multi-shot meta-search.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)
from mem2.scoring.mdl import MDLScorer

logger = logging.getLogger(__name__)


ADAS_SYSTEM = (
    "You are a Meta Agent designing a memory-reorganization policy. "
    "You will see prior proposals and their outcomes. Output JSON for the "
    "NEXT proposal, incorporating what worked / didn't work. Schema: "
    '{"merges":[{"aggregate_name":str,"members":[str,...],"description":str}],'
    '"deletions":[str],"renames":[{"from":str,"to":str}],"rationale":str}'
)


class ADASMetaSearchBuilder(ALMAStyleMetaEditMemoryBuilder):
    """Iterative reflexion-driven meta-edit. Subclass of F.2 ALMA-style."""

    name = "adas_style_search"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        max_reflexion_rounds: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_reflexion_rounds = int(max_reflexion_rounds)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if getattr(self, "_frozen", False):
            return memory
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        provider = self._resolve_provider(ctx)
        if provider is None:
            # No LLM → fall back to parent (F.2 behavior)
            return super().consolidate(ctx, memory)

        reflexion_buffer: list[dict[str, Any]] = []
        scorer = MDLScorer(per_concept_overhead=self.scorer.per_concept_overhead)
        best_mem = None
        best_mdl = scorer.score(mem).total
        total_committed: list[dict[str, Any]] = []

        for round_idx in range(self.max_reflexion_rounds):
            plan = self._propose_with_reflexion(
                ctx, mem, reflexion_buffer, provider,
            )
            if not plan or not plan.get("merges"):
                reflexion_buffer.append({
                    "round": round_idx,
                    "outcome": "skipped — no merges proposed",
                })
                continue

            # Apply plan into a sim; check MDL
            sim = ConceptMemory.from_payload(mem.to_payload())
            accepted_this_round = self._apply_plan(plan, sim)
            sim_mdl = scorer.score(sim).total

            if sim_mdl < best_mdl and accepted_this_round:
                # Accept into running-best
                mem = sim
                best_mdl = sim_mdl
                best_mem = sim
                total_committed.extend(accepted_this_round)
                reflexion_buffer.append({
                    "round": round_idx,
                    "outcome": f"accepted {len(accepted_this_round)} edits; mdl → {sim_mdl:.1f}",
                    "rationale": plan.get("rationale", ""),
                })
            else:
                reflexion_buffer.append({
                    "round": round_idx,
                    "outcome": f"rejected (mdl {sim_mdl:.1f} ≥ best {best_mdl:.1f})",
                    "rationale": plan.get("rationale", ""),
                })

        if not total_committed:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "adas_skipped",
                "reason": f"no reflexion round improved MDL after {self.max_reflexion_rounds} tries",
                "reflexion_buffer": reflexion_buffer,
            })
            return memory

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "adas_meta_search",
            "committed": total_committed,
            "rounds": len(reflexion_buffer),
            "reflexion_buffer": reflexion_buffer,
            "final_mdl": best_mdl,
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

    def _propose_with_reflexion(
        self, ctx: RunContext, mem: ConceptMemory,
        reflexion_buffer: list[dict[str, Any]], provider,
    ) -> dict[str, Any] | None:
        from mem2.concepts.graph import ConceptGraph
        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        top = sorted(
            mem.concepts.keys(),
            key=lambda n: graph.degree(n, kinds=["co_activation"]), reverse=True,
        )[: self.max_candidates]
        context = mem.to_string(concept_names=top)

        history_block = ""
        if reflexion_buffer:
            history_block = "\n\nPRIOR ROUNDS:\n" + "\n".join(
                f"round {b['round']}: {b['outcome']} {('rationale: ' + b['rationale']) if b.get('rationale') else ''}"
                for b in reflexion_buffer
            )

        prompt = (
            f"{ADAS_SYSTEM}\n\nCurrent concept memory (top-{len(top)} by degree):\n"
            f"{context}{history_block}\n\nOutput JSON only."
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = completions[0] if completions else "{}"
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover
            logger.warning("adas_meta_search proposal failed: %s", exc)
            return None

    def _apply_plan(self, plan: dict[str, Any], mem: ConceptMemory) -> list[dict[str, Any]]:
        """Apply a plan to `mem` in-place; return the accepted edits. Same
        validation rules as the parent ALMA-style (MDL + min_group_size)."""
        accepted: list[dict[str, Any]] = []
        scorer = MDLScorer(per_concept_overhead=self.scorer.per_concept_overhead)
        for merge in plan.get("merges", []):
            name = str(merge.get("aggregate_name", "")).strip()
            members = [str(m).strip() for m in merge.get("members", []) if m]
            if not name or len(members) < self.min_group_size:
                continue
            if self.objective == "mdl":
                gain = -scorer.local_diff(mem, members)
                if gain < self.min_mdl_gain:
                    continue
            agg = self._aggregate(name, members, mem)
            if merge.get("description"):
                agg.description = str(merge["description"])[:2000]
            mem.concepts[name] = agg
            if name not in mem.categories[agg.kind]:
                mem.categories[agg.kind].append(name)
            accepted.append({"type": "merge", "aggregate": name, "members": members})
        return accepted
