"""ALMAStyleMetaEditMemoryBuilder: Meta-Agent-proposed code-level reorg edits.

Axis F: hand-coded reorg (``arcmemo_reorg``) vs ALMA-style meta-edit.

ALMA (2602.07755, Feb 2026) learns an executable-code policy for the whole
memory architecture. We don't re-train a Meta-Agent here; instead we treat
the reorg step as a *proposed edit* emitted by an LLM and applied if
accepted by an MDL-gated validator.

This scaffold makes the axis toggleable (hand-coded vs LLM-proposed) at the
cost of an extra LLM call per reorg. The proposed edit has a strict schema:

    {
      "merges": [
        {"aggregate_name": str, "members": [str, ...], "description": str},
        ...
      ],
      "deletions": [str, ...],
      "renames": [{"from": str, "to": str}, ...]
    }

If the provider/LLM is not wired at consolidate time, the builder falls back
to the hand-coded reorg operation so runs never abort.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.data import Concept, ParameterSpec
from mem2.concepts.graph import ConceptGraph
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


META_EDIT_SYSTEM = (
    "You are a memory-reorganization agent for an ARC concept library. "
    "Propose a small set of merges, deletions, and renames that reduce "
    "redundancy WITHOUT losing coverage. Output JSON matching this schema: "
    '{"merges":[{"aggregate_name":str,"members":[str,...],"description":str}],'
    '"deletions":[str],"renames":[{"from":str,"to":str}]}'
)


class ALMAStyleMetaEditMemoryBuilder(ArcMemoReorgMemoryBuilder):
    """Meta-edit reorg with an LLM-proposed edit plan + MDL gate.

    Inherits trigger/scope/objective/input-basis knobs from
    ``ArcMemoReorgMemoryBuilder``; adds an LLM hook to propose the edit plan.
    Falls back to the parent's hand-coded selection if no provider is wired.
    """

    name = "alma_style_metaedit"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        max_candidates: int = 16,
        min_mdl_gain: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_candidates = int(max_candidates)
        self.min_mdl_gain = float(min_mdl_gain)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if getattr(self, "_frozen", False):
            return memory
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        plan = self._propose_edit_plan(ctx, mem)
        if not plan or not plan.get("merges"):
            return self._record_alma_skip(memory, reorg, "provider absent or no merge plan returned")

        scorer = MDLScorer(per_concept_overhead=self.scorer.per_concept_overhead)
        before = scorer.score(mem).total
        accepted: list[dict[str, Any]] = []
        renames = plan.get("renames", [])
        deletions = plan.get("deletions", [])

        for merge in plan.get("merges", []):
            name = str(merge.get("aggregate_name") or "").strip()
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

        for raw in renames:
            frm = str(raw.get("from", "")).strip()
            to = str(raw.get("to", "")).strip()
            if not frm or not to or frm not in mem.concepts or to in mem.concepts:
                continue
            c = mem.concepts.pop(frm)
            c.name = to
            mem.concepts[to] = c
            for kind, names in mem.categories.items():
                if frm in names:
                    names[names.index(frm)] = to
            accepted.append({"type": "rename", "from": frm, "to": to})

        for name in deletions:
            name = str(name).strip()
            if name in mem.concepts:
                kind = mem.concepts[name].kind
                del mem.concepts[name]
                if name in mem.categories.get(kind, []):
                    mem.categories[kind].remove(name)
                accepted.append({"type": "delete", "name": name})

        if not accepted:
            return self._record_alma_skip(memory, reorg, "no proposed edits passed validation")

        after = scorer.score(mem).total
        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "alma_meta_edit",
            "accepted": accepted,
            "mdl_before": before,
            "mdl_after": after,
            "mdl_delta": after - before,
            "scope": self.scope,
            "objective": self.objective,
        })
        new_payload["reorg"] = reorg
        memory.payload = new_payload
        return memory

    def _record_alma_skip(
        self, memory: MemoryState, reorg: dict[str, Any], reason: str
    ) -> MemoryState:
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "alma_meta_edit_skipped",
            "reason": reason,
            "scope": self.scope,
            "objective": self.objective,
            "input_basis": self.input_basis,
        })
        memory.payload["reorg"] = reorg
        return memory

    # ----------------------------------------------------------------- #
    def _propose_edit_plan(
        self, ctx: RunContext, mem: ConceptMemory
    ) -> dict[str, Any] | None:
        """Return a parsed plan dict, or None if no provider is wired.

        This reads a provider handle from ``ctx.config`` if present
        (``ctx.config["_meta_edit_provider"]``). We keep the LLM wiring
        optional so the builder can run in offline / mock mode and the axis-F
        experiment can still be orchestrated (falling back to hand-coded).
        """
        provider = None
        try:
            provider = (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            provider = None
        if provider is None:
            return None

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        candidates = sorted(
            mem.concepts.keys(),
            key=lambda n: graph.degree(n, kinds=["co_activation"]),
            reverse=True,
        )[: self.max_candidates]
        context = mem.to_string(concept_names=candidates)

        prompt = (
            f"{META_EDIT_SYSTEM}\n\nCurrent concept memory (top-{len(candidates)} by degree):\n"
            f"{context}\n\nOutput JSON only."
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = completions[0] if completions else "{}"
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - provider-dependent
            logger.warning("alma_meta_edit LLM proposal failed: %s", exc)
            return None
