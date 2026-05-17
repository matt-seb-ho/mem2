"""LILO library-growth / LLM-guided abstraction — axis A.4.

Port of LILO (Grand, Wong, Bowers et al., ICLR'24; arxiv 2310.19791).

Paper: literature/2310.19791.pdf
Repo:  third_party/lilo/ (entry: src/models/gpt_abstraction.py::GPTLibraryLearner)

Specifically ported:
    - The *iterative-abstraction-proposal* loop from
      `GPTLibraryLearner.generate_abstraction`: for each of
      `n_function_generated` iterations, prompt the LLM with the current
      library + task/program examples, ask for ONE new named abstraction as
      JSON ({readable_name, function_expression, description}), validate, and
      accept into the growing library so the next iteration sees the updated
      state.
    - The abstraction schema validation flow (`_parse_completion` + `check_valid`):
      name uniqueness, members-exist check, and parseability.

Deliberate simplifications (LLM-optional):
    - The LLM provider is read from `ctx.config["_meta_edit_provider"]` (the
      same hook used by F.2 / F.3). If absent, fall back to a template-based
      generator: each pass picks the highest-co-activation pair still
      un-aggregated and emits `auto_lilo_abstraction_<i>` as the name. This
      is distinct from A.1 (which does a single global rebuild with
      community-detection clusters) because we emit one abstraction per pass
      — the iterative dynamic is the LILO-distinctive bit, not the specific
      cluster source.
    - DreamCoder/Stitch compression around LILO (the "C" and "O" of LILO) is
      deferred — those are Axis A.2 and A.3 respectively. This module ports
      only the "LL" (LLM library-growth) piece.

A.4 vs F.2/F.3:
    - A.4 (this module) is a *memory-builder* (Axis A role) that GROWS the
      library with N new named abstractions per consolidation.
    - F.2/F.3 are *meta-edit* builders (Axis F role) that PROPOSE edit plans
      (merges + deletions + renames) in one shot (F.2) or with reflexion
      (F.3).
    - A.4's iteration is over *abstractions* (one per call); F.3's iteration
      is over *plans* (many merges per call, reflected on between calls).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext
from mem2.scoring.mdl import MDLScorer

logger = logging.getLogger(__name__)


LILO_SYSTEM = (
    "You are a library-growth agent for an ARC concept memory. "
    "Propose ONE new named abstraction that subsumes a cluster of related "
    "existing concepts, reducing redundancy without losing coverage. "
    "Output JSON only, matching this schema: "
    '{"readable_name": str, "members": [str, ...], "description": str}'
)


class LILOLibraryGrowthBuilder(ArcMemoReorgMemoryBuilder):
    """Iterative LLM-guided abstraction growth. Subclass of A.1 reorg."""

    name = "reorg_lilo"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        n_function_generated: int = 10,
        n_task_examples: int = 10,
        n_program_examples: int = 20,
        max_candidates: int = 16,
        min_mdl_gain: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_function_generated = int(n_function_generated)
        self.n_task_examples = int(n_task_examples)
        self.n_program_examples = int(n_program_examples)
        self.max_candidates = int(max_candidates)
        self.min_mdl_gain = float(min_mdl_gain)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if getattr(self, "_frozen", False):
            return memory
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        provider = self._resolve_provider(ctx)
        scorer = MDLScorer(per_concept_overhead=self.scorer.per_concept_overhead)

        grown: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, ...]] = set()
        used_names: set[str] = set(mem.concepts.keys())

        for fn_idx in range(self.n_function_generated):
            if provider is not None:
                proposal = self._propose_via_llm(ctx, mem, provider, fn_idx, grown)
            else:
                proposal = self._propose_via_template(mem, fn_idx, seen_pairs)

            if proposal is None:
                continue
            accepted = self._accept_if_valid(
                proposal, mem, scorer, used_names,
            )
            if accepted is None:
                continue
            used_names.add(accepted["readable_name"])
            grown.append(accepted)

        if not grown:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "lilo_skipped",
                "reason": "no proposal passed validation across "
                         f"{self.n_function_generated} iterations",
            })
            return memory

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "lilo_library_growth",
            "abstractions_added": len(grown),
            "grown": grown,
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

    def _propose_via_llm(
        self,
        ctx: RunContext,
        mem: ConceptMemory,
        provider,
        fn_idx: int,
        grown: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        candidates = sorted(
            mem.concepts.keys(),
            key=lambda n: graph.degree(n, kinds=["co_activation"]), reverse=True,
        )[: self.max_candidates]
        context = mem.to_string(concept_names=candidates)

        prior_block = ""
        if grown:
            prior_block = "\n\nAbstractions already grown this round:\n" + "\n".join(
                f"- {g['readable_name']}: [{', '.join(g['members'])}]" for g in grown
            )

        prompt = (
            f"{LILO_SYSTEM}\n\n"
            f"Iteration {fn_idx + 1} of {self.n_function_generated}.\n"
            f"Current library (top-{len(candidates)} by co-activation):\n{context}"
            f"{prior_block}\n\n"
            "Output JSON only."
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = completions[0] if completions else "{}"
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - provider-dependent
            logger.warning("lilo LLM proposal failed at iter %d: %s", fn_idx, exc)
            return None

    def _propose_via_template(
        self,
        mem: ConceptMemory,
        fn_idx: int,
        seen_pairs: set[tuple[str, ...]],
    ) -> dict[str, Any] | None:
        """Fallback: pick the highest-co-activation pair not yet proposed."""
        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        edges: list[tuple[float, str, str]] = []
        for src in mem.concepts:
            for dst, kind, weight in graph.neighbors(src, kinds=["co_activation"]):
                if dst <= src:
                    continue
                edges.append((float(weight), src, dst))
        edges.sort(reverse=True)
        for _, src, dst in edges:
            pair = tuple(sorted((src, dst)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            return {
                "readable_name": f"auto_lilo_abstraction_{fn_idx}",
                "members": [src, dst],
                "description": (
                    f"Template-generated abstraction over co-activated pair "
                    f"({src}, {dst}). LILO distinctive: one abstraction per pass."
                ),
            }
        return None

    def _accept_if_valid(
        self,
        proposal: dict[str, Any],
        mem: ConceptMemory,
        scorer: MDLScorer,
        used_names: set[str],
    ) -> dict[str, Any] | None:
        name = str(proposal.get("readable_name") or "").strip()
        members = [str(m).strip() for m in proposal.get("members", []) if m]
        description = str(proposal.get("description") or "").strip()

        if not name or len(members) < self.min_group_size:
            return None
        if name in used_names:
            return None
        if not all(m in mem.concepts for m in members):
            return None
        if self.objective == "mdl":
            gain = -scorer.local_diff(mem, members)
            if gain < self.min_mdl_gain:
                return None

        agg = self._aggregate(name, members, mem)
        if description:
            agg.description = description[:2000]
        mem.concepts[name] = agg
        if name not in mem.categories[agg.kind]:
            mem.categories[agg.kind].append(name)

        return {
            "readable_name": name,
            "members": members,
            "description": description,
        }
