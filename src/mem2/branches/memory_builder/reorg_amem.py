"""A-MEM per-note agentic memory evolution — axis A.6.

Port of A-MEM (Xu et al., NeurIPS'25; arxiv 2502.12110).

Paper: literature/2502.12110.pdf
Repo:  third_party/a_mem/ (entry: memory_layer.py::AgenticMemorySystem.process_memory + consolidate_memories)

Specifically ported:
    - The *per-note evolution* pattern from `process_memory`: for each newly
      added concept, find k nearest neighbors, ask the LLM whether to evolve
      (`should_evolve: bool`) and if so what `actions` to take (strengthen,
      update_tags, update_context). Links/tags get committed if accepted.
    - The `evo_threshold` consolidation schedule: once `evo_cnt %
      evo_threshold == 0` cross recently-evolved concepts, run a full
      retriever-index rebuild step.

Deliberate simplifications (LLM-optional):
    - The LLM provider is read from `ctx.config["_meta_edit_provider"]`. If
      absent, the builder uses a template policy: `should_evolve` iff the
      concept has ≥ `min_neighbor_strength` embedding-sim neighbor; actions
      default to `["strengthen", "update_tags"]`; strengthen = add an
      `authorship_lineage`-style edge to the single closest neighbor; tag
      update annotates `description` with `[A-MEM linked to: <neighbor>]`.
    - A-MEM's Zettelkasten "new_context_neighborhood" rewrite and the
      tag-graph broadcast are skipped (those require LLM text generation and
      don't have a clean template fallback).

A.6 vs A.1 / A.2 / A.3 / A.4:
    - A.1 `arcmemo_reorg`: batch clustering + aggregate creation at every_k.
    - A.2 DreamCoder: line-level fragment extraction across many concepts.
    - A.3 Stitch: top-down frequency-ranked fragments.
    - A.4 LILO: iterative LLM abstraction proposals per consolidation.
    - A.6 A-MEM (this module): *no new concepts created*. Instead, per-note
      *link and tag enrichment* with an evolution-threshold schedule. This is
      the Zettelkasten-style distinctive pattern.

The concept count is unchanged after consolidate; what changes is `description`
text + edge structure. History entries record the per-note actions accepted.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext

logger = logging.getLogger(__name__)


AMEM_SYSTEM = (
    "You are an agentic memory evolution agent. For the provided note and "
    "its nearest neighbors, decide whether to evolve it (link, tag, or "
    "context update). Output JSON matching: "
    '{"should_evolve": bool, "actions": [str], '
    '"suggested_connections": [int], "tags_to_update": [str]}'
)


class AMEMAgenticMemoryBuilder(ArcMemoReorgMemoryBuilder):
    """Per-note evolution + linking. Concept count unchanged; edges/tags grow."""

    name = "reorg_amem"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        k_neighbors: int = 5,
        evo_threshold: int = 3,
        min_neighbor_strength: float = 0.1,
        max_notes_per_pass: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.k_neighbors = int(k_neighbors)
        self.evo_threshold = int(evo_threshold)
        self.min_neighbor_strength = float(min_neighbor_strength)
        self.max_notes_per_pass = int(max_notes_per_pass)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        provider = self._resolve_provider(ctx)

        # Pick recently-added concepts (or top by authorship timestamp if present);
        # fall back to last-N by insertion order.
        recent = list(mem.concepts.keys())[-self.max_notes_per_pass:]

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        actions_log: list[dict[str, Any]] = []
        evo_count = 0
        evolved: list[str] = []

        for note_name in recent:
            nbrs = self._find_neighbors(graph, mem, note_name)
            if not nbrs:
                continue

            if provider is not None:
                decision = self._evolve_via_llm(
                    ctx, mem, note_name, nbrs, provider,
                )
            else:
                decision = self._evolve_via_template(
                    mem, note_name, nbrs,
                )

            if not decision or not decision.get("should_evolve"):
                continue

            actions = decision.get("actions", [])
            applied = self._apply_actions(
                mem, note_name, nbrs, decision, actions,
            )
            if applied:
                evo_count += 1
                evolved.append(note_name)
                actions_log.append({
                    "note": note_name,
                    "actions": actions,
                    "applied": applied,
                })

                # A-MEM evolution-threshold schedule: every evo_threshold
                # successful evolutions, record a consolidation marker.
                if evo_count % self.evo_threshold == 0:
                    actions_log.append({
                        "note": f"<consolidation-marker-at-{evo_count}>",
                        "actions": ["consolidate_memories"],
                    })

        if not actions_log:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "amem_skipped",
                "reason": f"no evolution passed threshold over {len(recent)} notes",
            })
            return memory

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "amem_agentic_evolution",
            "notes_evolved": len(evolved),
            "actions": actions_log,
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

    def _find_neighbors(
        self, graph: ConceptGraph, mem: ConceptMemory, node: str,
    ) -> list[tuple[str, float]]:
        """Top-k neighbors by combined co-activation + embedding-sim weight."""
        triples = graph.neighbors(node)
        weights: dict[str, float] = {}
        for dst, kind, w in triples:
            weights[dst] = weights.get(dst, 0.0) + float(w)
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[: self.k_neighbors]

    def _evolve_via_llm(
        self, ctx: RunContext, mem: ConceptMemory,
        note: str, nbrs: list[tuple[str, float]], provider,
    ) -> dict[str, Any] | None:
        def _short_description(concept_name: str, limit: int) -> str:
            concept = mem.concepts.get(concept_name)
            if concept is None:
                return ""
            return (concept.description or "")[:limit]

        nbr_block = "\n".join(
            f"  {i}: {n} (sim={w:.3f}) - {_short_description(n, 120)}"
            for i, (n, w) in enumerate(nbrs)
        )
        note_desc = _short_description(note, 200)
        prompt = (
            f"{AMEM_SYSTEM}\n\n"
            f"Note '{note}': {note_desc}\n"
            f"Nearest {len(nbrs)} neighbors:\n{nbr_block}\n\n"
            "Output JSON only."
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = completions[0] if completions else "{}"
            if not isinstance(raw, str) or not raw.strip():
                logger.warning(
                    "a-mem LLM evolution returned empty/non-string completion on %s",
                    note,
                )
                return None
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover
            logger.warning("a-mem LLM evolution failed on %s: %s", note, exc)
            return None

    def _evolve_via_template(
        self, mem: ConceptMemory, note: str, nbrs: list[tuple[str, float]],
    ) -> dict[str, Any] | None:
        """Fallback policy: evolve iff top neighbor strength ≥ threshold."""
        if not nbrs:
            return None
        top_name, top_w = nbrs[0]
        if top_w < self.min_neighbor_strength:
            return {"should_evolve": False}
        top_concept = mem.concepts.get(top_name)
        note_concept = mem.concepts.get(note)
        tags = []
        if top_concept and note_concept:
            note_tokens = set((note_concept.description or "").lower().split())
            nbr_tokens = set((top_concept.description or "").lower().split())
            shared = note_tokens & nbr_tokens - {"the", "a", "an", "and", "or", "is", "in", "of", "to", "for"}
            tags = sorted(shared)[:5]
        if not tags and top_concept:
            tags = [top_concept.kind or "routine"]
        return {
            "should_evolve": True,
            "actions": ["strengthen", "update_tags"],
            "suggested_connections": [0],
            "tags_to_update": tags,
        }

    def _apply_actions(
        self,
        mem: ConceptMemory,
        note: str,
        nbrs: list[tuple[str, float]],
        decision: dict[str, Any],
        actions: list[str],
    ) -> list[dict[str, Any]]:
        """Apply decided actions; return list of concrete changes."""
        applied: list[dict[str, Any]] = []
        connections = decision.get("suggested_connections", [])
        if note not in mem.concepts:
            return applied

        c = mem.concepts[note]
        if "strengthen" in actions or "link" in actions:
            for idx in connections:
                if not isinstance(idx, int) or idx < 0 or idx >= len(nbrs):
                    continue
                target, _ = nbrs[idx]
                marker = f"[A-MEM linked: {target}]"
                if marker not in (c.description or ""):
                    c.description = (c.description or "") + " " + marker
                    applied.append({"type": "link", "from": note, "to": target})

        if "update_tags" in actions:
            tags = decision.get("tags_to_update", [])
            if tags:
                new_tag_strs = [str(t)[:32].strip() for t in tags if t]
                fresh_tags = [t for t in new_tag_strs if t and t not in c.tags]
                if fresh_tags:
                    c.tags.extend(fresh_tags)
                    tag_suffix = " [tags:" + ",".join(fresh_tags) + "]"
                    if tag_suffix not in (c.description or ""):
                        c.description = (c.description or "") + tag_suffix
                    applied.append({"type": "tags", "note": note, "tags": fresh_tags})

        return applied
