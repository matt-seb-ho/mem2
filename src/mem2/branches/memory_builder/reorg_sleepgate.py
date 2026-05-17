"""SleepGate temporal-supersession forgetting — axis A.10.

Port of SleepGate (Xie, 2026; arxiv 2603.14517).

Paper: literature/2603.14517.pdf
Repo:  no public official implementation.

SleepGate is an architecture-level intervention on the KV cache of a
transformer — a neural forgetting gate, a conflict-aware temporal tagger,
and a consolidation module for slow-wave-sleep-inspired replay. This does
NOT literally map to mem2's declarative concept memory.

What IS portable — the *temporal supersession* conceptual pattern:
    - When a newer entry SUPERSEDES an older one (same key, different value),
      the older one is a "stale association" that should decay.
    - In a concept memory, "newer supersedes older" maps to:
      ``authorship_lineage`` edges from older → newer (aggregate relations),
      combined with semantic conflict (newer concept's description overlaps
      with older's AND has higher `used_in` count).
    - The "forgetting gate" then evicts the stale concept and records a
      consolidation event into the next-round memory.

Specifically ported (adapted):
    - Conflict-aware temporal tagger → authorship_lineage edge detection
      + token-overlap conflict scoring.
    - Forgetting gate → evict the SUPERSEDED concept (the older one).
    - Consolidation module → the newer concept's description absorbs a
      1-line "supersedes: <old_name>" annotation.

Deliberate simplifications:
    - No neural gate (paper uses a learned feed-forward network). Our gate
      is rule-based: evict iff `older.used_in ⊂ newer.used_in` AND token
      overlap ≥ threshold. This mirrors the paper's "active forgetting"
      constraint that only clearly-superseded entries should go.
    - Paper's O(log n) interference horizon is not directly testable without
      a PI-LLM benchmark; we report the number of superseded entries as a
      proxy for "interference reduced."

A.10 vs A.6 / A.7 / A.8:
    - A.6 A-MEM: per-note linking (no eviction).
    - A.7 Memp: prunes by FAILURE track record (score-based).
    - A.8 EvolveR: dedups by REDUNDANCY (Jaccard + quality tiebreak).
    - A.10 SleepGate (this module): evicts by TEMPORAL SUPERSESSION —
      explicit older→newer relation + coverage inclusion. Evicts even
      successful concepts if they've been semantically superseded.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"\w+")


def _toks(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t.lower() for t in WORD_RE.findall(s)}


class SleepGateForgettingBuilder(ArcMemoReorgMemoryBuilder):
    """Temporal-supersession forgetting gate."""

    name = "reorg_sleepgate"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        token_overlap_threshold: float = 0.4,
        require_coverage_inclusion: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.token_overlap_threshold = float(token_overlap_threshold)
        self.require_coverage_inclusion = bool(require_coverage_inclusion)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        if getattr(self, "_frozen", False):
            return memory
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        if not mem.concepts:
            return memory

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)

        # Precompute token sets.
        sig_cache: dict[str, set[str]] = {
            name: _toks(c.name) | _toks(c.description) | _toks(
                " ".join(c.cues or [])
            )
            for name, c in mem.concepts.items()
        }

        # Detect supersession candidates: for each authorship_lineage edge
        # src → dst, check if dst "supersedes" src.
        superseded: list[dict[str, Any]] = []
        to_evict: set[str] = set()

        for src_name in list(mem.concepts.keys()):
            if src_name in to_evict:
                continue
            for dst_name, kind, _w in graph.neighbors(
                src_name, kinds=["authorship_lineage"],
            ):
                if dst_name == src_name or dst_name in to_evict:
                    continue
                # Conflict tagger: semantic overlap.
                sig_src = sig_cache.get(src_name, set())
                sig_dst = sig_cache.get(dst_name, set())
                if not sig_src or not sig_dst:
                    continue
                overlap = len(sig_src & sig_dst) / max(len(sig_src | sig_dst), 1)
                if overlap < self.token_overlap_threshold:
                    continue
                # Coverage inclusion check.
                if self.require_coverage_inclusion:
                    src_probs = set(getattr(mem.concepts[src_name], "used_in", []) or [])
                    dst_probs = set(getattr(mem.concepts[dst_name], "used_in", []) or [])
                    if src_probs and not src_probs.issubset(dst_probs):
                        continue
                # Gate fires → evict src.
                to_evict.add(src_name)
                superseded.append({
                    "stale": src_name,
                    "successor": dst_name,
                    "overlap": round(overlap, 3),
                })
                # Consolidation module: append a supersedes marker to dst.
                dst = mem.concepts[dst_name]
                marker = f" [supersedes: {src_name}]"
                if marker not in (dst.description or ""):
                    dst.description = (dst.description or "") + marker
                break  # Only one supersession event per src this round.

        if not to_evict:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "sleepgate_skipped",
                "reason": "no authorship_lineage supersession candidates found",
                "n_lineage_edges_checked": sum(
                    1 for e in graph.edges() if e.kind == "authorship_lineage"
                ),
            })
            return memory

        # Apply eviction.
        for name in to_evict:
            c = mem.concepts.get(name)
            if c is None:
                continue
            kind = c.kind
            del mem.concepts[name]
            if name in mem.categories.get(kind, []):
                mem.categories[kind].remove(name)

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "sleepgate_temporal_supersession",
            "evicted_count": len(to_evict),
            "superseded": superseded,
            "token_overlap_threshold": self.token_overlap_threshold,
            "require_coverage_inclusion": self.require_coverage_inclusion,
        })
        new_payload["reorg"] = reorg
        memory.payload = new_payload
        return memory
