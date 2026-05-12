"""EvolveR semantic-dedup of scored principles — axis A.8.

Port of EvolveR (Wu, Wang, Mei et al., 2025; arxiv 2510.16079).

Paper: literature/2510.16079.pdf
Repo:  https://github.com/Edaizi/EvolveR (official) — not locally mirrored.

Specifically ported:
    - The **semantic-deduplication-of-scored-principles** step from
      EvolveR's maintenance pipeline (§3.2): when a new strategic principle
      is distilled, compare its semantic signature against every existing
      principle; if they duplicate, keep the one with the HIGHER historical
      metric score and drop the other.
    - The **dynamic metric score**: tracks each principle's historical
      effectiveness via hit/success ratios (like Memp), but uses the score
      for TIEBREAKING in dedup, not for pruning decisions.

Deliberate simplifications (paper-only port, no repo locally):
    - Principles = concepts. "Semantic signature" is replaced by token-set
      overlap on (name + description + cues). Jaccard ≥ threshold → dedup.
    - No LLM-based principle distillation — we dedup existing concepts
      in-place. With a provider wired, dedup can be verified by asking the
      LLM "are these two principles duplicates?" before removal.
    - Offline/online lifecycle collapses to "every consolidate cycle".

A.8 vs A.6 A-MEM vs A.7 Memp:
    - A.6 A-MEM: per-note LINKING of similar concepts; nothing removed.
    - A.7 Memp: PRUNE under-performing concepts regardless of duplication.
    - A.8 EvolveR (this module): REMOVE duplicates, keep the one with
      higher quality score. Concept count drops only when duplicates exist.
    - The axis-A ablation cleanly isolates: (i) linking vs removal
      (A.6 vs A.8), and (ii) performance-only vs duplication-aware pruning
      (A.7 vs A.8).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"\w+")


def _token_set(*parts: str | None) -> set[str]:
    tokens: set[str] = set()
    for p in parts:
        if p:
            tokens.update(tok.lower() for tok in WORD_RE.findall(p))
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class EvolveRDedupBuilder(ArcMemoReorgMemoryBuilder):
    """Semantic-deduplication of scored principles."""

    name = "reorg_evolver"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        jaccard_threshold: float = 0.6,
        min_principles_for_dedup: int = 10,
        require_llm_verify: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.jaccard_threshold = float(jaccard_threshold)
        self.min_principles_for_dedup = int(min_principles_for_dedup)
        self.require_llm_verify = bool(require_llm_verify)

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        reorg = memory.payload.get("reorg")
        if not reorg or not self._should_reorg(reorg):
            return memory

        mem = ConceptMemory.from_payload(memory.payload)
        if len(mem.concepts) < self.min_principles_for_dedup:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "evolver_skipped",
                "reason": f"only {len(mem.concepts)} principles (< min {self.min_principles_for_dedup})",
            })
            return memory

        provider = self._resolve_provider(ctx)
        outcomes_by_pid = self._build_outcomes_map(reorg)

        # Score each principle by (historical metric ≈ success_rate, hit_count).
        scores: dict[str, tuple[float, int]] = {}
        for name, c in mem.concepts.items():
            used_in = list(getattr(c, "used_in", []) or [])
            hit = len(used_in)
            success = sum(
                1 for pid in used_in
                if outcomes_by_pid.get(str(pid), 0.0) > 0.0
            )
            rate = success / max(hit, 1) if hit > 0 else 0.0
            scores[name] = (rate, hit)

        # Build token signatures once.
        signatures: dict[str, set[str]] = {
            name: _token_set(
                c.name, c.description,
                " ".join(c.cues or []),
            )
            for name, c in mem.concepts.items()
        }

        # Find duplicate pairs (O(n²) with early cutoffs).
        removed: list[dict[str, Any]] = []
        to_remove: set[str] = set()
        names_sorted = list(mem.concepts.keys())
        for i, a in enumerate(names_sorted):
            if a in to_remove:
                continue
            sig_a = signatures[a]
            for b in names_sorted[i + 1:]:
                if b in to_remove:
                    continue
                sim = _jaccard(sig_a, signatures[b])
                if sim < self.jaccard_threshold:
                    continue
                # Optional LLM verification (paper: dedup guided by LLM judgment).
                if self.require_llm_verify and provider is not None:
                    verified = self._verify_duplicate_via_llm(
                        provider, mem.concepts[a], mem.concepts[b],
                    )
                    if not verified:
                        continue
                # Decide which to drop: keep the higher-scored one.
                score_a = scores[a]
                score_b = scores[b]
                loser = b if score_a >= score_b else a
                winner = a if loser == b else b
                to_remove.add(loser)
                removed.append({
                    "loser": loser,
                    "winner": winner,
                    "similarity": round(sim, 3),
                    "loser_score": list(scores[loser]),
                    "winner_score": list(scores[winner]),
                })
                if loser == a:
                    break  # a is gone, skip rest of its pairs

        for name in to_remove:
            c = mem.concepts.get(name)
            if c is None:
                continue
            kind = c.kind
            del mem.concepts[name]
            if name in mem.categories.get(kind, []):
                mem.categories[kind].remove(name)

        if not removed:
            reorg.setdefault("history", []).append({
                "step": reorg.get("step", 0),
                "action": "evolver_skipped",
                "reason": f"no principle pairs met jaccard threshold ({self.jaccard_threshold})",
                "n_principles_checked": len(names_sorted),
            })
            return memory

        new_payload = mem.to_payload()
        reorg.setdefault("history", []).append({
            "step": reorg.get("step", 0),
            "action": "evolver_semantic_dedup",
            "removed_count": len(removed),
            "removed": removed[:20],  # cap for log size
            "used_llm_verify": self.require_llm_verify and provider is not None,
            "jaccard_threshold": self.jaccard_threshold,
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

    def _verify_duplicate_via_llm(self, provider, c_a, c_b) -> bool:
        prompt = (
            "Are the following two strategic principles duplicates? "
            'Respond with exactly "YES" or "NO".\n\n'
            f"Principle A: {c_a.name} — {(c_a.description or '')[:240]}\n"
            f"Principle B: {c_b.name} — {(c_b.description or '')[:240]}\n"
        )
        try:
            out = provider.generate(prompt, model=getattr(provider, "model", ""))
            if out and isinstance(out[0], str):
                return "YES" in out[0].upper()
        except Exception as exc:  # pragma: no cover
            logger.warning("evolver LLM dedup verify failed: %s", exc)
        return False
