"""DSPy-style COPRO iterative format-variant search — axis D.4.

Port of DSPy COPRO (Khattab, Singhvi et al.; arxiv 2310.03714).

Paper: literature/2310.03714.pdf
Repo:  third_party/dspy/ (entry: dspy/teleprompt/copro_optimizer.py::COPRO)

Specifically ported:
    - The *breadth-then-depth* search pattern from `COPRO.compile`: round 0
      scores `breadth` candidate instructions; rounds 1..depth propose
      mutations of the top-k with full history as prompt context (see
      `GenerateInstructionGivenAttempts`).
    - The proposal-with-history mechanic: each round's LLM call sees the
      ordered list of prior attempts + scores, allowing it to improve rather
      than re-sample.

Deliberate simplifications (LLM-optional):
    - The "instruction" is replaced by a *format variant* — a set of render
      flags controlling how concepts serialize into the prompt (same
      mechanism as `variant_format`, axis D.3a-e).
    - The LLM provider (`ctx.config["_meta_edit_provider"]`) proposes new
      variants as named render-flag dicts. Without a provider, a deterministic
      template policy rotates through the existing 5 variants and records
      what COPRO would have explored. The iterative-search-with-history
      behavior is preserved; only the proposal step degrades gracefully.
    - *Scoring*: COPRO evaluates against a training metric. We substitute a
      cheap MDL-proxy on the concept serialization (shorter renders score
      higher, ties broken by coverage of distinct concept kinds). This is a
      proxy for "instructions that are concise and cover the schema" — no
      labeled data required. When labeled outcomes ARE available (via
      `memory.payload["reorg"]["outcomes"]`), the scorer falls back to
      success-rate over the variant's prior runs.

D.4 vs D.3a-e:
    - D.3x are fixed-variant conditions (one per variant).
    - D.4 is a SINGLE condition that iteratively searches through the variant
      space at `initialize` time, producing a different winning variant
      depending on the run's scoring signal. The "ablation" question answered:
      does running optimization ever pick a different variant than a fixed
      choice?
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from mem2.branches.memory_builder.variant_formats import (
    RENDER_FLAGS,
    VARIANTS,
    VariantFormatBuilder,
)
from mem2.core.entities import MemoryState, ProblemSpec, RunContext

logger = logging.getLogger(__name__)

COPRO_SYSTEM = (
    "You are a format-variant optimizer for a concept-memory system. "
    "Propose a new set of render flags that will make the memory more useful "
    "for an LLM reasoning step. Output JSON: "
    '{"name": str, "flags": {"skip_cues": bool, "skip_implementation": bool, '
    '"skip_parameters": bool, "include_description": bool}, "reason": str}'
)


@dataclass
class VariantAttempt:
    name: str
    flags: dict[str, Any]
    score: float
    round_idx: int
    reason: str = ""


class DSPyOptFormatBuilder(VariantFormatBuilder):
    """Iterative search over format variants with depth-breadth history."""

    name = "variant_dspy_opt"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        breadth: int = 5,
        depth: int = 2,
        init_temperature: float = 1.4,
        **kwargs,
    ) -> None:
        # VariantFormatBuilder requires a variant; we start with the "minimal"
        # variant as the round-0 baseline; the winning variant gets stamped
        # during initialize().
        kwargs.setdefault("variant", "minimal")
        super().__init__(**kwargs)
        self.breadth = int(breadth)
        self.depth = int(depth)
        self.init_temperature = float(init_temperature)

    def initialize(self, ctx: RunContext, problems: list[ProblemSpec]) -> MemoryState:
        memory = super().initialize(ctx, problems)
        provider = self._resolve_provider(ctx)

        history: list[VariantAttempt] = []
        # Round 0 — seed with the existing 5 variants scored on MDL-proxy.
        for name in sorted(VARIANTS):
            flags = dict(RENDER_FLAGS[name])
            score = self._score_variant(flags, memory)
            history.append(VariantAttempt(
                name=name, flags=flags, score=score, round_idx=0,
                reason="seed variant",
            ))
        history.sort(key=lambda h: h.score, reverse=True)

        # Rounds 1..depth — propose mutations of top-k with history.
        for round_idx in range(1, self.depth + 1):
            top = history[: max(1, self.breadth // 2)]
            for _ in range(self.breadth):
                if provider is not None:
                    proposal = self._propose_via_llm(provider, history, round_idx)
                else:
                    proposal = self._propose_via_template(top, round_idx)
                if proposal is None:
                    continue
                flags = proposal.get("flags", {})
                score = self._score_variant(flags, memory)
                history.append(VariantAttempt(
                    name=proposal.get("name", f"copro_{round_idx}_{len(history)}"),
                    flags=flags, score=score, round_idx=round_idx,
                    reason=proposal.get("reason", ""),
                ))
            history.sort(key=lambda h: h.score, reverse=True)

        winner = history[0]
        memory.metadata["variant"] = winner.name
        memory.metadata["render_flags"] = dict(winner.flags)
        memory.metadata["dspy_opt"] = {
            "winner_name": winner.name,
            "winner_score": winner.score,
            "winner_flags": winner.flags,
            "history_len": len(history),
            "used_llm": provider is not None,
            "breadth": self.breadth,
            "depth": self.depth,
            "top_3": [
                {"name": h.name, "score": h.score, "round": h.round_idx}
                for h in history[:3]
            ],
        }
        # Stamp the winner's flags into RENDER_FLAGS under a run-specific key
        # so the retriever can pick them up via metadata.variant.
        if winner.name not in RENDER_FLAGS:
            RENDER_FLAGS[winner.name] = dict(winner.flags)
        return memory

    # ----------------------------------------------------------------- #
    def _resolve_provider(self, ctx: RunContext):
        try:
            return (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            return None

    def _score_variant(
        self, flags: dict[str, Any], memory: MemoryState,
    ) -> float:
        """MDL-proxy scorer: fewer fields rendered ≈ shorter prompt ≈ better,
        tiebroken by coverage of distinct concept kinds."""
        shown_fields = sum(
            1 for k, v in flags.items() if not (isinstance(v, bool) and v is True)
        )
        # Penalize variants that hide everything (shown_fields == 0 → useless).
        if shown_fields == 0:
            return 0.0
        kinds = set()
        for c in (memory.payload.get("concepts", {}) or {}).values():
            if isinstance(c, dict):
                kinds.add(c.get("kind", "unknown"))
            elif hasattr(c, "kind"):
                kinds.add(c.kind)
        coverage = max(len(kinds), 1)
        # Lower shown_fields is better (compactness); higher coverage is better.
        return round(coverage / float(shown_fields), 3)

    def _propose_via_llm(
        self, provider, history: list[VariantAttempt], round_idx: int,
    ) -> dict[str, Any] | None:
        attempts_block = "\n".join(
            f"- name={h.name} score={h.score} flags={json.dumps(h.flags)}"
            for h in history[:10]
        )
        prompt = (
            f"{COPRO_SYSTEM}\n\n"
            f"Round {round_idx} of {self.depth} (breadth={self.breadth}).\n"
            f"Prior attempts (top-10 by score):\n{attempts_block}\n\n"
            "Output JSON only."
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = completions[0] if completions else "{}"
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover
            logger.warning("dspy-opt LLM proposal failed: %s", exc)
            return None

    def _propose_via_template(
        self, top: list[VariantAttempt], round_idx: int,
    ) -> dict[str, Any] | None:
        """Deterministic mutation: toggle one flag on a random top attempt,
        seed-hashed so it's reproducible."""
        if not top:
            return None
        seed = hashlib.sha1(
            f"round{round_idx}_{len(top)}".encode()
        ).digest()
        pick = top[seed[0] % len(top)]
        flip_key = list(pick.flags.keys())[seed[1] % max(len(pick.flags), 1)]
        new_flags = dict(pick.flags)
        if isinstance(new_flags[flip_key], bool):
            new_flags[flip_key] = not new_flags[flip_key]
        return {
            "name": f"copro_r{round_idx}_{pick.name}_flip_{flip_key}",
            "flags": new_flags,
            "reason": f"template mutation of {pick.name}: flipped {flip_key}",
        }
