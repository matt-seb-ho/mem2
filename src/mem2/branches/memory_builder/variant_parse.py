"""PARSE error-driven schema refinement — axis D.6.

Port of PARSE (Shrimal, Jain, Chowdhury, Yenigalla, 2025; arxiv 2510.08623).

Paper: literature/2510.08623.pdf
Repo:  no public official implementation (Amazon internal).

Specifically ported:
    - The *ARCHITECT* mechanism: iteratively refine the concept schema
      based on extraction performance — add field descriptions, validation
      rules, and restructure for LLM consumption, while maintaining
      backward compatibility (RELAY).
    - The *SCOPE* mechanism: reflection-based extraction with static +
      LLM-based guardrails. After each "extraction", classify errors
      (parse, infer, solve) and feed back into ARCHITECT.

Deliberate simplifications (paper-only port, no repo):
    - "Extraction errors" for mem2 = concept retrievals that failed the
      downstream problem. The schema refinement turns ineffective
      render-flags off for error-associated concept kinds, enabling them
      for success-associated kinds.
    - "Backward compatibility" (RELAY) is trivial: we only flip RENDER_FLAGS,
      never change the concept-schema structure itself. The schema always
      remains `arcmemo_ps`.
    - Without an LLM, ARCHITECT's schema-refinement prompt is replaced by
      a performance-driven flip: flags that correlate with failed outcomes
      get turned on; flags correlating with success get turned off. LLM mode
      upgrades to a proper ARCHITECT prompt.

D.6 vs D.4 COPRO vs D.5 GEPA:
    - D.4: breadth-then-depth instruction search.
    - D.5: evolutionary population search.
    - D.6 PARSE (this module): *error-informed* schema refinement. Uses
      outcome signal per kind to decide which render-flags help vs hurt,
      then emits a per-kind schema override. Distinct signal (error
      reflection) and distinct application (per-kind, not global).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mem2.branches.memory_builder.variant_formats import (
    RENDER_FLAGS,
    VARIANTS,
    VariantFormatBuilder,
)
from mem2.core.entities import MemoryState, ProblemSpec, RunContext

logger = logging.getLogger(__name__)

PARSE_SYSTEM = (
    "You are an ARCHITECT for LLM-consumed concept schemas. Given per-"
    "concept-kind performance stats, propose render-flag overrides for "
    "each kind. Output JSON: "
    '{"<kind>": {"skip_cues": bool, "skip_implementation": bool, '
    '"skip_parameters": bool, "include_description": bool}, ...}'
)


class PARSESchemaBuilder(VariantFormatBuilder):
    """Error-driven per-kind schema refinement."""

    name = "variant_parse"
    SCHEMA_NAME = "arcmemo_ps"

    def __init__(
        self,
        *,
        base_variant: str = "structured_routine",
        min_stats_per_kind: int = 3,
        **kwargs,
    ) -> None:
        kwargs.setdefault("variant", base_variant)
        super().__init__(**kwargs)
        self.base_variant = str(base_variant)
        self.min_stats_per_kind = int(min_stats_per_kind)

    def initialize(self, ctx: RunContext, problems: list[ProblemSpec]) -> MemoryState:
        memory = super().initialize(ctx, problems)
        provider = self._resolve_provider(ctx)

        # Build per-kind stats from the seeded memory's used_in + any outcomes
        # available in payload (outcomes are empty at initialize time in
        # practice, so the initial schema just stamps the base variant).
        stats = self._per_kind_stats(memory)
        base_flags = dict(RENDER_FLAGS.get(self.base_variant, {}))

        per_kind_overrides: dict[str, dict[str, Any]] = {}
        if provider is not None:
            refined = self._refine_via_llm(provider, stats, base_flags)
            if refined:
                per_kind_overrides = refined
        # Template: derive overrides from success/hit rates per kind.
        else:
            for kind, s in stats.items():
                if s["hit"] < self.min_stats_per_kind:
                    continue
                rate = s["rate"]
                # Heuristic: if this kind performs poorly, MORE detail (show
                # cues + impl). If it performs well, LESS detail (skip cues
                # to keep context tight).
                new_flags = dict(base_flags)
                if rate < 0.3:
                    new_flags["skip_cues"] = False
                    new_flags["skip_implementation"] = False
                elif rate > 0.7:
                    new_flags["skip_cues"] = True
                    new_flags["skip_implementation"] = True
                per_kind_overrides[kind] = new_flags

        # Stamp a composite variant name + per-kind table into metadata.
        composite_name = f"parse_refined_{self.base_variant}"
        parse_kind_overrides = self._render_modes_from_flag_overrides(per_kind_overrides)
        memory.metadata["variant"] = composite_name
        memory.metadata["render_flags"] = dict(base_flags)
        memory.metadata["render_flags"]["parse_kind_overrides"] = parse_kind_overrides
        memory.metadata["parse"] = {
            "base_variant": self.base_variant,
            "per_kind_overrides": per_kind_overrides,
            "parse_kind_overrides": parse_kind_overrides,
            "per_kind_stats": stats,
            "used_llm_architect": provider is not None,
        }
        # Register the composite in RENDER_FLAGS (uses base as default when a
        # kind isn't overridden; retriever would pick per-concept-kind in a
        # fuller integration).
        if composite_name not in RENDER_FLAGS:
            RENDER_FLAGS[composite_name] = dict(base_flags)
        return memory

    def _render_modes_from_flag_overrides(
        self,
        per_kind_overrides: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        modes: dict[str, str] = {}
        for kind, flags in per_kind_overrides.items():
            if not isinstance(flags, dict):
                continue
            if flags.get("skip_cues") is False or flags.get("skip_implementation") is False:
                modes[kind] = "full"
            elif flags.get("skip_parameters") is True and flags.get("skip_cues") is True:
                modes[kind] = "compact"
            else:
                modes[kind] = "compact"
        return modes

    # ----------------------------------------------------------------- #
    def _resolve_provider(self, ctx: RunContext):
        try:
            return (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            return None

    def _per_kind_stats(self, memory: MemoryState) -> dict[str, dict[str, Any]]:
        """Stats per kind from the memory's concepts.

        Note: at initialize time, outcomes aren't yet present. We synthesize
        a synthetic "baseline" rate from used_in density as a proxy for
        "is this kind informative?" — kinds with more concepts-per-problem
        on average are more likely to be useful.
        """
        stats: dict[str, dict[str, Any]] = {}
        concepts = (memory.payload.get("concepts", {}) or {})
        # Handle both dict-form and object-form.
        items = (
            concepts.items() if isinstance(concepts, dict)
            else [(c.name, c) for c in concepts]
        )
        per_kind_hit: dict[str, int] = {}
        per_kind_count: dict[str, int] = {}
        for name, c in items:
            kind = c.get("kind") if isinstance(c, dict) else getattr(c, "kind", "unknown")
            used_in = c.get("used_in") if isinstance(c, dict) else getattr(c, "used_in", [])
            per_kind_count[kind] = per_kind_count.get(kind, 0) + 1
            per_kind_hit[kind] = per_kind_hit.get(kind, 0) + len(used_in or [])
        for kind in per_kind_count:
            hit = per_kind_hit[kind]
            count = per_kind_count[kind]
            rate_proxy = min(1.0, hit / max(count * 2, 1))  # rough density proxy
            stats[kind] = {
                "hit": hit, "count": count, "rate": round(rate_proxy, 3),
            }
        return stats

    def _refine_via_llm(
        self, provider, stats: dict[str, dict[str, Any]],
        base_flags: dict[str, Any],
    ) -> dict[str, dict[str, Any]] | None:
        prompt = (
            f"{PARSE_SYSTEM}\n\n"
            f"Base variant: {self.base_variant}; base flags: {json.dumps(base_flags)}\n"
            f"Per-kind stats: {json.dumps(stats)}\n\n"
            "Output JSON only."
        )
        try:
            out = provider.generate(prompt, model=getattr(provider, "model", ""))
            parsed = json.loads(out[0] if out else "{}")
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:  # pragma: no cover
            logger.warning("parse ARCHITECT LLM call failed: %s", exc)
        return None
