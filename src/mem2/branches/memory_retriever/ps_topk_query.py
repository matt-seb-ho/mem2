"""Query-conditioned top-k retriever for ArcMemo PS concept memories."""
from __future__ import annotations

import json
import math
import re

from mem2.branches.memory_retriever.ps_topk import _free_text_hint, _parse_override_hint
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)

WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _query_text(problem: ProblemSpec) -> str:
    direct = getattr(problem, "prompt_text", None)
    if direct:
        return str(direct)
    parts: list[str] = [str(getattr(problem, "uid", ""))]
    meta = getattr(problem, "metadata", {}) or {}
    for key in ("description", "instructions", "prompt", "query", "source"):
        if meta.get(key):
            parts.append(str(meta[key]))
    parts.append(json.dumps(problem.train_pairs, sort_keys=True, separators=(",", ":")))
    parts.append(json.dumps(problem.test_pairs, sort_keys=True, separators=(",", ":")))
    return " ".join(parts)


def _token_overlap(query_tokens: set[str], description: str) -> float:
    desc_tokens = _tokenize(description)
    if not query_tokens or not desc_tokens:
        return 0.0
    return len(query_tokens & desc_tokens) / math.sqrt(len(query_tokens) * len(desc_tokens))


class PsTopKQueryRetriever:
    name = "ps_topk_query"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4,
        usage_threshold: int = 0,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        skip_parameters: bool = False,
        skip_parameter_description: bool = True,
    ) -> None:
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.usage_threshold = int(usage_threshold)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.skip_parameters = bool(skip_parameters)
        self.skip_parameter_description = bool(skip_parameter_description)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        mem = ConceptMemory.from_payload(memory.payload)
        if not mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "empty"},
            )

        query = _query_text(problem)
        query_tokens = _tokenize(query)
        max_used = max((len(c.used_in) for c in mem.concepts.values()), default=0) or 1
        ranked = sorted(
            mem.concepts.values(),
            key=lambda c: (
                self.alpha * _token_overlap(query_tokens, c.description or "")
                + self.beta * (len(c.used_in) / max_used),
                len(c.used_in),
                c.name,
            ),
            reverse=True,
        )
        top = [c.name for c in ranked[: max(self.top_k, 0)]]

        variant_flags = (memory.metadata or {}).get("render_flags") if memory.metadata else None
        if variant_flags and isinstance(variant_flags, dict):
            include_description = variant_flags.get("include_description", self.include_description)
            skip_cues = variant_flags.get("skip_cues", self.skip_cues)
            skip_implementation = variant_flags.get("skip_implementation", self.skip_implementation)
            skip_parameters = variant_flags.get("skip_parameters", self.skip_parameters)
            skip_parameter_description = variant_flags.get(
                "skip_parameter_description", self.skip_parameter_description
            )
            skip_kind = variant_flags.get("skip_kind", True)
            skip_routine_subtype = variant_flags.get("skip_routine_subtype", True)
            parse_kind_overrides = variant_flags.get("parse_kind_overrides")
            variant = memory.metadata.get("variant")
        else:
            include_description = self.include_description
            skip_cues = self.skip_cues
            skip_implementation = self.skip_implementation
            skip_parameters = self.skip_parameters
            skip_parameter_description = self.skip_parameter_description
            skip_kind = True
            skip_routine_subtype = True
            parse_kind_overrides = None
            variant = None

        if variant == "free_text":
            hint = _free_text_hint(mem, top)
        elif isinstance(parse_kind_overrides, dict) and parse_kind_overrides:
            hint = _parse_override_hint(
                mem,
                top,
                parse_kind_overrides,
                include_description=include_description,
                skip_kind=skip_kind,
                skip_routine_subtype=skip_routine_subtype,
                skip_cues=skip_cues,
                skip_implementation=skip_implementation,
                skip_parameters=skip_parameters,
                skip_parameter_description=skip_parameter_description,
            )
        else:
            hint = mem.to_string(
                concept_names=top,
                include_description=include_description,
                skip_kind=skip_kind,
                skip_routine_subtype=skip_routine_subtype,
                skip_cues=skip_cues,
                skip_implementation=skip_implementation,
                skip_parameters=skip_parameters,
                skip_parameter_description=skip_parameter_description,
                usage_threshold=self.usage_threshold,
            )

        meta = {
            "retriever": self.name,
            "scoring_mode": "ps_topk_query",
            "top_k": self.top_k,
            "alpha": self.alpha,
            "beta": self.beta,
            "query_token_count": len(query_tokens),
            "num_concepts_scored": len(mem.concepts),
            "num_selected": len(top),
        }
        if variant is not None:
            meta["variant"] = variant
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in top],
            metadata=meta,
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        return self.retrieve(ctx, memory, problem, previous_attempts)
