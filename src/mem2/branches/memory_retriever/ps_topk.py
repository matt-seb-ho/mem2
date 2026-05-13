"""PsTopKRetriever: flat top-k PS retriever for the Phase-1 ablation.

Analogue of ``oe_topk`` for the ``arcmemo_ps`` schema. Needed because
``ps_selector`` in Mode 4 renders ALL concepts (no top_k bound), which blows
the context window with a 270-concept library.

Ranks concepts by descending ``len(used_in)`` (co-activation frequency in the
training set), takes top-k, renders via ``ConceptMemory.to_string``.

Not a novelty — serves as the "flat top-k" baseline for axis B ablation.
"""
from __future__ import annotations

from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


def _free_text_hint(mem: ConceptMemory, concept_names: list[str]) -> str:
    sentences: list[str] = []
    for name in concept_names:
        concept = mem.concepts.get(name)
        if concept is None:
            continue
        desc = (concept.description or "").strip()
        if desc:
            sentence = f"{concept.name} means {desc.rstrip('.')}."
        else:
            sentence = f"{concept.name} may be relevant."
        sentences.append(sentence)
    if not sentences:
        return ""
    return "Recall the following concepts that may be relevant: " + " ".join(sentences)


class PsTopKRetriever:
    name = "ps_topk"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        usage_threshold: int = 0,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        skip_parameters: bool = False,
        skip_parameter_description: bool = True,
    ) -> None:
        self.top_k = int(top_k)
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
        ranked = sorted(
            mem.concepts.values(),
            key=lambda c: (len(c.used_in), c.name),
            reverse=True,
        )
        top = [c.name for c in ranked[: max(self.top_k, 0)]]
        # Honor per-run render_flags stamped by D.3x variant_format or
        # D.4 DSPy optimizer. Falls back to constructor flags otherwise.
        # Mirrors ps_selector._extract_variant_flags pattern.
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
            variant = memory.metadata.get("variant")
        else:
            include_description = self.include_description
            skip_cues = self.skip_cues
            skip_implementation = self.skip_implementation
            skip_parameters = self.skip_parameters
            skip_parameter_description = self.skip_parameter_description
            skip_kind = True
            skip_routine_subtype = True
            variant = None
        if variant == "free_text":
            hint = _free_text_hint(mem, top)
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
            "scoring_mode": "topk",
            "top_k": self.top_k,
            "num_concepts_total": len(mem.concepts),
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
