"""PsSelectorRetriever: PS (Program Synthesis) concept selection retriever.

Selection modes (priority order):
1. ``selected_concepts_file`` — precomputed concept names, rendered at runtime
   through the filter → route → render pipeline. Preferred for experiments.
2. ``prompt_info_file`` — precomputed rendered hint text, returned directly.
   Legacy mode; bypasses the pipeline. Kept for backward compat / baselines.
3. Inline LLM selection — per-problem LLM call at runtime (legacy).
4. All concepts — no selection, returns everything (fallback).

Internal pipeline: select → filter → route → render

Filtering and routing are delegated to format-independent stages
(``ConceptFilter``, ``RetrievalRouter`` from ``mem2.retrieval``) that can be
reused by any retriever regardless of memory format.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from mem2.concepts.domain import DomainProfile
from mem2.concepts.memory import ConceptMemory
from mem2.concepts.prompts import DOMAIN_PROMPT_MAP
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)
from mem2.prompting.render import format_problem_for_prompt
from mem2.retrieval.filters import ConceptFilter
from mem2.retrieval.routers import RetrievalRouter

logger = logging.getLogger(__name__)

_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

_RENDER_PROFILES = {
    "full": dict(
        skip_cues=False, skip_implementation=False, skip_parameters=False,
        skip_parameter_description=True, include_description=True,
    ),
    "cues_only": dict(
        skip_cues=False, skip_implementation=True, skip_parameters=True,
        skip_parameter_description=True, include_description=True,
    ),
    "name_only": dict(
        skip_cues=True, skip_implementation=True, skip_parameters=True,
        skip_parameter_description=True, include_description=True,
    ),
}


def _free_text_hint(concept_mem: ConceptMemory, selected_names: list[str] | None) -> str:
    names = selected_names or list(concept_mem.concepts.keys())
    sentences: list[str] = []
    for name in names:
        concept = concept_mem.concepts.get(name)
        if concept is None:
            continue
        desc = (concept.description or "").strip()
        if desc:
            sentences.append(f"{concept.name} means {desc.rstrip('.')}.")
        else:
            sentences.append(f"{concept.name} may be relevant.")
    if not sentences:
        return ""
    return "Recall the following concepts that may be relevant: " + " ".join(sentences)


def _parse_override_hint(
    concept_mem: ConceptMemory,
    selected_names: list[str] | None,
    parse_kind_overrides: dict,
    render_flags: dict[str, Any],
) -> str:
    names = selected_names or list(concept_mem.concepts.keys())
    blocks: list[str] = []
    for name in names:
        concept = concept_mem.concepts.get(name)
        if concept is None:
            continue
        mode = str(parse_kind_overrides.get(concept.kind, "")).strip()
        if mode == "skip":
            continue
        if mode == "compact":
            blocks.append(concept.to_string(
                include_description=True,
                skip_kind=True,
                skip_routine_subtype=True,
                skip_cues=True,
                skip_implementation=True,
                skip_parameters=True,
                skip_parameter_description=True,
            ))
        elif mode == "full":
            blocks.append(concept.to_string(
                include_description=True,
                skip_kind=False,
                skip_routine_subtype=False,
                skip_cues=False,
                skip_implementation=False,
                skip_parameters=False,
                skip_parameter_description=False,
            ))
        else:
            blocks.append(concept.to_string(
                include_description=render_flags["include_description"],
                skip_kind=render_flags["skip_kind"],
                skip_routine_subtype=render_flags.get("skip_routine_subtype", True),
                skip_cues=render_flags["skip_cues"],
                skip_implementation=render_flags["skip_implementation"],
                skip_parameters=render_flags["skip_parameters"],
                skip_parameter_description=render_flags["skip_parameter_description"],
            ))
    return "\n".join(blocks)


class PsSelectorRetriever:
    """PS (Program Synthesis) concept selection retriever.

    Composes format-independent stages (ConceptFilter, RetrievalRouter) with
    format-specific stages (ConceptMemory deserialization, rendering).

    Internal pipeline: select → filter → route → render
    """

    name = "ps_selector"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        domain: str = "arc",
        use_llm_selector: bool = True,
        selector_model: str = "",
        selector_gen_cfg: dict[str, Any] | None = None,
        hint_template_key: str = "op3",
        prompt_info_file: str = "",
        selected_concepts_file: str = "",
        render_mode: str = "full",
        selector_render_mode: str = "full",
        max_frequency: float = 0.0,
        max_concepts_per_problem: int = 0,
        routing_strategy: str = "none",
        routing_max_hint_chars: int = 0,
        routing_max_concept_count: int = 0,
        routing_max_pre_filter_count: int = 0,
        concept_frequency_file: str = "",
    ):
        self.top_k = int(top_k)
        self.domain = domain
        self.use_llm_selector = bool(use_llm_selector)
        self.selector_model = str(selector_model or "")
        self.selector_gen_cfg = dict(
            selector_gen_cfg or {"n": 1, "temperature": 0.0, "max_tokens": 1024}
        )
        self.hint_template_key = hint_template_key
        self.render_mode = render_mode
        self.selector_render_mode = selector_render_mode

        # ── Format-independent stages (reusable by any retriever) ─────
        self._filter = ConceptFilter(
            max_frequency=float(max_frequency),
            max_concepts=int(max_concepts_per_problem),
            frequency_file=concept_frequency_file,
        )
        self._router = RetrievalRouter(
            strategy=routing_strategy,
            frequency_threshold=(
                float(max_frequency) if float(max_frequency) > 0.0 else 0.5
            ),
            max_hint_chars=int(routing_max_hint_chars),
            max_concept_count=int(routing_max_concept_count),
            max_pre_filter_count=int(routing_max_pre_filter_count),
            frequencies=self._filter.frequencies,
        )

        # ── Precomputed concept names (preferred — goes through pipeline) ─
        self._selected_concepts: dict[str, list[str]] | None = None
        if selected_concepts_file:
            path = Path(selected_concepts_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                self._selected_concepts = json.loads(path.read_text())
                logger.info(
                    f"Loaded pre-computed concept names for "
                    f"{len(self._selected_concepts)} problems from {path}"
                )
            else:
                logger.warning(f"selected_concepts_file not found: {path}")

        # ── Precomputed rendered hints (legacy — bypasses pipeline) ───
        self._prompt_info: dict[str, dict] | None = None
        if prompt_info_file:
            path = Path(prompt_info_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                self._prompt_info = json.loads(path.read_text())
                logger.info(
                    f"Loaded pre-computed hints for {len(self._prompt_info)} problems "
                    f"from {path}"
                )
            else:
                logger.warning(f"prompt_info_file not found: {path}")

    # ------------------------------------------------------------------ #
    #  Helpers (format-specific — know about ConceptMemory)                #
    # ------------------------------------------------------------------ #
    def _reconstruct_memory(self, memory: MemoryState) -> ConceptMemory:
        return ConceptMemory.from_payload(memory.payload)

    def _get_prompt_templates(self):
        return DOMAIN_PROMPT_MAP.get(self.domain, DOMAIN_PROMPT_MAP["arc"])

    def _format_problem_for_selection(self, problem: ProblemSpec) -> str:
        if self.domain == "code":
            return problem.metadata.get("question_content", problem.metadata.get("problem_text", str(problem.metadata)))
        if self.domain == "math":
            return problem.metadata.get("problem_text", str(problem.metadata))
        return format_problem_for_prompt(problem)

    def _build_profile(self, concept_mem: ConceptMemory) -> DomainProfile | None:
        if self.domain == "arc":
            return None
        kinds = sorted(concept_mem.categories.keys())
        if not kinds:
            return None
        return DomainProfile(
            valid_kinds=set(kinds),
            section_order=kinds,
            section_headers={k: f"## {k}" for k in kinds},
        )

    @staticmethod
    def _extract_variant_flags(memory: MemoryState) -> dict[str, Any] | None:
        """Read render_flags stamped by D.3x variant_format or D.4 DSPy optimizer."""
        if not memory.metadata:
            return None
        flags = memory.metadata.get("render_flags")
        if flags and isinstance(flags, dict):
            out = dict(flags)
            if memory.metadata.get("variant"):
                out["_variant"] = memory.metadata.get("variant")
            return out
        return None

    def _render_hint_text(
        self, concept_mem: ConceptMemory, selected_names: list[str] | None,
        *, variant_flags: dict[str, Any] | None = None,
    ) -> str:
        """Return raw rendered concept text (no hint-template wrapping).

        The inference engine is responsible for wrapping with its own hint
        template at prompt-build time — matching arc_memo's pattern.
        """
        profile = self._build_profile(concept_mem)
        if variant_flags is not None and variant_flags.get("_variant") == "free_text":
            return _free_text_hint(concept_mem, selected_names)
        if variant_flags is not None:
            render_flags = {
                "skip_cues": variant_flags.get("skip_cues", False),
                "skip_implementation": variant_flags.get("skip_implementation", False),
                "skip_parameters": variant_flags.get("skip_parameters", False),
                "skip_parameter_description": variant_flags.get("skip_parameter_description", True),
                "include_description": variant_flags.get("include_description", True),
                "skip_kind": variant_flags.get("skip_kind", True),
                "skip_routine_subtype": variant_flags.get("skip_routine_subtype", True),
            }
        else:
            render_flags = dict(_RENDER_PROFILES.get(self.render_mode, _RENDER_PROFILES["full"]))
            render_flags.setdefault("skip_kind", True)
            render_flags.setdefault("skip_routine_subtype", True)

        parse_kind_overrides = (
            variant_flags.get("parse_kind_overrides")
            if isinstance(variant_flags, dict)
            else None
        )
        if isinstance(parse_kind_overrides, dict) and parse_kind_overrides:
            return _parse_override_hint(
                concept_mem,
                selected_names,
                parse_kind_overrides,
                render_flags,
            )

        if selected_names:
            return concept_mem.to_string(
                concept_names=selected_names,
                skip_parameter_description=render_flags["skip_parameter_description"],
                skip_cues=render_flags["skip_cues"],
                skip_implementation=render_flags["skip_implementation"],
                skip_parameters=render_flags["skip_parameters"],
                skip_kind=render_flags["skip_kind"],
                include_description=render_flags["include_description"],
                usage_threshold=0,
                show_other_concepts=True,
                profile=profile,
            )
        return concept_mem.to_string(
            skip_parameter_description=render_flags["skip_parameter_description"],
            skip_cues=render_flags["skip_cues"],
            skip_implementation=render_flags["skip_implementation"],
            skip_parameters=render_flags["skip_parameters"],
            skip_kind=render_flags["skip_kind"],
            include_description=render_flags["include_description"],
            usage_threshold=0,
            profile=profile,
        )

    def _parse_concept_selection(
        self, completion: str, valid_names: set[str]
    ) -> tuple[list[str], str | None]:
        if not completion.strip():
            return [], "empty_completion"
        m = _YAML_BLOCK_RE.search(completion)
        yaml_text = m.group(1) if m else None
        if yaml_text is None:
            return [], "no_yaml_block"
        try:
            parsed = yaml.safe_load(yaml_text)
        except Exception as exc:
            return [], f"yaml_parse_error: {exc}"
        if not isinstance(parsed, list):
            return [], f"unsupported_yaml_type: {type(parsed).__name__}"

        selected: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict) and len(item) == 1:
                k, v = next(iter(item.items()))
                name = f"{v}".strip() if isinstance(v, str) else f"{k}".strip()
            else:
                continue
            if name in valid_names and name not in selected:
                selected.append(name)

        if not selected:
            return [], "no_valid_names"
        return selected, None

    # ------------------------------------------------------------------ #
    #  Precomputed paths                                                   #
    # ------------------------------------------------------------------ #
    def _retrieve_precomputed_rendered(self, problem: ProblemSpec) -> RetrievalBundle:
        """Legacy: look up pre-rendered hint text. Bypasses pipeline."""
        entry = self._prompt_info.get(problem.uid)
        if entry and entry.get("hint"):
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=entry["hint"],
                retrieved_items=[],
                metadata={"selector_mode": "precomputed"},
            )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=None,
            retrieved_items=[],
            metadata={"selector_mode": "precomputed_miss"},
        )

    def _retrieve_precomputed_names(
        self, concept_mem: ConceptMemory, problem: ProblemSpec
    ) -> RetrievalBundle:
        """Precomputed concept names → filter → route → render."""
        names = self._selected_concepts.get(problem.uid)
        if not names:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "precomputed_miss"},
            )
        return self._apply_pipeline(
            concept_mem=concept_mem,
            selected_names=names,
            problem=problem,
            selector_mode="precomputed",
        )

    # ------------------------------------------------------------------ #
    #  Shared pipeline: filter → route → render                            #
    # ------------------------------------------------------------------ #
    def _apply_pipeline(
        self,
        *,
        concept_mem: ConceptMemory,
        selected_names: list[str] | None,
        problem: ProblemSpec,
        selector_mode: str,
        extra_metadata: dict[str, Any] | None = None,
        variant_flags: dict[str, Any] | None = None,
    ) -> RetrievalBundle:
        """Apply filter → route → render after selection."""
        pre_filter_count = len(selected_names) if selected_names is not None else 0
        if selected_names is not None:
            filtered = self._filter.filter(selected_names)
        else:
            filtered = None

        hint_text = self._render_hint_text(concept_mem, filtered,
                                            variant_flags=variant_flags)

        # Route (delegated to format-independent RetrievalRouter)
        decision = self._router.should_include(
            filtered, hint_text, pre_filter_count=pre_filter_count
        )
        if not decision:
            hint_text = None

        metadata: dict[str, Any] = {
            "selector_mode": selector_mode,
            "concept_count": len(concept_mem.concepts),
            "render_mode": self.render_mode,
            "routing_included": decision.include,
            "pre_filter_count": pre_filter_count,
        }
        if decision.reasons:
            metadata["routing_skip_reasons"] = decision.reasons
        if filtered is not None:
            metadata["selected_count"] = len(filtered)
            metadata["selected_names"] = filtered
        if extra_metadata:
            metadata.update(extra_metadata)

        retrieved_items = (
            [{"concept": n} for n in filtered]
            if filtered is not None
            else [{"concept": n} for n in concept_mem.concepts]
        )

        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text,
            retrieved_items=retrieved_items,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        """Synchronous retrieve.

        Selection mode priority:
        1. selected_concepts_file → precomputed names → pipeline
        2. prompt_info_file → precomputed rendered → bypass pipeline
        3. All concepts → pipeline
        """
        # Mode 2: legacy precomputed rendered (bypasses pipeline)
        if self._selected_concepts is None and self._prompt_info is not None:
            return self._retrieve_precomputed_rendered(problem)

        # All other modes need ConceptMemory from payload
        concept_mem = self._reconstruct_memory(memory)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "empty", "concept_count": 0},
            )

        # Mode 1: precomputed names → through pipeline
        if self._selected_concepts is not None:
            return self._retrieve_precomputed_names(concept_mem, problem)

        vflags = self._extract_variant_flags(memory)
        return self._apply_pipeline(
            concept_mem=concept_mem,
            selected_names=None,
            problem=problem,
            selector_mode="all_concepts",
            variant_flags=vflags,
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
        """Async retrieve — precomputed, inline LLM, or all concepts.

        Selection mode priority:
        1. selected_concepts_file → precomputed names → pipeline
        2. prompt_info_file → precomputed rendered → bypass pipeline
        3. use_llm_selector → LLM selection → pipeline
        4. All concepts → pipeline
        """
        # Mode 2: legacy precomputed rendered (bypasses pipeline)
        if self._selected_concepts is None and self._prompt_info is not None:
            return self._retrieve_precomputed_rendered(problem)

        # All other modes need ConceptMemory from payload
        concept_mem = self._reconstruct_memory(memory)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "empty", "concept_count": 0},
            )

        # Mode 1: precomputed names → through pipeline
        if self._selected_concepts is not None:
            return self._retrieve_precomputed_names(concept_mem, problem)

        vflags = self._extract_variant_flags(memory)

        # Mode 4: all concepts (no LLM)
        if not self.use_llm_selector:
            return self._apply_pipeline(
                concept_mem=concept_mem,
                selected_names=None,
                problem=problem,
                selector_mode="all_concepts",
                variant_flags=vflags,
            )

        # Mode 3: inline LLM selection → through pipeline
        profile = self._build_profile(concept_mem)
        sel_flags = _RENDER_PROFILES.get(
            self.selector_render_mode, _RENDER_PROFILES["full"]
        )
        full_concepts_str = concept_mem.to_string(
            usage_threshold=0,
            profile=profile,
            skip_cues=sel_flags["skip_cues"],
            skip_implementation=sel_flags["skip_implementation"],
            skip_parameters=sel_flags["skip_parameters"],
            skip_parameter_description=sel_flags["skip_parameter_description"],
            include_description=sel_flags["include_description"],
        )

        select_template, _ = self._get_prompt_templates()
        puzzle_str = self._format_problem_for_selection(problem)
        selection_prompt = select_template.format(
            concepts=full_concepts_str,
            puzzle=puzzle_str,
        )

        model_name = self.selector_model or selector_model
        if not model_name:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "no_model"},
            )

        try:
            completions = await provider.async_generate(
                prompt=selection_prompt,
                model=model_name,
                gen_cfg=self.selector_gen_cfg,
            )
            selector_completion = str(completions[0]) if completions else ""
        except Exception as exc:
            logger.warning(f"Concept selector LLM call failed: {exc}")
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={
                    "selector_mode": "no_hints_fallback",
                    "selector_error": f"{type(exc).__name__}: {exc}",
                },
            )

        valid_names = set(concept_mem.concepts.keys())
        selected_names, parse_error = self._parse_concept_selection(
            selector_completion, valid_names
        )

        if not selected_names:
            logger.info(f"Concept selection parse failed: {parse_error}")
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={
                    "selector_mode": "no_hints_parse_fallback",
                    "selector_parsing_error": parse_error,
                    "selector_completion": selector_completion,
                },
            )

        return self._apply_pipeline(
            concept_mem=concept_mem,
            selected_names=selected_names,
            problem=problem,
            selector_mode="llm_selected",
            variant_flags=vflags,
        )
