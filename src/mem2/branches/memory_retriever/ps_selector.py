"""PsSelectorRetriever: PS (Program Synthesis) concept selection retriever.

Supports two modes:
1. **Precomputed (preferred)**: Loads hints from a prompt_info.json file produced
   by the offline ``scripts/select_concepts.py`` pipeline.
2. **Inline LLM (legacy)**: Per-problem LLM selection at runtime. Kept for
   backward compat but not recommended — offline selection is more debuggable.

Internal pipeline: select → filter → route → render
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


class PsSelectorRetriever:
    """PS (Program Synthesis) concept selection retriever.

    When ``prompt_info_file`` is set (recommended), loads pre-computed hints
    produced by ``scripts/select_concepts.py``.  No LLM calls at runtime.

    When ``prompt_info_file`` is not set, falls back to inline LLM selection
    (legacy behavior).

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
        render_mode: str = "full",
        max_frequency: float = 0.0,
        max_concepts_per_problem: int = 0,
        routing_strategy: str = "none",
        routing_max_hint_chars: int = 0,
        concept_frequency_file: str = "",
        **kwargs,
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
        self.max_frequency = float(max_frequency)
        self.max_concepts_per_problem = int(max_concepts_per_problem)
        self.routing_strategy = routing_strategy
        self.routing_max_hint_chars = int(routing_max_hint_chars)

        # Precomputed hints
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

        # Concept frequencies
        self._concept_frequencies: dict[str, float] = {}
        if concept_frequency_file:
            path = Path(concept_frequency_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                self._concept_frequencies = json.loads(path.read_text())
                logger.info(
                    f"Loaded concept frequencies for {len(self._concept_frequencies)} concepts"
                )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
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

    # ------------------------------------------------------------------ #
    #  Internal pipeline: select → filter → route → render                 #
    # ------------------------------------------------------------------ #
    def _select_concepts(
        self, concept_mem: ConceptMemory, problem: ProblemSpec
    ) -> list[str] | None:
        """Return selected concept names, or None for 'all concepts'."""
        # In sync mode without LLM, return None (all concepts)
        return None

    def _filter_concepts(self, selected_names: list[str]) -> list[str]:
        """Apply frequency filter and max cap."""
        filtered = selected_names
        if self.max_frequency > 0.0 and self._concept_frequencies:
            filtered = [
                n for n in filtered
                if self._concept_frequencies.get(n, 0.0) <= self.max_frequency
            ]
        if self.max_concepts_per_problem > 0:
            filtered = filtered[:self.max_concepts_per_problem]
        return filtered

    def _should_include_hints(
        self, selected_names: list[str] | None, hint_text: str | None
    ) -> bool:
        """Routing gate: decide whether to include hints for this problem."""
        if self.routing_strategy == "none":
            return True

        if self.routing_strategy == "selection_confidence":
            if not selected_names or not self._concept_frequencies:
                return True
            # Skip hints if all selected concepts are high-frequency (generic)
            if self.max_frequency > 0.0:
                threshold = self.max_frequency
            else:
                threshold = 0.5
            all_generic = all(
                self._concept_frequencies.get(n, 0.0) > threshold
                for n in selected_names
            )
            return not all_generic

        if self.routing_strategy == "hint_length":
            if not hint_text:
                return True
            if self.routing_max_hint_chars > 0 and len(hint_text) > self.routing_max_hint_chars:
                return False
            return True

        return True

    def _render_hint_text(
        self, concept_mem: ConceptMemory, selected_names: list[str] | None
    ) -> str:
        """Return raw rendered concept text (no hint-template wrapping).

        The inference engine is responsible for wrapping with its own hint
        template at prompt-build time — matching arc_memo's pattern.
        """
        profile = self._build_profile(concept_mem)
        render_flags = _RENDER_PROFILES.get(self.render_mode, _RENDER_PROFILES["full"])

        if selected_names:
            return concept_mem.to_string(
                concept_names=selected_names,
                skip_parameter_description=render_flags["skip_parameter_description"],
                skip_cues=render_flags["skip_cues"],
                skip_implementation=render_flags["skip_implementation"],
                skip_parameters=render_flags["skip_parameters"],
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
    #  Precomputed-hints path                                              #
    # ------------------------------------------------------------------ #
    def _retrieve_precomputed(self, problem: ProblemSpec) -> RetrievalBundle:
        """Look up pre-computed hint for this problem."""
        entry = self._prompt_info.get(problem.uid)
        if entry and entry.get("hint"):
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=entry["hint"],
                retrieved_items=[],
                metadata={"selector_mode": "precomputed"},
            )
        # No pre-computed hint for this problem — solve without hints
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=None,
            retrieved_items=[],
            metadata={"selector_mode": "precomputed_miss"},
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
    ) -> RetrievalBundle:
        """Apply filter → route → render after selection."""
        # Filter
        if selected_names is not None:
            filtered = self._filter_concepts(selected_names)
        else:
            filtered = None

        # Render
        hint_text = self._render_hint_text(concept_mem, filtered)

        # Route
        if not self._should_include_hints(filtered, hint_text):
            hint_text = None

        metadata: dict[str, Any] = {
            "selector_mode": selector_mode,
            "concept_count": len(concept_mem.concepts),
            "render_mode": self.render_mode,
        }
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
        """Synchronous retrieve."""
        if self._prompt_info is not None:
            return self._retrieve_precomputed(problem)

        # Reconstruct ConceptMemory from payload
        concept_mem = self._reconstruct_memory(memory)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "empty", "concept_count": 0},
            )

        return self._apply_pipeline(
            concept_mem=concept_mem,
            selected_names=None,
            problem=problem,
            selector_mode="all_concepts",
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
        """Async retrieve — precomputed or inline LLM selection."""
        # ── Precomputed path (preferred) ─────────────────────────────
        if self._prompt_info is not None:
            return self._retrieve_precomputed(problem)

        # ── Inline LLM selection (legacy) ────────────────────────────
        concept_mem = self._reconstruct_memory(memory)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "empty", "concept_count": 0},
            )

        if not self.use_llm_selector:
            return self._apply_pipeline(
                concept_mem=concept_mem,
                selected_names=None,
                problem=problem,
                selector_mode="all_concepts",
            )

        profile = self._build_profile(concept_mem)
        full_concepts_str = concept_mem.to_string(usage_threshold=0, profile=profile)

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
        )
