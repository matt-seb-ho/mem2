"""ConceptSelectorRetriever: concept selection retriever using pre-computed hints.

Supports two modes:
1. **Precomputed (preferred)**: Loads hints from a prompt_info.json file produced
   by the offline ``scripts/select_concepts.py`` pipeline.
2. **Inline LLM (legacy)**: Per-problem LLM selection at runtime. Kept for
   backward compat but not recommended — offline selection is more debuggable.
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


class ConceptSelectorRetriever:
    """Concept selection retriever.

    When ``prompt_info_file`` is set (recommended), loads pre-computed hints
    produced by ``scripts/select_concepts.py``.  No LLM calls at runtime.

    When ``prompt_info_file`` is not set, falls back to inline LLM selection
    (legacy behavior).
    """

    name = "concept_selector"

    def __init__(
        self,
        top_k: int = 10,
        domain: str = "arc",
        use_llm_selector: bool = True,
        selector_model: str = "",
        selector_gen_cfg: dict[str, Any] | None = None,
        hint_template_key: str = "op3",
        prompt_info_file: str = "",
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

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _reconstruct_memory(self, memory: MemoryState) -> ConceptMemory:
        return ConceptMemory.from_payload(memory.payload)

    def _get_prompt_templates(self):
        return DOMAIN_PROMPT_MAP.get(self.domain, DOMAIN_PROMPT_MAP["arc"])

    def _format_problem_for_selection(self, problem: ProblemSpec) -> str:
        if self.domain in ("math", "code"):
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

    def _render_hint_text(
        self, concept_mem: ConceptMemory, selected_names: list[str] | None
    ) -> str:
        """Return raw rendered concept text (no hint-template wrapping).

        The inference engine is responsible for wrapping with its own hint
        template at prompt-build time — matching arc_memo's pattern.
        """
        profile = self._build_profile(concept_mem)
        if selected_names:
            return concept_mem.to_string(
                concept_names=selected_names,
                skip_parameter_description=False,
                usage_threshold=0,
                show_other_concepts=True,
                profile=profile,
            )
        return concept_mem.to_string(usage_threshold=0, profile=profile)

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

        # Legacy fallback: no hints
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=None,
            retrieved_items=[],
            metadata={"selector_mode": "sync_no_hints"},
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
            hint_text = self._render_hint_text(concept_mem, None)
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=hint_text,
                retrieved_items=[{"concept": n} for n in concept_mem.concepts],
                metadata={
                    "selector_mode": "all_concepts",
                    "concept_count": len(concept_mem.concepts),
                },
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

        hint_text = self._render_hint_text(concept_mem, selected_names)
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text,
            retrieved_items=[{"concept": n} for n in selected_names],
            metadata={
                "selector_mode": "llm_selected",
                "selected_count": len(selected_names),
                "selected_names": selected_names,
            },
        )
