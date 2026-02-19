"""ConceptSelectorRetriever: LLM-based concept selection producing rich hint_text.

Uses ConceptMemory and SELECT_PROMPT_TEMPLATE for concept selection,
then renders selected concepts through HINT_TEMPLATE_OP3.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import yaml

from mem2.concepts.memory import ConceptMemory
from mem2.concepts.prompts import HINT_TEMPLATE_OP3, SELECT_PROMPT_TEMPLATE
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


def _extract_first_yaml_block(text: str) -> str | None:
    m = _YAML_BLOCK_RE.search(text)
    if not m:
        return None
    return m.group(1)


class ConceptSelectorRetriever:
    """LLM-based concept selection retriever using rich ConceptMemory.

    Workflow:
    1. Reconstruct ConceptMemory from memory.payload
    2. Render full concept list
    3. Build selection prompt with SELECT_PROMPT_TEMPLATE
    4. LLM call -> YAML list of concept names
    5. Re-render with selection (concept_names=selected, show_other_concepts=True)
    6. Format through HINT_TEMPLATE_OP3
    7. Return RetrievalBundle with rich hint_text
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
    ):
        self.top_k = int(top_k)
        self.domain = domain
        self.use_llm_selector = bool(use_llm_selector)
        self.selector_model = str(selector_model or "")
        self.selector_gen_cfg = dict(
            selector_gen_cfg or {"n": 1, "temperature": 0.0, "max_tokens": 1024}
        )
        self.hint_template_key = hint_template_key

    def _reconstruct_memory(self, memory: MemoryState) -> ConceptMemory:
        return ConceptMemory.from_payload(memory.payload)

    def _parse_concept_selection(
        self, completion: str, valid_names: set[str]
    ) -> tuple[list[str], str | None]:
        if not completion.strip():
            return [], "empty_completion"

        yaml_text = _extract_first_yaml_block(completion) or completion
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
            elif isinstance(item, dict):
                if len(item) == 1:
                    k, v = next(iter(item.items()))
                    name = f"{v}".strip() if isinstance(v, str) else f"{k}".strip()
                else:
                    continue
            else:
                continue
            if name in valid_names and name not in selected:
                selected.append(name)

        if not selected:
            return [], "no_valid_names"
        return selected, None

    def _render_hint_text(
        self, concept_mem: ConceptMemory, selected_names: list[str] | None
    ) -> str:
        if selected_names:
            rendered = concept_mem.to_string(
                concept_names=selected_names,
                skip_parameter_description=False,
                usage_threshold=0,
                show_other_concepts=True,
            )
        else:
            rendered = concept_mem.to_string(usage_threshold=0)

        return HINT_TEMPLATE_OP3.format(hints=rendered)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        """Synchronous fallback: render all concepts without LLM selection."""
        concept_mem = self._reconstruct_memory(memory)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"selector_mode": "empty", "concept_count": 0},
            )

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
        """Async retrieve with LLM-based concept selection."""
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

        # Step 1: render full concept list for selection
        full_concepts_str = concept_mem.to_string(usage_threshold=0)

        # Step 2: build selection prompt
        puzzle_str = format_problem_for_prompt(problem)
        selection_prompt = SELECT_PROMPT_TEMPLATE.format(
            concepts=full_concepts_str,
            puzzle=puzzle_str,
        )

        # Step 3: LLM call
        model_name = self.selector_model or selector_model
        if not model_name:
            hint_text = self._render_hint_text(concept_mem, None)
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=hint_text,
                retrieved_items=[{"concept": n} for n in concept_mem.concepts],
                metadata={
                    "selector_mode": "all_concepts_no_model",
                    "concept_count": len(concept_mem.concepts),
                },
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
            hint_text = self._render_hint_text(concept_mem, None)
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=hint_text,
                retrieved_items=[{"concept": n} for n in concept_mem.concepts],
                metadata={
                    "selector_mode": "all_concepts_fallback",
                    "concept_count": len(concept_mem.concepts),
                    "selector_error": f"{type(exc).__name__}: {exc}",
                    "selector_prompt": selection_prompt,
                },
            )

        # Step 4: parse selection
        valid_names = set(concept_mem.concepts.keys())
        selected_names, parse_error = self._parse_concept_selection(
            selector_completion, valid_names
        )

        if not selected_names:
            logger.info(f"Concept selection parse failed: {parse_error}")
            hint_text = self._render_hint_text(concept_mem, None)
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=hint_text,
                retrieved_items=[{"concept": n} for n in concept_mem.concepts],
                metadata={
                    "selector_mode": "all_concepts_parse_fallback",
                    "concept_count": len(concept_mem.concepts),
                    "selector_parsing_error": parse_error,
                    "selector_prompt": selection_prompt,
                    "selector_completion": selector_completion,
                },
            )

        # Step 5: render selected concepts with full detail
        hint_text = self._render_hint_text(concept_mem, selected_names)

        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text,
            retrieved_items=[{"concept": n} for n in selected_names],
            metadata={
                "selector_mode": "llm_selected",
                "concept_count": len(concept_mem.concepts),
                "selected_count": len(selected_names),
                "selected_names": selected_names,
                "selector_prompt": selection_prompt,
                "selector_completion": selector_completion,
            },
        )
