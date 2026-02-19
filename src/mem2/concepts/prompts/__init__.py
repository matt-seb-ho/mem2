"""Prompt template registry for concept-based pipelines."""

from mem2.concepts.prompts.arc_hints import HINT_TEMPLATE_OP3
from mem2.concepts.prompts.arc_select import SELECT_PROMPT_TEMPLATE

CONCEPT_PROMPT_TEMPLATES = {
    "arc_select": SELECT_PROMPT_TEMPLATE,
    "hint_op3": HINT_TEMPLATE_OP3,
}

__all__ = [
    "CONCEPT_PROMPT_TEMPLATES",
    "HINT_TEMPLATE_OP3",
    "SELECT_PROMPT_TEMPLATE",
]
