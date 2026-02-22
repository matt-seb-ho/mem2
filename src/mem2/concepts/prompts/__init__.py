"""Prompt template registry for concept-based pipelines."""

from mem2.concepts.prompts.arc_hints import HINT_TEMPLATE_OP3
from mem2.concepts.prompts.arc_select import SELECT_PROMPT_TEMPLATE
from mem2.concepts.prompts.math_hints import MATH_HINT_TEMPLATE
from mem2.concepts.prompts.math_select import MATH_SELECT_PROMPT_TEMPLATE

CONCEPT_PROMPT_TEMPLATES = {
    "arc_select": SELECT_PROMPT_TEMPLATE,
    "hint_op3": HINT_TEMPLATE_OP3,
    "math_select": MATH_SELECT_PROMPT_TEMPLATE,
    "math_hint": MATH_HINT_TEMPLATE,
}

# Domain -> (select_template, hint_template)
DOMAIN_PROMPT_MAP = {
    "arc": (SELECT_PROMPT_TEMPLATE, HINT_TEMPLATE_OP3),
    "math": (MATH_SELECT_PROMPT_TEMPLATE, MATH_HINT_TEMPLATE),
    "code": (MATH_SELECT_PROMPT_TEMPLATE, MATH_HINT_TEMPLATE),  # reuse math for now
}

__all__ = [
    "CONCEPT_PROMPT_TEMPLATES",
    "DOMAIN_PROMPT_MAP",
    "HINT_TEMPLATE_OP3",
    "MATH_HINT_TEMPLATE",
    "MATH_SELECT_PROMPT_TEMPLATE",
    "SELECT_PROMPT_TEMPLATE",
]
