from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ArcMemoPromptOptions:
    """ArcMemo-aligned prompt option surface for default ARC runs."""

    include_hint: bool = False
    hint_template_key: str = "selected"
    require_hint_citations: bool = False
    instruction_key: str = "default"
    system_prompt_key: str = "default"
    problem_data: str | None = None
    problem_data_variant_key: str | None = None

