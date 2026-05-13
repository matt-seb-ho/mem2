# LILO adapted memory prompt

Paper: LILO, arXiv:2310.19791.

Mechanism rationale: LILO proposes human-interpretable library abstractions with language grounding, then uses compression and documentation loops around a program library. This adapter rewrites one ARC concept into a LILO-like library card: a candidate abstraction, member-role hints, language grounding, and notes for iterative growth. It must remain honest that the record is non-executable.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "library_profile": "how this concept behaves as a reusable library entry",
  "abstraction_proposal": {{
    "readable_name_hint": "human-readable abstraction name",
    "members_or_roles": ["member concepts or conceptual roles this abstraction could cover"],
    "function_expression_hint": "non-executable pseudocode or DSL-like expression",
    "description": "natural-language documentation for the abstraction"
  }},
  "language_grounding": [
    {{"phrase": "user-facing phrase", "grounding": "what source field supports it"}}
  ],
  "abstraction_terms": ["terms useful for selecting this library abstraction"],
  "iterative_growth_notes": "how this should be used in a one-abstraction-per-iteration growth loop"
}}

Rules:
- Keep all content faithful to the supplied concept.
- Do not claim runnable LILO code, Stitch compression, or AutoDoc validation.
- Include 2 to 6 language_grounding entries and 4 to 10 abstraction_terms.
- function_expression_hint may be pseudocode, but must be grounded in the concept fields.

# Concept fields

concept_id: {concept_id}
name: {name}
kind: {kind}
routine_subtype: {routine_subtype}
output_typing: {output_typing}
description: {description}
parameters: {parameters}
cues: {cues}
implementation: {implementation}

Return the JSON object now.
