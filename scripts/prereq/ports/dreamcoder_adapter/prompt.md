# DreamCoder adapted memory prompt

Paper: DreamCoder, arXiv:2006.08381.

Mechanism rationale: DreamCoder alternates wake search over task frontiers with sleep-time library compression. Its compression step invents reusable typed primitives from repeated program fragments when they reduce description length. This adapter rewrites one ARC concept into a DreamCoder-like compression record that can stand in for a frontier fragment card, while explicitly staying non-executable.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "frontier_signature": "what this concept would look like as a recurring frontier/program fragment",
  "invented_primitive_candidate": {{
    "name_hint": "dreamcoder-style primitive name",
    "arity_hint": 0,
    "typed_inputs": ["input type hints"],
    "typed_output": "output type hint",
    "reusable_behavior": "behavior abstracted by the primitive"
  }},
  "compression_roles": [
    {{"role": "shared_subtree|frontier_task|recognition_cue|mdl_gain_cue", "text": "grounded role text"}}
  ],
  "fragment_terms": ["query or fragment terms useful for retrieval"],
  "mdl_notes": "why this fragment may or may not reduce description length"
}}

Rules:
- Keep the record faithful to the supplied concept only.
- Do not claim executable DreamCoder programs or solved task frontiers.
- arity_hint must be a non-negative integer.
- Include 2 to 6 compression_roles and 4 to 10 fragment_terms.

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
