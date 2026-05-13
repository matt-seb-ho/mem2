# MediQ policy adapted memory prompt

Paper: MediQ, arXiv:2406.00922, Sections 2.2 and 2.2.2.

Mechanism rationale: MediQ converts static QA into an interactive consultation. The Expert system first performs an initial assessment, then uses an abstention module to decide whether to answer or ask an atomic information-seeking question. The question generation module targets the most useful missing feature, and the conversation log grows until confidence is high enough to commit. This adapter rewrites each ARC concept into policy metadata that can guide multi-round concept retrieval with abstention and targeted follow-up.

You are adapting one ARC-AGI concept-memory entry into the native policy form expected by MediQ-style interactive retrieval.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "initial_assessment": "one sentence about what this concept can diagnose in a partial ARC state",
  "question_type": "object_property|spatial_relation|color_pattern|counting|transformation|container_boundary|symmetry|other",
  "missing_information_targets": ["specific feature to ask about"],
  "atomic_question_templates": ["one bite-sized ARC follow-up question"],
  "expected_info_gain": 0.0,
  "abstention_policy": {{
    "ask_when": "condition under which this concept should ask rather than commit",
    "commit_when": "condition under which this concept is enough to answer",
    "confidence_threshold_hint": 0.0
  }},
  "evidence_integration": "how a response updates the known ARC state",
  "routing_keywords": ["query term", "query term"],
  "retrieval_notes": "one sentence explaining why this policy metadata belongs to the concept"
}}

Rules:
- The atomic question must be one specific, bite-sized question about an ARC puzzle feature, not a broad instruction.
- expected_info_gain and confidence_threshold_hint must be numbers between 0 and 1.
- missing_information_targets should identify concrete visual or solver-state gaps.
- question_type must be one of the listed labels.
- All JSON strings must be single-line strings.
- Do not include unescaped quotation marks inside string values. Use apostrophes or plain wording instead.
- Atomic questions should not quote colors, object names, or labels with double quotes.
- Do not invent puzzle IDs, task outcomes, or medical content.
- Ground every field in the supplied concept fields.

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
