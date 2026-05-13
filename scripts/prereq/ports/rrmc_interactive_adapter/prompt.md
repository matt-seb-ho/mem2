# RRMC interactive adapted memory prompt

Source: `origins/threads/interactive_retrieval/source.md`.

Mechanism rationale: Aaron's RRMC interactive retrieval framing treats retrieval as a multi-round selection process rather than a one-shot top-k lookup. A concept should carry metadata about when it is useful as an initial probe, when it becomes useful after previous evidence, what coverage it contributes, and when the selector should stop refining. This adapter rewrites each ARC concept into that native multi-round selector form before the RRMC interactive retriever scores concepts.

You are adapting one ARC-AGI concept-memory entry into multi-round selector metadata for RRMC-style interactive retrieval.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "selector_role": "seed_probe|refinement_probe|disambiguation_probe|commit_signal|other",
  "round_1_relevance": 0.0,
  "round_2_relevance": 0.0,
  "coverage_targets": ["coverage dimension this concept adds"],
  "probe_plan": [
    {{
      "round": 1,
      "probe_question": "specific selector question",
      "expected_signal": "what observation would make this concept relevant",
      "selector_update": "how to update the selected concept set"
    }}
  ],
  "convergence_signal": "condition that says additional rounds add little value",
  "routing_keywords": ["query term", "query term"],
  "retrieval_notes": "one sentence explaining the multi-round selector role"
}}

Rules:
- Include exactly two probe_plan items, one for round 1 and one for round 2.
- round_1_relevance and round_2_relevance must be numbers between 0 and 1.
- selector_role must be one of the listed labels.
- Probe questions must be ARC-specific and bite-sized.
- coverage_targets should identify what this concept contributes beyond already selected concepts.
- Do not invent puzzle IDs or task outcomes.
- Ground every field in the supplied concept fields.
- All JSON strings must be single-line strings.
- Do not include unescaped quotation marks inside string values.

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
