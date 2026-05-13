# Memp adapted memory prompt

Paper: MemP, arXiv:2508.06433.

Mechanism rationale: Memp stores procedural memories distilled from trajectories, retrieves them for analogous tasks, and updates or prunes them using hit/success and adjustment signals. This adapter rewrites one ARC concept into a Memp-like procedural card so the local port can expose procedure-shaped memory rather than only flat concept text.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "procedure_card": "compact procedural memory preserving the concept's solver meaning",
  "workflow_steps": ["ordered procedural steps grounded in the source concept"],
  "success_conditions": ["conditions under which this memory should help"],
  "failure_or_adjustment_signals": ["signals that the memory should be revised, downweighted, or pruned"],
  "procedure_terms": ["retrieval terms for analogous tasks"],
  "hit_success_notes": "how hit/success accounting should treat this procedure"
}}

Rules:
- Keep content faithful to the supplied concept only.
- Do not invent trajectory IDs, success statistics, or task outcomes.
- Include 2 to 6 workflow_steps, 2 to 6 success_conditions, and 4 to 10 procedure_terms.
- failure_or_adjustment_signals should be conservative and source-grounded.

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
