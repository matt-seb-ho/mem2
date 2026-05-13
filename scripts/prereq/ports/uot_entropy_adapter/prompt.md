# UoT entropy adapted memory prompt

Source: `literature/2402.03271.pdf`.

Paper mechanism: Uncertainty of Thoughts asks candidate yes/no questions, simulates possible future answer branches, scores each question with an uncertainty-based reward motivated by information gain, then propagates rewards to select the next question. The reward is highest when the candidate question splits the possibility set close to half and lowest when one branch has near-zero probability.

You are adapting one ARC-AGI concept-memory entry into UoT-style uncertainty-reduction metadata.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "uncertainty_state": "what ambiguity this concept helps resolve",
  "candidate_question": "yes/no ARC question that tests whether this concept should be included",
  "yes_partition_hint": ["what observations belong in the yes branch"],
  "no_partition_hint": ["what observations belong in the no branch"],
  "expected_yes_ratio": 0.5,
  "entropy_reward": 0.0,
  "information_gain_target": "uncertainty that should shrink after the answer",
  "simulation_tree_role": "root_candidate|answer_branch|questioner_branch|terminal_commit|other",
  "reward_propagation_notes": "how this concept should affect later expected reward",
  "routing_keywords": ["query term", "query term"],
  "retrieval_notes": "one sentence explaining the uncertainty-reduction role"
}}

Rules:
- candidate_question must be answerable yes or no.
- expected_yes_ratio must be a number between 0 and 1, preferably near 0.5 when the concept naturally halves possibilities.
- entropy_reward must be a number between 0 and 1. Use higher values for balanced, uncertainty-reducing questions.
- Include at least one yes_partition_hint and one no_partition_hint.
- simulation_tree_role must be one of the listed labels.
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
