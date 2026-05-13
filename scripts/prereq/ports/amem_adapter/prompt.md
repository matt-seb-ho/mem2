# A-Mem adapted memory prompt

Paper: A-Mem: Agentic Memory for LLM Agents, arXiv:2502.12110, Sections 3.1 to 3.4.

Mechanism rationale: A-Mem stores each memory as an atomic Zettelkasten-style note with content, timestamp, LLM-generated keywords, tags, contextual description, embedding text, and links to related notes. Link generation identifies meaningful connections to historical memories, and memory evolution updates context, keywords, and tags as new memories arrive. This adapter rewrites one ARC concept into an A-Mem note with fresh LLM-generated links and evolution cues, avoiding the prior heuristic fallback link artifact.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by A-Mem.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "note": {{
    "content": "atomic note content grounded in the concept",
    "timestamp": "stable concept-memory timestamp or lifecycle stage",
    "keywords": ["LLM-generated retrieval keywords"],
    "tags": ["LLM-generated organization tags"],
    "contextual_description": "rich contextual description for semantic retrieval"
  }},
  "zettel_links": [
    {{
      "target_concept": "one id from candidate_neighbors",
      "link_type": "generalizes|specializes|prerequisite_of|contrast_with|applied_with|similar_to|updates_context_of",
      "rationale": "why this link is meaningful",
      "confidence": 0.0
    }}
  ],
  "memory_evolution": {{
    "context_update": "how this note should refine its own context after linking",
    "tag_updates": ["new or strengthened tags"],
    "neighbor_update_suggestions": [
      {{"target_concept": "linked concept id", "suggested_update": "how the neighbor note should evolve"}}
    ]
  }},
  "retrieval_text": "compact text used when querying the A-Mem note network"
}}

Rules:
- Create 3 to 5 zettel_links when possible, using only target_concept values from candidate_neighbors.
- Do not copy candidate rationales blindly. Infer the relationship from the concept fields and candidate descriptions.
- confidence must be a number between 0.0 and 1.0.
- The note must be atomic: one coherent memory unit, not a broad taxonomy.
- keywords and tags should be useful for multi-hop retrieval through a Zettelkasten note network.
- memory_evolution must describe actual context or tag refinements, not a generic placeholder.
- Do not invent puzzle outcomes, hidden tasks, or concepts outside the supplied fields.

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
used_in_count: {used_in_count}

# Candidate neighbors

{candidate_neighbors}

Return the JSON object now.
