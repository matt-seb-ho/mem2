# HippoRAG PPR adapted memory prompt

Paper: HippoRAG, arXiv:2405.14831, Sections 2.2-2.3.

Mechanism rationale: HippoRAG stores passages through a schemaless OpenIE knowledge graph. The offline memory substrate contains passage text, noun-phrase or named-entity nodes, relation triples, and enough passage-node association data for Personalized PageRank to propagate from query nodes back to passages. This adapter rewrites each ARC concept into that native passage-plus-KG shape before the HippoRAG PPR retriever scores concepts.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by HippoRAG PPR.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "passage_text": "one compact passage that preserves the concept's solver meaning",
  "entity_mentions": [
    {{
      "text": "noun phrase or named entity node",
      "type": "operation|object|attribute|parameter|condition|transformation|spatial_relation|pattern|output|concept|other",
      "role": "how this node functions in the passage",
      "supporting_text": "phrase grounded in the supplied concept fields"
    }}
  ],
  "triples": [
    {{
      "subject": "entity mention text",
      "predicate": "short relation phrase",
      "object": "entity mention text",
      "confidence": 0.0,
      "supporting_text": "phrase grounded in the supplied concept fields"
    }}
  ],
  "query_node_terms": ["terms likely to appear in ARC task text or solver reasoning"],
  "node_specificity_hints": [
    {{"node": "entity mention text", "specificity": "high|medium|low", "reason": "local reason"}}
  ],
  "retrieval_notes": "one sentence explaining why this record should seed or receive PPR mass"
}}

Rules:
- Include 4 to 10 entity_mentions when the source has enough information.
- Include 2 to 6 triples. Triples must use entities from entity_mentions when possible.
- Do not invent task outcomes, puzzle IDs, or concepts outside the supplied fields.
- Keep passage_text faithful to the concept. If description is missing, synthesize from name, kind, parameters, cues, and implementation only.
- query_node_terms should be concise and useful for matching partial query cues.
- confidence values must be numbers between 0 and 1.

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

# Existing shared entity graph nodes for this concept

{entity_context}

# Existing shared OpenIE facts for this concept

{fact_context}

Return the JSON object now.
