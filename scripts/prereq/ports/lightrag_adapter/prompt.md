# LightRAG adapted memory prompt

Source: `literature/2410.05779.pdf`.

Paper mechanism: LightRAG builds a graph-indexed memory from chunks by extracting entities and relationships, enriching each entity and relation as key-value summaries, then retrieving at two levels. Low-level retrieval matches local keywords to entity nodes. High-level retrieval matches global keywords to relationships and gathers one-hop neighboring graph context.

You are adapting one ARC-AGI concept-memory entry into LightRAG-style dual-level graph metadata.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "local_entities": [
    {{"mention": "entity mention", "entity_type": "type", "entity_summary": "value summary for local retrieval"}}
  ],
  "global_relationships": [
    {{"relation": "relationship label", "target_concept": "neighbor concept or entity", "relation_summary": "value summary for global retrieval", "strength": 0.0}}
  ],
  "low_level_keywords": ["specific entity keyword"],
  "high_level_keywords": ["abstract relationship or theme keyword"],
  "entity_value_summary": "paragraph-style local entity value",
  "relation_value_summary": "paragraph-style global relationship value",
  "one_hop_neighbors": ["neighbor concept"],
  "chunk_reference": "concept-text chunk this graph record summarizes",
  "retrieval_notes": "one sentence explaining the dual-level retrieval role"
}}

Rules:
- local_entities should represent entity-node membership for this concept.
- global_relationships should represent relationship-edge membership or plausible high-level relation membership.
- low_level_keywords should be concrete entity words. high_level_keywords should be broader relationship or theme words.
- one_hop_neighbors must use only concepts or entities supported by the supplied LightRAG substrate context or the supplied concept fields.
- strength must be a plain JSON number between 0 and 1, for example 0.9. Never put words before or after the number.
- Do not invent puzzle IDs or task outcomes.
- Ground every field in the supplied concept fields and LightRAG substrate context.
- All JSON strings must be single-line strings.
- Do not include unescaped quotation marks inside string values.
- The response must parse with json.loads exactly as returned: no comments, no stray labels, no trailing prose.

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

# LightRAG substrate context

{lightrag_context}

Return the JSON object now.
