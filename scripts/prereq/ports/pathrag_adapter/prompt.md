# PathRAG adapted memory prompt

Paper: PathRAG, arXiv:2502.14902, Methodology section.

Mechanism rationale: PathRAG does not flatten retrieved graph nodes and edges into an unordered context. It retrieves relevant graph nodes from query keywords, extracts reliable relational paths between those nodes with flow-based pruning, and renders each path as a textual sequence of node chunks and edge chunks. This adapter rewrites each ARC concept into a small set of native PathRAG path records so the retriever can prompt with relational paths rather than flat concept descriptions.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by PathRAG.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "query_keywords": ["keywords that should retrieve graph nodes for this concept"],
  "path_nodes": [
    {{"node_id": "n1", "label": "entity or concept node", "text_chunk": "node textual chunk", "node_type": "operation|object|attribute|parameter|condition|transformation|spatial_relation|pattern|output|concept|other"}}
  ],
  "entity_paths": [
    {{
      "path_id": "p1",
      "nodes": ["n1", "n2", "n3"],
      "edges": [
        {{"src": "n1", "dst": "n2", "relation": "short relation", "text_chunk": "edge textual chunk"}}
      ],
      "textual_path": "node chunk; relation chunk; next node chunk",
      "reliability_hint": 0.0,
      "pruning_rationale": "why this path is important under flow-based pruning"
    }}
  ],
  "answer_generation_notes": "one sentence about why the paths should be preserved as paths"
}}

Rules:
- Produce 3 to 5 entity_paths when possible.
- Each path must contain at least 2 nodes and at least 1 edge.
- Use path_nodes node_id values inside entity_paths.nodes and entity_paths.edges.
- textual_path must preserve the order of nodes and edge text chunks.
- reliability_hint must be a number between 0 and 1. Higher means more likely to survive PathRAG flow pruning.
- Do not invent task outcomes, puzzle IDs, or concepts outside the supplied fields.
- If the source concept is simple, produce shorter paths but still preserve node-edge-node structure.

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
