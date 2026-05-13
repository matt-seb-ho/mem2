# H-MEM hierarchical adapted memory prompt

Paper: H-MEM, arXiv:2507.22925, Sections 3.1-3.2.

Mechanism rationale: H-MEM stores memory in semantic abstraction layers. A top-level domain points to categories, categories point to memory traces, and traces point to fine-grained episodes. Each upper-level entry carries positional indices for its children so retrieval can route layer by layer rather than compare a query with every low-level memory. This adapter rewrites each ARC concept into that native hierarchical memory form before the H-MEM retriever scores concepts.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by H-MEM.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "domain": "ARC-AGI",
  "category": "broad semantic category",
  "category_position_index": "L1:<short-id>",
  "subcategory": "memory trace group",
  "subcategory_position_index": "L2:<short-id>",
  "memory_trace": {{
    "title": "compact trace title",
    "keywords": ["routing keyword", "routing keyword"],
    "trace_summary": "one sentence describing the trace-level abstraction"
  }},
  "episode": {{
    "summary": "fine-grained concept memory episode",
    "grounded_operations": ["operation or object from supplied fields"],
    "when_to_route_here": "query condition that should descend to this episode"
  }},
  "routing_keywords": ["query term", "query term"],
  "confidence_weight": 0.0,
  "retrieval_notes": "one sentence explaining the top-down routing path"
}}

Rules:
- Use the supplied hierarchy hint when it is compatible with the concept. If the hint is missing or clearly wrong, choose a faithful ARC category and subcategory.
- category and subcategory must be semantic ARC groups, not generic words like "routine" or "structure".
- category_position_index and subcategory_position_index must be stable short strings suitable as routing pointers.
- memory_trace is the intermediate abstraction. episode is the concept-specific memory.
- routing_keywords should help match ARC task language and solver reasoning.
- confidence_weight must be a number between 0 and 1, reflecting how clearly this concept belongs to the chosen route.
- Do not invent puzzle IDs, task outcomes, or behavior not grounded in the supplied concept fields.

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

# Existing hierarchy hint

{hierarchy_context}

Return the JSON object now.
