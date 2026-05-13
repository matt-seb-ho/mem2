# MAGMA adapted memory prompt

Paper: MAGMA: A Multi-Graph Agentic Memory Architecture, arXiv:2601.03236, Sections 3.2 and 3.3.

Mechanism rationale: MAGMA stores each memory item as an event node with structured attributes, dense semantic content, and typed relation edges split across orthogonal semantic, temporal, causal, and entity graphs. Retrieval uses query analysis to select views, identify anchors, traverse graph neighbors with policy weights, and linearize the retrieved subgraph with provenance. This adapter rewrites one ARC concept into a MAGMA event-node record with view memberships and policy cues so the local retriever can select and render view-specific context instead of treating the concept as a flat text row.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by MAGMA multi-graph retrieval.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "event_node": {{
    "content": "1-3 sentence event-node content for this concept",
    "timestamp_hint": "stable ordering or lifecycle hint for this memory item",
    "attributes": ["structured attribute strings such as operation, entity, color, shape, transformation, criterion"]
  }},
  "view_memberships": [
    {{
      "view": "semantic|temporal|causal|entity|structural",
      "node_refs": ["node ids from typed view context when available, or concept::{concept_id}"],
      "edge_refs": ["short edge references grounded in typed view context"],
      "role": "how this concept participates in this view",
      "traversal_value": "why a MAGMA beam traversal should expand through this view",
      "query_intents": ["WHY", "WHEN", "ENTITY", "SEMANTIC", "STRUCTURAL"]
    }}
  ],
  "anchor_keywords": ["keywords for anchor identification"],
  "policy_hints": {{
    "preferred_views": ["semantic|temporal|causal|entity|structural"],
    "why_signal": "causal or rationale cue",
    "when_signal": "temporal or sequence cue",
    "entity_signal": "entity/object permanence cue"
  }},
  "graph_linearization_card": "structured context block preserving provenance and typed-view dependencies",
  "salience_budget": {{
    "keep_full": ["fields or relations that should stay verbose"],
    "summarize_if_needed": ["fields or relations that can be compressed"]
  }}
}}

Rules:
- Use at least two view_memberships.
- Include semantic, causal, or structural memberships when supported by the typed view context.
- Use only evidence from concept fields and typed view context. Do not invent puzzle outcomes.
- node_refs should use supplied node ids when possible. If none are supplied for a view, use "concept::{concept_id}".
- policy_hints.preferred_views must be ordered by expected usefulness for retrieval.
- graph_linearization_card should read like MAGMA context scaffolding with reference IDs and relation types.
- Keep the record concise but specific enough for retrieval scoring.

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

# MAGMA typed-view context for this concept

{typed_view_context}

Return the JSON object now.
