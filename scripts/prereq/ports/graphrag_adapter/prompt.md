# GraphRAG adapted memory prompt

Paper: From Local to Global: A GraphRAG Approach to Query-Focused Summarization, arXiv:2404.16130, Section 3.1.

Mechanism rationale: GraphRAG indexes a corpus as an entity and relationship graph, partitions it into hierarchical Leiden communities, and pre-generates community reports. At query time, community reports produce partial answers that are reduced into a global answer. This adapter rewrites each ARC concept into a community-report contribution record so the retriever can use concept-level roles inside GraphRAG community summaries instead of flat concept descriptions.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by GraphRAG global search.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "primary_community_id": "best matching community id from the supplied community context",
  "community_role": "short role this concept plays in the community report",
  "contribution_to_cluster": "2-4 sentence report-style summary of how this concept contributes to the community",
  "map_reduce_card": "compact text that can be used as a GraphRAG map-step context unit",
  "summary_path": [
    {{"level": 0, "community_id": "community id", "role_at_level": "leaf/intermediate/root role", "report_connection": "how this concept relates to that report"}}
  ],
  "entity_relationship_claims": [
    {{"claim": "entity or relationship claim grounded in supplied fields", "importance": "high|medium|low"}}
  ],
  "query_focus_keywords": ["keywords for scoring community reports against ARC queries"]
}}

Rules:
- Use only community ids supplied in the community context.
- summary_path should include 1 to 3 levels when available, from specific to broad.
- contribution_to_cluster and map_reduce_card must be faithful to the concept fields and community summaries.
- entity_relationship_claims should be concise GraphRAG-style claims about entities, relationships, or solver operations.
- query_focus_keywords should include terms likely to connect user queries to this community report.
- Do not invent task outcomes, puzzle IDs, or concepts outside the supplied fields.

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

# Community context containing this concept

{community_context}

# Hierarchical report path candidates containing this concept

{hierarchical_context}

Return the JSON object now.
