# HippoRAG2 adapted memory prompt

Paper: HippoRAG2 / From RAG to Memory, arXiv:2502.14802.

Mechanism rationale: HippoRAG2 keeps HippoRAG's graph/PPR retrieval core but adds a second-stage fact filter that trims a broad PPR candidate set to the passages directly relevant to the query. This adapter rewrites each ARC concept into (1) a PPR passage and (2) filter-ready evidence so the `hipporag2_filter` port can rank with HippoRAG2-shaped records rather than flat concept descriptions.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by the HippoRAG2 filter retriever.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "ppr_passage": "one compact passage that preserves the concept's solver meaning",
  "candidate_profile": "why this concept should remain after a HippoRAG2 fact filter",
  "query_filter_terms": ["query terms or reasoning cues that should keep this candidate"],
  "filter_evidence": [
    {{
      "claim": "short grounded fact about what the concept helps solve",
      "supporting_text": "phrase grounded in the supplied concept fields",
      "specificity": "high|medium|low"
    }}
  ],
  "reject_signals": ["queries or situations where this concept is likely unrelated"],
  "rerank_notes": "one sentence explaining how the filter should use this record"
}}

Rules:
- Include 4 to 10 query_filter_terms when the source has enough information.
- Include 2 to 6 filter_evidence entries.
- Do not invent puzzle IDs, outcomes, or external concepts.
- Keep ppr_passage faithful to the concept. If description is missing, synthesize only from name, kind, parameters, cues, and implementation.
- reject_signals should be conservative. Use source-grounded boundaries, not arbitrary exclusions.

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
