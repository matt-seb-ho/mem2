# MemTree adapted memory prompt

Paper: MemTree: A Structured Memory Representation for Efficient Long-Term Context, arXiv:2410.14052, Sections 3.1 and 3.2.

Mechanism rationale: MemTree stores memory as a dynamic tree T = (V, E). Each node contains aggregated textual content, a parent pointer, children, depth, and an embedding text. New information is inserted by traversing from the root, comparing the new content to child nodes with a depth-adaptive threshold, then either descending into a similar subtree or creating a new leaf. Retrieval uses collapsed-tree search over all nodes. This adapter rewrites one ARC concept into a MemTree leaf with path-to-root summaries, insertion rationale, and collapsed-tree retrieval text.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by MemTree.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "tree_position": {{
    "leaf_node_id": "memtree leaf id for this concept",
    "parent_node_id": "best parent community id from report_context",
    "depth": 2,
    "insertion_decision": "traverse_deeper|create_new_leaf|expand_leaf",
    "depth_threshold_rationale": "why this depth and parent preserve hierarchy"
  }},
  "node_content": {{
    "leaf_content": "textual content stored at the concept leaf",
    "embedding_text": "dense-retrieval text for this node",
    "aggregate_contribution": "how this leaf updates parent aggregated content"
  }},
  "path_to_root": [
    {{"node_id": "leaf or ancestor id", "depth": 0, "content_summary": "node summary", "update_role": "how this node participates in ancestor aggregation"}}
  ],
  "collapsed_retrieval_card": "compact node text for collapsed-tree retrieval",
  "retrieval_keywords": ["keywords for collapsed-tree query matching"],
  "sibling_group": {{
    "sibling_role": "how this leaf relates to siblings under the parent",
    "near_sibling_concepts": ["concept ids from report_context if available"]
  }}
}}

Rules:
- parent_node_id must be copied exactly from this allowed list: {allowed_parent_ids}.
- If report_context is broad because the concept has no direct membership, choose the closest allowed parent by semantic fit.
- path_to_root must start with the leaf_node_id and include at least one ancestor community from report_context.
- depth should be 2 or greater unless the context truly indicates root-level placement.
- insertion_decision should mirror MemTree update logic: descend if similar to an existing subtree, create a new leaf if distinct, or expand a leaf if it splits a specific concept cluster.
- collapsed_retrieval_card and embedding_text must be faithful to the concept fields.
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

# Hierarchical report context

allowed_parent_ids: {allowed_parent_ids}

{report_context}

Return the JSON object now.
