# RAPTOR adapted memory prompt

Paper: RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval, arXiv:2401.18059, Sections 3-4.

Mechanism rationale: RAPTOR recursively embeds, clusters, and summarizes text chunks into a tree. Retrieval can traverse the tree from broad summaries to fine leaves, or collapse all layers and retrieve the best nodes across granularities. This adapter assigns each ARC concept to a RAPTOR leaf and records a path-to-root summary chain so retrieval uses tree-positioned memory rather than flat concept records.

You are adapting one ARC-AGI concept-memory entry into the native memory form expected by RAPTOR.

Return exactly one JSON object. No markdown fences. No commentary.

Required JSON schema:
{{
  "concept_id": "{concept_id}",
  "leaf_node_id": "best matching level-0 RAPTOR node id from the supplied candidates",
  "tree_membership_rationale": "why this concept belongs in that leaf cluster",
  "leaf_text": "RAPTOR leaf-style text chunk for this concept",
  "path_to_root": [
    {{"level": 0, "node_id": "node id", "summary_role": "how this concept relates to the node summary", "retrieval_text": "text useful when this node is retrieved"}}
  ],
  "collapsed_tree_keywords": ["keywords for collapsed-tree retrieval across all layers"],
  "tree_traversal_cues": ["query cues that should descend toward this leaf"]
}}

Rules:
- Use only node ids supplied in the RAPTOR tree context.
- leaf_node_id must be a level-0 node containing this concept when one is supplied.
- path_to_root should start with the selected leaf and include broader parent/root nodes when supplied.
- leaf_text must be faithful to the concept fields and suitable as a leaf text chunk.
- retrieval_text should summarize the concept's role at that tree level, not just repeat the node summary.
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

# RAPTOR tree context containing this concept

{tree_context}

Return the JSON object now.
