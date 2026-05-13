# RAPTOR Adapted Memory

Paper: RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval, arXiv:2401.18059.

This adapter rewrites each ARC concept into RAPTOR's native tree substrate: a leaf text chunk, selected leaf node, path-to-root summaries, collapsed-tree keywords, and tree-traversal cues. The retriever can then use per-concept tree placement rather than only a flat concept record plus shared tree summaries.

## Artifact

`data/arc_agi/concept_memory/ports/raptor_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "raptor",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "leaf_node_id": "rt_L0_N000",
      "tree_membership_rationale": "...",
      "leaf_text": "...",
      "path_to_root": [{"level": 0, "node_id": "rt_L0_N000", "summary_role": "...", "retrieval_text": "..."}],
      "collapsed_tree_keywords": ["..."],
      "tree_traversal_cues": ["..."]
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/raptor_adapter/build_adapted_memory.py --force
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
