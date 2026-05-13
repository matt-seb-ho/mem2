# MemTree Adapted Memory

Paper: MemTree: A Structured Memory Representation for Efficient Long-Term Context, arXiv:2410.14052.

This adapter rewrites each ARC concept into MemTree's native tree-node form. Each record contains a concept leaf, parent community placement, insertion decision, leaf content, embedding text, path-to-root summaries, collapsed-tree retrieval card, and sibling-group cues. It uses the shared hierarchical reports as the available tree substrate and asks the LLM to place each concept into that hierarchy.

## Artifact

`data/arc_agi/concept_memory/ports/memtree_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "source_hierarchy": "data/arc_agi/concept_memory/shared/hierarchical_reports_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "memtree",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "tree_position": {
        "leaf_node_id": "memtree::concept name",
        "parent_node_id": "L0_C000",
        "depth": 2,
        "insertion_decision": "traverse_deeper",
        "depth_threshold_rationale": "..."
      },
      "node_content": {
        "leaf_content": "...",
        "embedding_text": "...",
        "aggregate_contribution": "..."
      },
      "path_to_root": [{"node_id": "memtree::concept name", "depth": 2, "content_summary": "...", "update_role": "..."}],
      "collapsed_retrieval_card": "...",
      "retrieval_keywords": ["..."],
      "sibling_group": {"sibling_role": "...", "near_sibling_concepts": ["..."]}
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/memtree_adapter/build_adapted_memory.py --smoke --ignore-cache
.venv/bin/python scripts/prereq/ports/memtree_adapter/build_adapted_memory.py --force --ignore-cache
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
