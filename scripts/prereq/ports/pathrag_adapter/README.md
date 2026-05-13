# PathRAG Adapted Memory

Paper: PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths, arXiv:2502.14902.

This adapter rewrites each ARC concept into PathRAG's native retrieval unit: query keywords, graph node chunks, graph edge chunks, and textual relational paths. The retriever can then present relation-preserving paths instead of only a flat set of concept descriptions.

## Artifact

`data/arc_agi/concept_memory/ports/pathrag_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "pathrag",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "query_keywords": ["..."],
      "path_nodes": [{"node_id": "n1", "label": "...", "text_chunk": "...", "node_type": "operation"}],
      "entity_paths": [
        {
          "path_id": "p1",
          "nodes": ["n1", "n2"],
          "edges": [{"src": "n1", "dst": "n2", "relation": "...", "text_chunk": "..."}],
          "textual_path": "...",
          "reliability_hint": 0.9,
          "pruning_rationale": "..."
        }
      ],
      "answer_generation_notes": "..."
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/pathrag_adapter/build_adapted_memory.py --force
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
