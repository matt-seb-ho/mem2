# MAGMA Adapted Memory

Paper: MAGMA: A Multi-Graph Agentic Memory Architecture, arXiv:2601.03236.

This adapter rewrites each ARC concept into MAGMA's native event-node and typed-view memory form. Each record contains event content, structured attributes, view memberships across semantic, causal, structural, entity, or temporal views, policy hints for query-aware traversal, and a graph-linearization card for grounded context rendering.

## Artifact

`data/arc_agi/concept_memory/ports/magma_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "source_typed_views": "data/arc_agi/concept_memory/shared/magma_typed_views_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "magma",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "event_node": {
        "content": "event-node text",
        "timestamp_hint": "stable ordering hint",
        "attributes": ["operation: extract objects"]
      },
      "view_memberships": [
        {
          "view": "semantic",
          "node_refs": ["concept::extract objects"],
          "edge_refs": ["semantic:routine"],
          "role": "...",
          "traversal_value": "...",
          "query_intents": ["SEMANTIC", "ENTITY"]
        }
      ],
      "anchor_keywords": ["object extraction"],
      "policy_hints": {
        "preferred_views": ["semantic", "entity"],
        "why_signal": "...",
        "when_signal": "...",
        "entity_signal": "..."
      },
      "graph_linearization_card": "<ref:concept> ...",
      "salience_budget": {
        "keep_full": ["..."],
        "summarize_if_needed": ["..."]
      }
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/magma_adapter/build_adapted_memory.py --smoke --ignore-cache
.venv/bin/python scripts/prereq/ports/magma_adapter/build_adapted_memory.py --force --ignore-cache
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
