# H-MEM Hierarchical Adapted Memory

Paper: H-MEM, arXiv:2507.22925.

This adapter rewrites each ARC concept into the native H-MEM indexing shape: an ARC domain, semantic category, memory trace group, episode summary, positional routing indices, and routing keywords. The retriever can then descend through the adapted hierarchy instead of relying only on the flat concept-memory record or the older static hierarchy.

## Artifact

`data/arc_agi/concept_memory/ports/hmem_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "hmem_hierarchical",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "domain": "ARC-AGI",
      "category": "semantic category",
      "category_position_index": "L1:category",
      "subcategory": "memory trace group",
      "subcategory_position_index": "L2:trace",
      "memory_trace": {
        "title": "trace title",
        "keywords": ["..."],
        "trace_summary": "..."
      },
      "episode": {
        "summary": "...",
        "grounded_operations": ["..."],
        "when_to_route_here": "..."
      },
      "routing_keywords": ["..."],
      "confidence_weight": 0.9,
      "retrieval_notes": "..."
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
.venv/bin/python scripts/prereq/ports/hmem_hierarchical_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/hmem_hierarchical_adapter/build_adapted_memory.py
```

The builder performs one LLM call per concept with three retries after the initial attempt. Backoff waits are 2, 4, and 8 seconds. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
