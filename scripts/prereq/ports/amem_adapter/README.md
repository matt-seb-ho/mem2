# A-Mem Adapted Memory

Paper: A-Mem: Agentic Memory for LLM Agents, arXiv:2502.12110.

This adapter rewrites each ARC concept into A-Mem's native Zettelkasten note form. Each record contains an atomic note, generated keywords and tags, contextual description, fresh LLM-selected note links, memory-evolution suggestions, and retrieval text. It intentionally does not use the earlier shared `amem_link_graph_v1.json` as output substrate because that artifact contained heuristic fallback links from failed calls.

## Artifact

`data/arc_agi/concept_memory/ports/amem_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "amem",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "note": {
        "content": "atomic note text",
        "timestamp": "stable lifecycle stage",
        "keywords": ["..."],
        "tags": ["..."],
        "contextual_description": "..."
      },
      "zettel_links": [
        {
          "target_concept": "related concept",
          "link_type": "applied_with",
          "rationale": "...",
          "confidence": 0.84
        }
      ],
      "memory_evolution": {
        "context_update": "...",
        "tag_updates": ["..."],
        "neighbor_update_suggestions": [{"target_concept": "...", "suggested_update": "..."}]
      },
      "retrieval_text": "..."
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/amem_adapter/build_adapted_memory.py --smoke --ignore-cache
.venv/bin/python scripts/prereq/ports/amem_adapter/build_adapted_memory.py --force --ignore-cache
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact. There is no heuristic fallback path.
