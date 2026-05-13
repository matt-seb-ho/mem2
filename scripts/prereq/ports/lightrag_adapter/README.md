# LightRAG Adapted Memory

Source: `literature/2410.05779.pdf`.

This adapter rewrites each ARC concept into LightRAG-style dual-level graph metadata. Each record contains local entity membership, global relationship membership, low-level and high-level keywords, entity and relation value summaries, one-hop neighbors, chunk reference, and retrieval notes.

The builder uses read-only context from `data/arc_agi/concept_memory/shared/lightrag_embed_v1.json`, `entity_graph_v1.json`, and `openie_facts_v1.json`. The generated port artifact is separate from those shared substrates.

## Artifact

`data/arc_agi/concept_memory/ports/lightrag_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "lightrag",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "local_entities": [{"mention": "...", "entity_type": "...", "entity_summary": "..."}],
      "global_relationships": [{"relation": "...", "target_concept": "...", "relation_summary": "...", "strength": 0.8}],
      "low_level_keywords": ["..."],
      "high_level_keywords": ["..."],
      "entity_value_summary": "...",
      "relation_value_summary": "...",
      "one_hop_neighbors": ["..."],
      "chunk_reference": "...",
      "retrieval_notes": "..."
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
.venv/bin/python scripts/prereq/ports/lightrag_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/lightrag_adapter/build_adapted_memory.py
```

The builder performs one LLM call per concept with three retries after the initial attempt. Backoff waits are 2, 4, and 8 seconds. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
