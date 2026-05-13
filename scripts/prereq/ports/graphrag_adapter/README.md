# GraphRAG Adapted Memory

Paper: From Local to Global: A GraphRAG Approach to Query-Focused Summarization, arXiv:2404.16130.

This adapter rewrites each ARC concept into GraphRAG's native community-report form: primary community, role within the report, map-reduce context card, summary path through the hierarchy, entity or relationship claims, and query-focus keywords. The retriever can then score and render community reports with concept-level report contributions rather than only flat concept descriptions.

## Artifact

`data/arc_agi/concept_memory/ports/graphrag_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "graphrag",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "primary_community_id": "community_0",
      "community_role": "...",
      "contribution_to_cluster": "...",
      "map_reduce_card": "...",
      "summary_path": [{"level": 0, "community_id": "L0_C000", "role_at_level": "...", "report_connection": "..."}],
      "entity_relationship_claims": [{"claim": "...", "importance": "high"}],
      "query_focus_keywords": ["..."]
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/graphrag_adapter/build_adapted_memory.py --force
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
