# HippoRAG PPR Adapted Memory

Paper: HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models, arXiv:2405.14831.

This adapter rewrites each ARC concept into the native HippoRAG indexing shape: a compact passage, entity or noun-phrase nodes, OpenIE-style triples, query-node terms, and node-specificity hints. The retriever can then seed Personalized PageRank from entities and triples instead of treating the original flat concept description as the only memory record.

## Artifact

`data/arc_agi/concept_memory/ports/hipporag_ppr_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "hipporag_ppr",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "passage_text": "HippoRAG-style passage",
      "entity_mentions": [{"text": "...", "type": "operation", "role": "...", "supporting_text": "..."}],
      "triples": [{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9, "supporting_text": "..."}],
      "query_node_terms": ["..."],
      "node_specificity_hints": [{"node": "...", "specificity": "high", "reason": "..."}],
      "retrieval_notes": "..."
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/hipporag_ppr_adapter/build_adapted_memory.py --force
```

The builder performs one LLM call per concept with three explicit retry attempts. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
