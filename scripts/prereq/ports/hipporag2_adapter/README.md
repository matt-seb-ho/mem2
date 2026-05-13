# HippoRAG2 Adapted Memory

Paper: From RAG to Memory: Non-Parametric Continual Learning for Large Language Models, arXiv:2502.14802.

This adapter rewrites each ARC concept into the native shape used by the HippoRAG2 port: a PPR passage plus candidate-filter evidence. The existing `hipporag2_filter` retriever can then seed its first-stage PPR with adapted text and score the second-stage filter with query-filter terms, candidate profiles, and evidence statements instead of only flat concept descriptions.

## Artifact

`data/arc_agi/concept_memory/ports/hipporag2_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "hipporag2",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "ppr_passage": "compact passage for PPR seeding",
      "candidate_profile": "why this concept should survive the filter stage",
      "query_filter_terms": ["terms likely to appear in ARC queries"],
      "filter_evidence": [{"claim": "...", "supporting_text": "...", "specificity": "high"}],
      "reject_signals": ["queries where this concept should be filtered out"],
      "rerank_notes": "one sentence"
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/hipporag2_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/hipporag2_adapter/build_adapted_memory.py --force
```

The builder performs one LLM call per concept with three explicit parse/validation attempts. `--smoke` adapts the first five concepts and writes only to `/private/tmp/mem2_per_port_adapters/hipporag2_smoke.json`.
