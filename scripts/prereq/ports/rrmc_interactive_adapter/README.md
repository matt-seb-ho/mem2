# RRMC Interactive Adapted Memory

Source: `origins/threads/interactive_retrieval/source.md`.

This adapter rewrites each ARC concept into RRMC-style multi-round selector metadata: selector role, round-one and round-two relevance, coverage targets, probe questions, selector updates, convergence signal, routing keywords, and retrieval notes. The retriever can then prioritize concepts by interactive coverage and refinement value rather than only graph degree.

## Artifact

`data/arc_agi/concept_memory/ports/rrmc_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "rrmc_interactive",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "selector_role": "seed_probe",
      "round_1_relevance": 0.8,
      "round_2_relevance": 0.6,
      "coverage_targets": ["..."],
      "probe_plan": [
        {
          "round": 1,
          "probe_question": "...",
          "expected_signal": "...",
          "selector_update": "..."
        }
      ],
      "convergence_signal": "...",
      "routing_keywords": ["..."],
      "retrieval_notes": "..."
    }
  ],
  "stats": {"num_concepts": 270, "num_failures": 0, "estimated_cost_usd": 0.0}
}
```

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
.venv/bin/python scripts/prereq/ports/rrmc_interactive_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/rrmc_interactive_adapter/build_adapted_memory.py
```

The builder performs one LLM call per concept with three retries after the initial attempt. Backoff waits are 2, 4, and 8 seconds. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
