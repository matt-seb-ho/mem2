# MediQ Policy Adapted Memory

Paper: MediQ, arXiv:2406.00922.

This adapter rewrites each ARC concept into MediQ-style interactive policy metadata: initial assessment, question type, missing-information targets, atomic follow-up questions, expected information gain, abstention policy, evidence integration, and routing keywords. The retriever can then prefer concepts that ask useful targeted questions under incomplete ARC information instead of treating every concept as a flat context item.

## Artifact

`data/arc_agi/concept_memory/ports/mediq_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "mediq_policy",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "initial_assessment": "...",
      "question_type": "object_property",
      "missing_information_targets": ["..."],
      "atomic_question_templates": ["..."],
      "expected_info_gain": 0.8,
      "abstention_policy": {
        "ask_when": "...",
        "commit_when": "...",
        "confidence_threshold_hint": 0.7
      },
      "evidence_integration": "...",
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
.venv/bin/python scripts/prereq/ports/mediq_policy_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/mediq_policy_adapter/build_adapted_memory.py
```

The builder performs one LLM call per concept with three retries after the initial attempt. Backoff waits are 2, 4, and 8 seconds. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
