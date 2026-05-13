# UoT Entropy Adapted Memory

Source: `literature/2402.03271.pdf`.

This adapter rewrites each ARC concept into UoT-style uncertainty-reduction metadata. Each record contains a yes/no candidate question, branch hints for affirmative and negative answers, an expected split ratio, an entropy reward, an information-gain target, simulation-tree role, reward-propagation notes, routing keywords, and retrieval notes.

The retriever uses this artifact when present, preferring records that match the current query and carry balanced, high-reward candidate questions. If the artifact is absent, it falls back to the existing one-step kind-distribution entropy retriever.

## Artifact

`data/arc_agi/concept_memory/ports/uot_memory_v1.json`

Schema:

```json
{
  "schema_version": "1",
  "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
  "model": "deepseek/deepseek-v4-flash",
  "port": "uot_entropy",
  "adapted_concepts": [
    {
      "concept_id": "concept name",
      "uncertainty_state": "...",
      "candidate_question": "...",
      "yes_partition_hint": ["..."],
      "no_partition_hint": ["..."],
      "expected_yes_ratio": 0.5,
      "entropy_reward": 1.0,
      "information_gain_target": "...",
      "simulation_tree_role": "root_candidate",
      "reward_propagation_notes": "...",
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
.venv/bin/python scripts/prereq/ports/uot_entropy_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/uot_entropy_adapter/build_adapted_memory.py
```

The builder performs one LLM call per concept with three retries after the initial attempt. Backoff waits are 2, 4, and 8 seconds. If any concept still fails parsing or validation, the script raises and does not write a partial artifact.
