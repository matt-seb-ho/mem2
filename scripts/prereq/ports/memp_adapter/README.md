# Memp Adapted Memory

Paper: MemP: Exploring Agent Procedural Memory, arXiv:2508.06433.

This adapter rewrites each ARC concept into a Memp-like procedural-memory card: workflow steps, success conditions, failure or adjustment signals, procedure terms, and hit/success tracking notes. This is a best-effort partial substrate conversion because the local ARC port does not distill full agent trajectories into procedural memory. The adapted records expose the procedural-memory shape for retrieval and auditing.

## Substrate Gap

Memp's native substrate is an evolving repository of procedural memories distilled from full agent trajectories with build, retrieval, update, and adjustment policies. ARC concept memory lacks trajectory transcripts, step-level action histories, and online self-learning state. This adapter preserves the procedural card and update-signal shape, but it cannot reproduce offline/online trajectory distillation or full memory-adjustment workflows.

## Artifact

`data/arc_agi/concept_memory/ports/memp_memory_v1.json`

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/memp_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/memp_adapter/build_adapted_memory.py --force
```

`--smoke` adapts the first five concepts and writes only to `/private/tmp/mem2_per_port_adapters/memp_smoke.json`.
