# LILO Adapted Memory

Paper: LILO: Learning Interpretable Libraries by Compressing and Documenting Code, arXiv:2310.19791.

This adapter rewrites each ARC concept into a LILO-like library-growth record: a library profile, a natural-language abstraction proposal, language grounding, abstraction terms, and iterative-growth notes. This is a best-effort partial substrate conversion because the local ARC port does not run LILO's full DreamCoder/Stitch/AutoDoc loop or maintain executable program libraries. The adapted records expose the LLM library-growth shape for retrieval and auditing.

## Substrate Gap

LILO's native substrate is an executable DSL/program library with frontiers, DreamCoder/Stitch compression, and language documentation. The ARC concept memory stores natural-language routines and snippets. This adapter preserves the LL-style abstraction proposal and documentation layer, but it cannot provide executable function expressions, compression rescoring, AutoDoc validation, or a full dual-system LILO run.

## Artifact

`data/arc_agi/concept_memory/ports/lilo_memory_v1.json`

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/lilo_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/lilo_adapter/build_adapted_memory.py --force
```

`--smoke` adapts the first five concepts and writes only to `/private/tmp/mem2_per_port_adapters/lilo_smoke.json`.
