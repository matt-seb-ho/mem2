# DreamCoder Adapted Memory

Paper: DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning, arXiv:2006.08381.

This adapter rewrites each ARC concept into a DreamCoder-like frontier/compression record: a program-fragment signature, an invented-primitive candidate, compression roles, and MDL cues. This is a best-effort partial substrate conversion because the local ARC port does not run DreamCoder's enumerator, recognition model, frontiers, or OCaml compressor. The adapted records are useful as retrieval cards and as an audit artifact for the fragment-compression mechanism, not as full DreamCoder execution state.

## Substrate Gap

DreamCoder's native substrate is task frontiers plus typed lambda-program fragments scored by grammar likelihood and MDL. The ARC concept bank has natural-language routines, cues, and implementation snippets. This adapter preserves the compression/abstraction shape but cannot recover executable programs, posterior frontiers, Helmholtz recognition training, or inside-outside grammar updates.

## Artifact

`data/arc_agi/concept_memory/ports/dreamcoder_memory_v1.json`

## Regeneration

```bash
cd /Users/aaronzhfeng/workspace/workstation_00_arc/mem2
set -a && source .env && set +a
.venv/bin/python scripts/prereq/ports/dreamcoder_adapter/build_adapted_memory.py --smoke
.venv/bin/python scripts/prereq/ports/dreamcoder_adapter/build_adapted_memory.py --force
```

`--smoke` adapts the first five concepts and writes only to `/private/tmp/mem2_per_port_adapters/dreamcoder_smoke.json`.
