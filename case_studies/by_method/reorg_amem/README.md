# Reorg Amem Case Studies

## Paper

- Title: A-MEM
- Citation: arxiv:2502.12110
- Mechanism: memory_layer.py::AgenticMemorySystem.process_memory + consolidate_memories (per-note evolution + evo_threshold schedule; no Zettelkasten context rewrite; no tag-graph broadcast)

## Parity Grade

- Current: partial-with-disclosed-gap
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_amem
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
