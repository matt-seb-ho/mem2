# Reorg Memp Case Studies

## Paper

- Title: Memp
- Citation: arxiv:2508.06433
- Mechanism: ProcedureMem/memory.py::Memory.update (hit/success quality tracking + performance-based pruning at hit>=min_hits AND success/hit < threshold; no round/direct workflow LLM distillation)

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_memp
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
