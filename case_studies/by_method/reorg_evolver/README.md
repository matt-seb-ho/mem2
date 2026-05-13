# Reorg Evolver Case Studies

## Paper

- Title: EvolveR
- Citation: arxiv:2510.16079
- Mechanism: paper section3.2 maintenance pipeline - semantic dedup of scored principles with quality-metric tiebreak. Token-set Jaccard substitutes embedding similarity; hit/success metric tracks historical effectiveness (distinct from A.7 Memp which prunes, this removes only duplicates).

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_evolver
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
