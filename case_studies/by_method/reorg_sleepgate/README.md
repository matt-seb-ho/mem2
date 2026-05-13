# Reorg Sleepgate Case Studies

## Paper

- Title: SleepGate
- Citation: arxiv:2603.14517
- Mechanism: paper mechanisms adapted from KV-cache level to concept memory - conflict-aware temporal tagger = authorship_lineage + token-overlap; forgetting gate = eviction rule (older->newer supersession with coverage inclusion); consolidation = supersedes-marker on successor. Distinct from A.6 (links), A.7 (performance prune), A.8 (dedup) by triggering on temporal-supersession signal.

## Parity Grade

- Current: wrong-disclosed
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_sleepgate
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
