# Pathrag Case Studies

## Paper

- Title: PathRAG
- Citation: arxiv:2502.14902
- Mechanism: paper sectionMethod - path-based retrieval with flow-based pruning. Token-overlap seed selection substitutes keyword-to-node match; path reliability = product of co-activation edge weights, distance-decayed; render paths ascending by reliability (paper's 'lost in the middle' mitigation).

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/1.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: retriever
- Builder: unchanged
- Retriever: pathrag

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
