# Hipporag Ppr Case Studies

## Paper

- Title: HippoRAG / HippoRAG 2
- Citation: arxiv:2405.14831, 2502.14802
- Mechanism: HippoRAG.py::run_ppr + graph_search_with_fact_entities (PPR core only; no OpenIE / DPR)

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
- Retriever: hipporag_ppr

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
