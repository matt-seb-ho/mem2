# Graphrag Case Studies

## Paper

- Title: GraphRAG (Microsoft)
- Citation: arxiv:2404.16130
- Mechanism: global_search community-report retrieval (Louvain instead of Leiden; template summaries instead of LLM map-reduce)

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
- Retriever: graphrag

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
