# Hipporag2 Filter Case Studies

## Paper

- Title: HippoRAG 2
- Citation: arxiv:2502.14802
- Mechanism: src/hipporag/rerank.py::DSPyFilter.rerank (PPR first stage + LLM fact filter; token-overlap fallback when no _meta_edit_provider)

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
- Retriever: hipporag2_filter

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
