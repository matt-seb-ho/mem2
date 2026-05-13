# Colbert Rerank Case Studies

## Paper

- Title: ColBERT
- Citation: arxiv:2004.12832
- Mechanism: colbert/modeling/colbert.py::colbert_score_reduce (MaxSim per-query-token aggregation; lexical exact-match similarity substitutes BERT token embeddings; two-stage ps_topk-expand + MaxSim rerank instead of end-to-end index)

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
- Retriever: colbert_rerank

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
