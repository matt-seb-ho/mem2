# Lightrag Case Studies

## Paper

- Title: LightRAG
- Citation: arxiv:2410.05779
- Mechanism: operate.py::kg_query dual-level retrieval (token-overlap instead of vector sim; template relation text instead of LLM summary)

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
- Retriever: lightrag

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
