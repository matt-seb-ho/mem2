# Corpus Hipporag Init Case Studies

## Paper

- Title: HippoRAG / HippoRAG 2
- Citation: arxiv:2405.14831, 2502.14802
- Mechanism: NOT YET PORTED - needs (a) corpus text acquisition + (b) LLM OpenIE at init time

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/6.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: arcmemo_ps
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
