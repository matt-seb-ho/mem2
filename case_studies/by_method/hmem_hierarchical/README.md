# Hmem Hierarchical Case Studies

## Paper

- Title: H-MEM
- Citation: arxiv:2507.22925
- Mechanism: paper section3 Method - 4-layer multi-layer routing (Domain->Category->Trace->Episode). Kind-based categories substitute for learned domain embeddings; trace groups cluster concepts by shared used_in problems; token-overlap scoring per layer.

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
- Retriever: hmem_hierarchical

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
