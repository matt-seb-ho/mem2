# Reorg Memtree Case Studies

## Paper

- Title: MemTree
- Citation: arxiv:2410.14052
- Mechanism: paper section3 Method - tree structure [content, embedding, parent, children, depth]; root-to-leaf traversal with similarity routing; ancestor summary aggregation. Co-activation proxy for embedding sim; template summaries fall back when no _meta_edit_provider.

## Parity Grade

- Current: partial-with-disclosed-gap
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_memtree
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
