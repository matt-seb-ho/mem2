# Raptor Case Studies

## Paper

- Title: RAPTOR
- Citation: arxiv:2401.18059
- Mechanism: tree_retriever.py::retrieve_information_collapse_tree + cluster_tree_builder.py::construct_tree (Louvain instead of GMM; template summaries instead of LLM)

## Parity Grade

- Current: faithful
- Source: configs/axes/1.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: retriever
- Builder: unchanged
- Retriever: raptor

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
