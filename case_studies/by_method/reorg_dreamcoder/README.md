# Reorg Dreamcoder Case Studies

## Paper

- Title: DreamCoder
- Citation: arxiv:2006.08381
- Mechanism: compression.py::induceGrammar + fragmentGrammar.py::insideOutside (line-level fragment extraction; no OCaml/PyPy; greedy instead of beam)

## Parity Grade

- Current: faithful
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_dreamcoder
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
