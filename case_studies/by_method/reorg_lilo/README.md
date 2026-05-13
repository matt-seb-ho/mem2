# Reorg Lilo Case Studies

## Paper

- Title: LILO
- Citation: arxiv:2310.19791
- Mechanism: src/models/gpt_abstraction.py::GPTLibraryLearner.generate_abstraction (iterative-abstraction-proposal; schema validation from _parse_completion + check_valid; no DreamCoder/Stitch wrap)

## Parity Grade

- Current: partial-with-disclosed-gap
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_lilo
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
