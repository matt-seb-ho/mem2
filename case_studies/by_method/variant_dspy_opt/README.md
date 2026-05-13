# Variant Dspy Opt Case Studies

## Paper

- Title: DSPy
- Citation: arxiv:2310.03714
- Mechanism: dspy/teleprompt/copro_optimizer.py::COPRO.compile (breadth-then-depth instruction search with proposal-given-attempts history; MDL-proxy scorer stand-in for labeled metric; LLM proposal via _meta_edit_provider, template mutation fallback)

## Parity Grade

- Current: faithful
- Source: configs/axes/4.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: variant_dspy_opt
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
