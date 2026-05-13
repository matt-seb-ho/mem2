# Mediq Policy Case Studies

## Paper

- Title: MediQ
- Citation: arxiv:2406.00922
- Mechanism: expert.py::Expert + expert_functions.py::fixed_abstention_decision (coverage-based abstention instead of LLM confidence)

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/3.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: retriever
- Builder: unchanged
- Retriever: mediq_policy

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
