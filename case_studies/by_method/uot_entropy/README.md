# Uot Entropy Case Studies

## Paper

- Title: UoT
- Citation: arxiv:2402.03271
- Mechanism: src/uot/uot.py::UoTNode.reward_function (damped-Shannon entropy info-gain per round; one-step signal instead of full tree search; no LLM question generation)

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
- Retriever: uot_entropy

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
