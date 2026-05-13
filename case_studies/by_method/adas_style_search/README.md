# Adas Style Search Case Studies

## Paper

- Title: ADAS: Automated Design of Agentic Systems
- Citation: arxiv:2408.08435
- Mechanism: _arc/search.py meta-agent reflexion loop (3-round reflexion buffer; same LLM-optional pattern as F.2 - falls back to F.2 then A.1 if no provider wired)

## Parity Grade

- Current: surface-port-only-disclosed
- Source: configs/axes/5.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: adas_style_search
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
