# Magma Multigraph Case Studies

## Paper

- Title: MAGMA
- Citation: arxiv:2601.03236
- Mechanism: paper section3 Method - 4 orthogonal relational views (semantic/temporal/causal/entity) + adaptive traversal policy. Views built over same ConceptGraph by filtering edge kinds + synthesizing causal view from used_in; template policy ranks by query-token hit count per view.

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
- Retriever: magma_multigraph

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
