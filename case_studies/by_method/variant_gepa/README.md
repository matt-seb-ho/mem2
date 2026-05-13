# Variant Gepa Case Studies

## Paper

- Title: GEPA
- Citation: arxiv:2507.19457
- Mechanism: dspy/teleprompt/gepa/gepa.py::GEPA (population + tournament selection + crossover + reflective-mutation structure; MDL-proxy scorer stand-in; LLM reflective proposal via _meta_edit_provider, template mutation+crossover fallback)

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/4.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: variant_gepa
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
