# Alma Style Metaedit Case Studies

## Paper

- Title: ALMA
- Citation: arxiv:2602.07755
- Mechanism: src/models/gpt_abstraction.py-inspired one-shot meta-edit plan + MDL gate. LLM provider wired via SyncMetaEditProviderAdapter (runner.py injects ctx.config['_meta_edit_provider'] after run_context serialization). Falls back to hand_coded_reorg when provider unavailable.

## Parity Grade

- Current: surface-port-only-disclosed
- Source: configs/axes/5.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: alma_style_metaedit
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
