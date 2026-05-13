# Reorg Lrll Case Studies

## Paper

- Title: LRLL
- Citation: arxiv:2406.18746
- Mechanism: paper sectionIII Method - wake-sleep cycle, experience-filtered abstraction. Simulator self-verification is substituted with success-rate filter on used_in outcomes; DreamCoder fragment extraction runs on the filtered slice. Distinct from A.2 (unfiltered), A.7 (prunes post-hoc).

## Parity Grade

- Current: reduced-but-honest
- Source: configs/axes/2.yaml

## What We Adapted for Faithfulness

- Substrate(s) built: TODO
- Wiring changes: TODO
- Validation runs: see `runs/` directory

## Method Wiring

- Override group: builder
- Builder: reorg_lrll
- Retriever: unchanged

## Runs

This section is updated by `case_studies/scripts/link_to_method.py`.
