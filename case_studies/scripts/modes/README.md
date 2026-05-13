# Case-Study Modes

Mode scripts read persisted traces from `case_studies/runs/<run_id>/` and write derived markdown reports under `analyses/`. They never call an LLM and never mutate the raw trace files.

## Scripts

- `error_analysis.py`: failed-problem review for one run.
- `comparative.py`: side-by-side review for two or more runs.
- `counterfactual.py`: dry-run bundle manipulation plan for one run.
- `adversarial.py`: placeholder for future ARC input perturbation studies.
- `mechanistic_attribution.py`: placeholder for future retrieval-to-response attribution.
- `provenance_load_bearing.py`: placeholder for future load-bearing concept provenance.
- `phase_shift_envelope.py`: placeholder for future repeated-run stability reports.

Use `case_studies/MODES.md` for the research motivation and expected inputs for each mode.
