# mem2 Benchmarks

`benchmarks/` is the planning surface for running the same memory-builder, retriever, router, and inference stack across multiple task families. ARC is the active implementation today; the placeholder directories describe how the other datasets already present under `data/` should become first-class benchmark adapters.

## Naming Conventions

- One directory per benchmark family, using the dataset name or the intended stream type.
- Active adapters live at `benchmarks/<name>/`.
- Future adapters live under `benchmarks/_placeholder/<name>/` until a config, task adapter, evaluator, and smoke test exist.
- Run outputs stay in `outputs/` and trace outputs stay in `case_studies/runs/`; benchmark directories are metadata and adapter homes, not result stores.

## Adding a Benchmark

1. Document the data source, schema, and scoring contract in the benchmark README.
2. Add or wire the task adapter, benchmark adapter, evaluator, and smoke config.
3. Add a minimal case-study run using `case_studies/scripts/run_case_study.py` once the provider path is safe.
4. Update `_index.md` with runnable conditions and the latest smoke or case-study run.

## Success Criteria

Phase G on a benchmark means at least one baseline and one memory-augmented condition can run end to end, every provider call is captured in `case_studies/`, and analysis stubs can read the run traces without benchmark-specific hacks.
