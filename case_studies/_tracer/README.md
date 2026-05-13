# Case Study Tracer

The tracer is opt-in. Normal mem2 runs do not write case-study traces.

To enable tracing, set `components.provider.trace_dir` to a run directory such as `case_studies/runs/2026-05-13T10-30Z_graphrag_n3_seed42_smoke`. The provider wrapper records every provider call under `problems/<task_id>/iter_<N>/llm_calls/`, while the runner records retrieval, parsed attempt, and evaluation files for the same problem and iteration.

The runner sets the current task and iteration with context variables before retrieval and inference. The provider wrapper reads those variables at call time, so concurrent jobs keep separate trace locations.
