# mem2 Case Studies

This directory stores labeled, durable traces for small case-study runs. The goal is to preserve the exact prompt, response, retrieval bundle, parsed output, evaluation record, and call metadata for every model call that costs money or wall time.

## Layout

- `runs/<run_id>/`: auto-populated trace store for one run.
- `runs/<run_id>/meta.json`: run config, port label, seed, model, problem count, total cost if available, and call count.
- `runs/<run_id>/problems/<task_id>/iter_<N>/`: per-problem, per-iteration trace files.
- `by_method/<method>/`: curated method view with a README and symlinks to relevant runs.
- `scripts/`: command-line helpers for running, rendering, inspecting, linking, and diffing case studies.
- `_tracer/`: opt-in middleware used by the runner when `components.provider.trace_dir` is configured.

## Trace Files

Each `iter_<N>/` directory may contain:

- `prompt.txt`: first rendered LLM prompt observed for this problem and iteration.
- `response.txt`: first full completion observed for this problem and iteration.
- `retrieval_bundle.json`: retrieved items, hint text, and retrieval metadata.
- `parsed.json`: attempt completions and extracted Python blocks when available.
- `eval.json`: evaluator records for the attempts.
- `call_meta.json`: metadata for the first provider call.
- `llm_calls/call_XXXX/`: one directory per provider call so repeated calls are never overwritten.

## Conventions

- Run IDs should use: `<ISO-time>_<port>_n<N>_seed<S>_<label-slug>`.
- Labels should be short, stable, and human-readable.
- Generated summaries are starting points. Human observations belong in the `Notable observations` section of `summary.md`.
- Do not commit large generated run traces unless the researcher explicitly asks for that run to become a durable artifact.
