# mem2 Analysis

`analysis/` is the home for post-run evidence extraction. It consumes durable traces from `case_studies/runs/<run_id>/` and turns them into failure labels, memory-growth records, retrieval telemetry, and provenance reports.

## Relationship to case_studies/

`case_studies/` is the source of truth for prompts, responses, retrieval bundles, parsed attempts, eval records, and call metadata. Analysis modules must treat those files as immutable inputs and write derived outputs elsewhere, usually under a caller-provided output path.

## Modules

- `failure_taxonomy/`: classifies failures into the LN-018 typology. The shipped classifier is a no-LLM placeholder.
- `memory_growth/`: extracts memory-size timelines from run traces and future episodic streams.
- `retrieval_telemetry/`: counts retrieval-bundle usage by concept and method.
- `provenance/`: tracks which task or episode introduced each concept.
- `_shared/`: small utilities for loading case-study runs.

## Success Criteria

A completed analysis pass should be reproducible from a run ID alone, cite every source file it reads, and avoid any provider calls unless a future command explicitly opts into a judge model.
