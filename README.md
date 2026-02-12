# mem2

`mem2` is the modular migration target for `arc_memo`.

It implements a thin orchestrator + branch contracts architecture so new benchmarks,
provider profiles, memory systems, retrievers, and trajectory/feedback policies are additive.

## Quick Start

```bash
python -m pip install -e .
python -m mem2.cli.run --config configs/experiments/smoke_arc.yaml
```

Outputs are written under `outputs/_runs/...`.

## API keys

By default, provider clients use the current shell environment (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, etc.). Recommended workflow:

```bash
set -a
source .env
set +a
```

If you want file-based loading, pass an explicit `dotenv_path` (or `env_file`) in
`components.provider` config.

## ArcMemo strict compatibility mode

For migration-fidelity runs, use:

```bash
python -m mem2.cli.run --config configs/experiments/arcmemo_arc_strict.yaml
```

This enables `run.strict_arcmemo_compat: true`, which hard-validates inference-critical
ArcMemo settings (provider profile, model, generation config, retry settings) and fails
fast on any drift.

Execution mode is controlled via `run.execution_mode`:

- `sequential`: process one problem at a time within each pass.
- `arc_batch`: build/run all inference jobs for a pass concurrently (ArcMemo-like scheduling).

Strict ArcMemo config uses `arc_batch`.

For logic-parity checks with a cheaper model/provider (same mechanics, different backend/model),
use:

```bash
python -m mem2.cli.run --config configs/experiments/arcmemo_arc_logic_parity_openrouter.yaml
```

## Parity lock harness

Run offline parity checks (prompt/scoring/retry reproducibility):

```bash
python scripts/parity/run_arc_default_parity_lock.py
```

Run with optional live verification (if `OPENAI_API_KEY` is set, or `api_key.txt` exists):

```bash
python scripts/parity/run_arc_default_parity_lock.py --allow-live --live-problems 1
```

Compare prompts/artifacts between completed `arc_memo` and `mem2` runs:

```bash
python scripts/parity/compare_arc_memo_mem2_runs.py \
  --arc-run-dir /path/to/arc_memo/outputs/... \
  --mem2-run-dir /path/to/mem2/outputs/_runs/.../... \
  --out outputs/parity/cross_repo_prompt_parity_report.json
```

Deterministic retry-prompt replay parity (uses ArcMemo run artifacts, independent of live model variance):

```bash
python scripts/parity/replay_retry_prompt_parity.py \
  --arc-run-dir /path/to/arc_memo/outputs/... \
  --out outputs/parity/replay_retry_prompt_parity_report.json
```

Lockstep replay mode (replay ArcMemo pass-1 completions + pass-2 reselected lessons inside mem2):

```bash
python -m mem2.cli.run --config configs/experiments/arcmemo_arc_lockstep_replay.yaml
```

Set `run.lockstep_replay.source_run_dir` to the ArcMemo run you want to mirror.

One-command parity gate (runs lockstep config, compares against ArcMemo run, exits non-zero on mismatch):

```bash
python scripts/parity/prove_lockstep_prompt_parity.py \
  --arc-run-dir /path/to/arc_memo/outputs/... \
  --out outputs/parity/lockstep_prompt_parity_gate_report.json
```

Freeze baseline hashes:

```bash
python scripts/parity/freeze_parity_baseline.py
```
