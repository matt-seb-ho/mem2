"""Subprocess fan-out wrapper for ablation_matrix.py.

Launches multiple axes in parallel as independent `ablation_matrix.py`
subprocesses, so Phase-1 stages 4b/4c wall-clock scales with max-parallel
rather than axis count.

Usage::

    python scripts/sweeps/ablation_matrix_parallel.py \\
        --axes 1,2,3,4,5,6 --step 4b --max-parallel 6 --stagger 5

The parallel wrapper:
  1. Runs strict-parity lock as precondition (fails fast if parity is red).
  2. Spawns one child per axis, up to --max-parallel at a time.
  3. Staggers child starts by --stagger seconds to avoid burst-429s from
     the LLM provider warmup path.
  4. Streams child stdout/stderr to per-axis log files.
  5. Aborts remaining children if any child exits with non-zero AND
     --abort-on-error is set (default: off; per-axis failures isolated).
  6. Runs `aggregate_axis.py` for each axis as its subprocess completes.

Per researcher (2026-04-21): OpenRouter multiplexes DeepSeek V3.2 across
many backend providers, so high cross-axis concurrency is acceptable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_parity_precondition() -> bool:
    print("[parallel] precondition: running strict-parity lock...", flush=True)
    cp = subprocess.run(
        [sys.executable, "scripts/parity/run_arc_default_parity_lock.py"],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        print("[parallel] PARITY LOCK FAILED — aborting. Stdout/stderr:")
        print(cp.stdout[-2000:])
        print(cp.stderr[-2000:])
        return False
    tail = cp.stdout.strip().split("\n")[-3:]
    for line in tail:
        print(f"[parallel] {line}")
    return True


def spawn_axis(axis: str, step: str, extra_args: list[str], log_dir: Path) -> tuple[str, int, Path]:
    log_path = log_dir / f"axis_{axis}.log"
    print(f"[parallel] [{axis}] spawning -> {log_path}", flush=True)
    with log_path.open("w") as log_fh:
        cp = subprocess.run(
            [sys.executable, "scripts/sweeps/ablation_matrix.py",
             "--axis", axis, "--step", step, *extra_args],
            stdout=log_fh, stderr=subprocess.STDOUT,
        )
    print(f"[parallel] [{axis}] done (rc={cp.returncode})", flush=True)
    return (axis, cp.returncode, log_path)


def aggregate_all(step: str, log_dir: Path) -> int:
    """Run aggregator ONCE with --all after all axis subprocesses finish.
    Writes the consolidated stage file (e.g. 65_phase1_step4b_results_*.md)."""
    print(f"[parallel] aggregating all axes (step {step})...", flush=True)
    log_path = log_dir / f"aggregate_all_step_{step}.log"
    with log_path.open("w") as log_fh:
        cp = subprocess.run(
            [sys.executable, "scripts/sweeps/aggregate_axis.py",
             "--step", step, "--all"],
            stdout=log_fh, stderr=subprocess.STDOUT,
        )
    return cp.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axes", default="1,2,3,4,5,6",
                   help="comma-separated axis names (default: all six, execution-priority order)")
    p.add_argument("--step", required=True, choices=["4a", "4b", "4c"],
                   help="stage preset passed through to ablation_matrix.py")
    p.add_argument("--max-parallel", type=int, default=6,
                   help="max concurrent axis subprocesses")
    p.add_argument("--stagger", type=float, default=5.0,
                   help="seconds between child-spawn starts to avoid burst-429s")
    p.add_argument("--abort-on-error", action="store_true",
                   help="kill remaining children if any child exits non-zero")
    p.add_argument("--skip-parity-check", action="store_true",
                   help="skip strict-parity precondition (use with care)")
    p.add_argument("--log-dir", default="outputs/phase1_sweeps/_parallel_logs",
                   help="where per-axis stdout/stderr get streamed")
    p.add_argument("passthrough", nargs="*",
                   help="extra args forwarded to ablation_matrix.py verbatim")
    args = p.parse_args()

    if not args.skip_parity_check:
        if not run_parity_precondition():
            sys.exit(2)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    log_dir = Path(args.log_dir) / f"step_{args.step}"
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    failed_axes: list[str] = []
    passed_axes: list[str] = []

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        future_to_axis: dict = {}
        for i, ax in enumerate(axes):
            if i > 0:
                time.sleep(args.stagger)
            future_to_axis[pool.submit(spawn_axis, ax, args.step, args.passthrough, log_dir)] = ax

        for future in as_completed(future_to_axis):
            ax = future_to_axis[future]
            try:
                axis, rc, log_path = future.result()
                if rc == 0:
                    passed_axes.append(axis)
                else:
                    failed_axes.append(axis)
                    print(f"[parallel] [{axis}] FAILED (rc={rc}); log at {log_path}")
                    if args.abort_on_error:
                        for other_future in future_to_axis:
                            other_future.cancel()
                        break
            except Exception as exc:
                failed_axes.append(ax)
                print(f"[parallel] [{ax}] raised {type(exc).__name__}: {exc}")

    # Aggregate once after all axes finish (writes consolidated stage file)
    if passed_axes:
        agg_rc = aggregate_all(args.step, log_dir)
        if agg_rc != 0:
            print(f"[parallel] aggregator failed (rc={agg_rc}); see {log_dir}/aggregate_all_step_{args.step}.log")

    dt = time.time() - t0
    print(f"\n[parallel] === summary (step {args.step}, {dt:.1f}s) ===")
    print(f"[parallel] passed: {sorted(passed_axes) or '—'}")
    print(f"[parallel] failed: {sorted(failed_axes) or '—'}")
    sys.exit(0 if not failed_axes else 1)


if __name__ == "__main__":
    main()
