from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from compare_arc_memo_mem2_runs import compare_runs
from mem2.io.json_io import write_json


def _latest_run_dir(run_root: Path, run_type: str) -> Path:
    base = run_root / run_type
    if not base.exists():
        raise FileNotFoundError(f"Run-type directory not found: {base}")
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No runs found under: {base}")
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _run_mem2(mem2_root: Path, config_path: Path) -> None:
    cmd = [sys.executable, "-m", "mem2.cli.run", "--config", str(config_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(mem2_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"mem2 run failed with exit code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lockstep prompt parity gate (ArcMemo vs mem2) with explicit pass/fail."
    )
    parser.add_argument("--arc-run-dir", required=True, help="Path to ArcMemo run directory")
    parser.add_argument(
        "--mem2-root",
        default="/root/workspace/mem2",
        help="Path to mem2 repo root",
    )
    parser.add_argument(
        "--mem2-config",
        default="/root/workspace/mem2/configs/experiments/arcmemo_arc_lockstep_replay.yaml",
        help="mem2 config for lockstep replay",
    )
    parser.add_argument(
        "--mem2-run-dir",
        default=None,
        help="Optional explicit mem2 run dir. If unset, latest run under run-type is used.",
    )
    parser.add_argument(
        "--run-root",
        default="outputs/_runs",
        help="Run root under mem2 root used to discover latest run dir",
    )
    parser.add_argument(
        "--run-type",
        default="arcmemo_arc_lockstep_replay",
        help="run_type directory name under run root",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip launching mem2 run and only compare artifacts",
    )
    parser.add_argument(
        "--out",
        default="outputs/parity/lockstep_prompt_parity_gate_report.json",
        help="Output JSON path for gate report",
    )
    args = parser.parse_args()

    mem2_root = Path(args.mem2_root).expanduser().resolve()
    arc_run_dir = Path(args.arc_run_dir).expanduser().resolve()
    mem2_config = Path(args.mem2_config).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not args.skip_run:
        _run_mem2(mem2_root=mem2_root, config_path=mem2_config)

    if args.mem2_run_dir:
        mem2_run_dir = Path(args.mem2_run_dir).expanduser().resolve()
    else:
        run_root = Path(args.run_root)
        if not run_root.is_absolute():
            run_root = (mem2_root / run_root).resolve()
        mem2_run_dir = _latest_run_dir(run_root=run_root, run_type=args.run_type)

    report = compare_runs(arc_run_dir=arc_run_dir, mem2_run_dir=mem2_run_dir)
    step_rows = list(report.get("step_comparison", []))
    equal_steps = sum(1 for row in step_rows if bool(row.get("prompt_equal", False)))
    total_steps = len(step_rows)

    gate = {
        "pass": bool(
            total_steps > 0
            and equal_steps == total_steps
            and report.get("arc_prompt_count") == report.get("mem2_prompt_count")
        ),
        "equal_steps": equal_steps,
        "total_steps": total_steps,
        "arc_prompt_count": int(report.get("arc_prompt_count", 0)),
        "mem2_prompt_count": int(report.get("mem2_prompt_count", 0)),
        "arc_run_dir": str(arc_run_dir),
        "mem2_run_dir": str(mem2_run_dir),
    }
    report["parity_gate"] = gate

    write_json(out_path, report)
    print(f"report -> {out_path}")
    print(f"prompt_equal_steps: {equal_steps}/{total_steps}")
    print("parity_gate_pass:", gate["pass"])

    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
