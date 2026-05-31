"""Extract tidy per-(run, puzzle, attempt, retry) records for future ensembling.

Every eval run already persists the full solve tree in
`iteration_<pass>/solution_trees.json`:
    prompt_branches[branch] -> threads[thread] -> steps[] (each step = one retry/pass)
with each step's `completion`, `parsing_error`, `train_results`, `test_results`.

This script flattens that into one JSONL row per solve step across all passes, so
downstream code can compute arbitrary ensembling (pass@k over independent samples,
majority vote, oracle/any-correct, first-correct, per-retry curves) WITHOUT re-walking
the trees. Width = independent attempts (n / branches / repeated runs); depth = retries
(passes / steps).

Row schema (JSONL):
  {run, run_type, run_dir, puzzle, pass_idx, branch, thread, step_idx,
   train_ok, n_train, n_train_ok, test_ok, n_test, n_test_ok,
   parsing_error, completion}

Usage:
  python scripts/extract_attempts.py --runs eval100_baseline eval100_arcmemo ...
  python scripts/extract_attempts.py --glob 'eval100_*'           # all eval runs
  # writes <run_dir>/attempt_records.jsonl per run + a combined file if --out given
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def _ok(results):
    results = results or []
    n_ok = sum(1 for r in results if r.get("correct"))
    return (len(results) > 0 and n_ok == len(results)), len(results), n_ok


def extract_run(run_dir: str) -> list[dict]:
    run_dir = run_dir.rstrip("/")
    run_type = "?"
    fc = Path(run_dir) / "frozen_config.json"
    if fc.exists():
        try:
            run_type = json.loads(fc.read_text())["run"]["run_type"]
        except Exception:
            pass
    rows: list[dict] = []
    for it in sorted(glob.glob(run_dir + "/iteration_*")):
        pass_idx = int(os.path.basename(it).split("_")[1])
        st_path = Path(it) / "solution_trees.json"
        if not st_path.exists():
            continue
        trees = json.loads(st_path.read_text())
        for uid, tree in trees.items():
            for bkey, branch in tree.get("prompt_branches", {}).items():
                for tkey, thread in branch.get("threads", {}).items():
                    for step in thread.get("steps", []):
                        train_ok, n_train, n_train_ok = _ok(step.get("train_results"))
                        test_ok, n_test, n_test_ok = _ok(step.get("test_results"))
                        rows.append({
                            "run": os.path.basename(run_dir),
                            "run_type": run_type,
                            "run_dir": run_dir,
                            "puzzle": uid,
                            "pass_idx": pass_idx,
                            "branch": bkey,
                            "thread": tkey,
                            "step_idx": step.get("step_idx"),
                            "train_ok": train_ok,
                            "n_train": n_train,
                            "n_train_ok": n_train_ok,
                            "test_ok": test_ok,
                            "n_test": n_test,
                            "n_test_ok": n_test_ok,
                            "parsing_error": step.get("parsing_error"),
                            "completion": step.get("completion") or "",
                        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[],
                    help="run-family names under outputs/_runs/ (e.g. eval100_baseline)")
    ap.add_argument("--glob", default=None, help="glob over run families, e.g. 'eval100_*'")
    ap.add_argument("--root", default="outputs/_runs")
    ap.add_argument("--out", default=None, help="optional combined JSONL across all runs")
    args = ap.parse_args()

    families = list(args.runs)
    if args.glob:
        families += [os.path.basename(p) for p in glob.glob(f"{args.root}/{args.glob}")]
    families = sorted(set(families))

    combined: list[dict] = []
    for fam in families:
        for run_dir in sorted(glob.glob(f"{args.root}/{fam}/*/")):
            run_dir = run_dir.rstrip("/")
            if not os.path.exists(run_dir + "/frozen_config.json"):
                continue
            rows = extract_run(run_dir)
            if not rows:
                continue
            out_path = Path(run_dir) / "attempt_records.jsonl"
            with out_path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            combined += rows
            n_puz = len({r["puzzle"] for r in rows})
            print(f"{fam}/{os.path.basename(run_dir)}: {len(rows)} step-records "
                  f"over {n_puz} puzzles -> {out_path}")

    if args.out and combined:
        with open(args.out, "w") as f:
            for r in combined:
                f.write(json.dumps(r) + "\n")
        print(f"combined: {len(combined)} records -> {args.out}")


if __name__ == "__main__":
    main()
