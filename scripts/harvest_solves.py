"""Phase 2 — harvest correct, on-policy solves from a vanilla solve run.

Reads every ``iteration_*/solution_trees.json`` in a run dir, and for each puzzle
finds the best solving step. A step is "train-correct" if all train pairs pass;
"test-correct" if all test pairs also pass. Per decision D1, the primary induction
pool is **train-correct** solves (what the solver itself can verify); test-correctness
is recorded alongside.

Output: ``<run_dir>/induction/solved_seeds.json``
  {uid: {code, completion, train_ok, test_ok, iteration, n_train, n_test, n_train_ok, n_test_ok}}

Usage:
    python scripts/harvest_solves.py --run-dir outputs/_runs/onpolicy_solve_barc/<hash>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mem2.utils.code_execution import extract_python_block


def _all_correct(results: list[dict]) -> tuple[bool, int]:
    results = results or []
    n_ok = sum(1 for r in results if r.get("correct"))
    return (len(results) > 0 and n_ok == len(results)), n_ok


def _iter_steps(tree: dict):
    for branch in tree.get("prompt_branches", {}).values():
        for thread in branch.get("threads", {}).values():
            for step in thread.get("steps", []):
                yield step


def harvest(run_dir: Path) -> dict[str, dict]:
    iter_dirs = sorted(
        run_dir.glob("iteration_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not iter_dirs:
        raise SystemExit(f"No iteration_* dirs under {run_dir}")

    best: dict[str, dict] = {}
    for it in iter_dirs:
        st_path = it / "solution_trees.json"
        if not st_path.exists():
            continue
        it_idx = int(it.name.split("_")[1])
        trees = json.loads(st_path.read_text())
        for uid, tree in trees.items():
            for step in _iter_steps(tree):
                if step.get("parsing_error"):
                    continue
                completion = step.get("completion") or ""
                train_ok, n_train_ok = _all_correct(step.get("train_results"))
                test_ok, n_test_ok = _all_correct(step.get("test_results"))
                n_train = len(step.get("train_results") or [])
                n_test = len(step.get("test_results") or [])
                # rank: prefer train_ok, then also test_ok, then earlier iteration
                rank = (train_ok, train_ok and test_ok, -it_idx)
                cur = best.get(uid)
                if cur is None or rank > cur["_rank"]:
                    code, _ = extract_python_block(completion)
                    best[uid] = {
                        "_rank": rank,
                        "uid": uid,
                        "code": code,
                        "completion": completion,
                        "train_ok": train_ok,
                        "test_ok": test_ok,
                        "iteration": it_idx,
                        "n_train": n_train,
                        "n_test": n_test,
                        "n_train_ok": n_train_ok,
                        "n_test_ok": n_test_ok,
                    }
    for v in best.values():
        v.pop("_rank", None)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="defaults to <run_dir>/induction/solved_seeds.json")
    ap.add_argument("--train-only", action="store_true",
                    help="emit only train-correct solves (the D1 primary pool)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    best = harvest(run_dir)

    train_ok = {u: v for u, v in best.items() if v["train_ok"]}
    full_ok = {u: v for u, v in train_ok.items() if v["test_ok"]}
    pool = train_ok if args.train_only else best

    out = Path(args.out) if args.out else run_dir / "induction" / "solved_seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=2))

    print(f"puzzles seen:        {len(best)}")
    print(f"train-correct (D1):  {len(train_ok)}")
    print(f"  also test-correct: {len(full_ok)}")
    print(f"emitted ({'train-only' if args.train_only else 'all'}): {len(pool)} -> {out}")


if __name__ == "__main__":
    main()
