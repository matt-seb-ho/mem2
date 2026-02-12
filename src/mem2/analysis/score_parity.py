from __future__ import annotations

from collections import defaultdict
from typing import Any

from mem2.core.entities import EvalRecord


def _detail_step_idx(rec: EvalRecord) -> int:
    md = rec.metadata or {}
    if "global_step_idx" in md:
        return int(md["global_step_idx"])
    return int(rec.attempt_idx)


def flatten_eval_records(records: list[EvalRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        step_idx = _detail_step_idx(rec)
        for detail in rec.train_details + rec.test_details:
            rows.append(
                {
                    "problem_uid": rec.problem_uid,
                    "step_idx": step_idx,
                    "is_train": bool(detail.get("is_train", False)),
                    "pair_idx": int(detail.get("pair_idx", -1)),
                    "correct": bool(detail.get("correct", False)),
                }
            )
    return rows


def official_score_sum(rows: list[dict[str, Any]]) -> float:
    # Arc official score: for each puzzle/test-pair, success if any attempt solves that pair.
    test_rows = [r for r in rows if not r["is_train"] and r["pair_idx"] >= 0]
    if not test_rows:
        return 0.0
    per_pair_solved: dict[tuple[str, int], bool] = defaultdict(bool)
    for row in test_rows:
        key = (row["problem_uid"], row["pair_idx"])
        per_pair_solved[key] = per_pair_solved[key] or row["correct"]

    puzzle_pair_scores: dict[str, list[bool]] = defaultdict(list)
    for (uid, _pair_idx), solved in per_pair_solved.items():
        puzzle_pair_scores[uid].append(solved)

    return float(sum(sum(1 for x in vals if x) / max(1, len(vals)) for vals in puzzle_pair_scores.values()))


def strict_score_sum(
    rows: list[dict[str, Any]],
    *,
    include_train: bool = True,
    step_selection: str = "last",  # all | last
    aggregate_step_method: str = "any",  # any | mean
) -> float:
    if not rows:
        return 0.0
    filtered = rows if include_train else [r for r in rows if not r["is_train"]]
    if not filtered:
        return 0.0

    if step_selection == "last":
        max_step_by_puzzle: dict[str, int] = {}
        for row in filtered:
            uid = row["problem_uid"]
            max_step_by_puzzle[uid] = max(max_step_by_puzzle.get(uid, -1), row["step_idx"])
        filtered = [r for r in filtered if r["step_idx"] == max_step_by_puzzle[r["problem_uid"]]]

    step_ok: dict[tuple[str, int], bool] = defaultdict(lambda: True)
    for row in filtered:
        key = (row["problem_uid"], row["step_idx"])
        step_ok[key] = step_ok[key] and row["correct"]

    per_puzzle: dict[str, list[bool]] = defaultdict(list)
    for (uid, _step_idx), ok in step_ok.items():
        per_puzzle[uid].append(ok)

    if aggregate_step_method == "mean":
        return float(sum(sum(1 for x in vals if x) / max(1, len(vals)) for vals in per_puzzle.values()))
    # default any
    return float(sum(1.0 if any(vals) else 0.0 for vals in per_puzzle.values()))

