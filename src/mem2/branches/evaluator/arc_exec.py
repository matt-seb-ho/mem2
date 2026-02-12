from __future__ import annotations

import numpy as np

from mem2.analysis.score_parity import (
    flatten_eval_records,
    official_score_sum,
    strict_score_sum,
)
from mem2.core.entities import AttemptRecord, EvalRecord, ProblemSpec, RunContext
from mem2.utils.code_execution import execute_transform, extract_python_block


class ArcExecutionEvaluator:
    name = "arc_exec"

    def __init__(self, require_all_tests: bool = True, timeout_s: float = 2.0):
        self.require_all_tests = bool(require_all_tests)
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _evaluate_on_pairs(
        code: str,
        io_pairs: list[dict],
        timeout_s: float,
        is_train: bool,
    ) -> list[dict]:
        details: list[dict] = []
        for i, pair in enumerate(io_pairs):
            exec_result = execute_transform(code, pair["input"], timeout_s=timeout_s)
            if exec_result["status"] != "ok":
                details.append(
                    {
                        "is_train": is_train,
                        "pair_idx": i,
                        "parsed": True,
                        "correct": False,
                        "error": exec_result.get("error"),
                        "output": None,
                        "expected": pair.get("output"),
                    }
                )
                continue

            output = np.array(exec_result["output"], dtype=int)
            expected = np.array(pair["output"], dtype=int)
            details.append(
                {
                    "is_train": is_train,
                    "pair_idx": i,
                    "parsed": True,
                    "correct": bool(np.array_equal(output, expected)),
                    "error": None,
                    "output": output.tolist(),
                    "expected": expected.tolist(),
                }
            )
        return details

    def evaluate(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
    ) -> list[EvalRecord]:
        records = []
        for idx, attempt in enumerate(attempts):
            code, parsing_error = extract_python_block(attempt.completion)
            train_details: list[dict] = []
            test_details: list[dict] = []

            if parsing_error:
                is_correct = False
                metadata = {"evaluator": self.name, "parsing_error": parsing_error}
            else:
                train_details = self._evaluate_on_pairs(
                    code=code,
                    io_pairs=problem.train_pairs,
                    timeout_s=self.timeout_s,
                    is_train=True,
                )
                test_details = self._evaluate_on_pairs(
                    code=code,
                    io_pairs=problem.test_pairs,
                    timeout_s=self.timeout_s,
                    is_train=False,
                )
                test_corrects = [bool(d["correct"]) for d in test_details]
                is_correct = (
                    all(test_corrects) if self.require_all_tests else any(test_corrects)
                ) if test_corrects else False
                metadata = {"evaluator": self.name, "parsing_error": None}

            records.append(
                EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=is_correct,
                    train_details=train_details,
                    test_details=test_details,
                    metadata=metadata,
                )
            )
        return records

    def aggregate(self, ctx: RunContext, records: list[EvalRecord]) -> dict:
        rows = flatten_eval_records(records)
        official = official_score_sum(rows)
        strict = strict_score_sum(
            rows,
            include_train=True,
            step_selection="last",
            aggregate_step_method="any",
        )

        if not records:
            return {
                "accuracy_per_attempt": 0.0,
                "strict_score": 0.0,
                "official_score": 0.0,
                "strict_solved_puzzles": 0,
                "official_score_sum": 0.0,
                "total_attempts": 0,
                "correct_attempts": 0,
            }

        total = len(records)
        correct = sum(1 for r in records if r.is_correct)

        # strict score (puzzle solved if any attempt is fully correct on test pairs)
        solved_by_puzzle: dict[str, bool] = {}
        official_by_puzzle: dict[str, float] = {}
        test_pair_solved: dict[str, dict[int, bool]] = {}
        test_pair_count: dict[str, int] = {}

        for rec in records:
            solved_by_puzzle.setdefault(rec.problem_uid, False)
            solved_by_puzzle[rec.problem_uid] = solved_by_puzzle[rec.problem_uid] or rec.is_correct

            pair_map = test_pair_solved.setdefault(rec.problem_uid, {})
            for d in rec.test_details:
                idx = int(d.get("pair_idx", -1))
                if idx < 0:
                    continue
                pair_map[idx] = pair_map.get(idx, False) or bool(d.get("correct", False))
            test_pair_count[rec.problem_uid] = max(
                test_pair_count.get(rec.problem_uid, 0), len(rec.test_details)
            )

        for uid, pair_map in test_pair_solved.items():
            n_pairs = max(1, test_pair_count.get(uid, len(pair_map)))
            solved_count = sum(1 for ok in pair_map.values() if ok)
            official_by_puzzle[uid] = solved_count / n_pairs

        return {
            "accuracy_per_attempt": correct / total,
            "official_score": official,
            "strict_score": strict,
            "strict_solved_puzzles": int(sum(1 for ok in solved_by_puzzle.values() if ok)),
            "official_score_sum": float(sum(official_by_puzzle.values())),
            "total_attempts": total,
            "correct_attempts": correct,
        }
