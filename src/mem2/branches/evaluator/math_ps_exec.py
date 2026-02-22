from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any

from mem2.core.entities import AttemptRecord, EvalRecord, ProblemSpec, RunContext
from mem2.utils.code_execution import extract_python_block


def _worker_exec_solve(code: str, queue: mp.Queue) -> None:
    """Execute code in an isolated process and call ``solve()``."""
    try:
        import math
        import itertools
        import functools
        import collections
        from fractions import Fraction

        local_ns: dict[str, Any] = {
            "math": math,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
            "Fraction": Fraction,
        }
        exec(code, local_ns, local_ns)
        fn = local_ns.get("solve")
        if not callable(fn):
            queue.put({
                "status": "error",
                "error": "function lookup error: expected callable `solve`.",
                "output": None,
            })
            return

        result = fn()
        queue.put({"status": "ok", "error": None, "output": result})
    except Exception as exc:
        queue.put({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
            "output": None,
        })


def execute_solve(code: str, timeout_s: float = 10.0) -> dict[str, Any]:
    """Run extracted code, call ``solve()``, return result dict."""
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker_exec_solve, args=(code, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "error", "error": f"timeout error (> {timeout_s}s)", "output": None}
    if queue.empty():
        return {"status": "error", "error": "execution error: empty worker response", "output": None}
    return queue.get()


class MathPsExecutionEvaluator:
    """Evaluate math-PS attempts by executing ``solve()`` and comparing to ground truth."""

    name = "math_ps_exec"

    def __init__(self, timeout_s: float = 10.0, **kwargs):
        self.timeout_s = float(timeout_s)

    def evaluate(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
    ) -> list[EvalRecord]:
        expected = problem.metadata.get("answer_int")
        records = []
        for idx, attempt in enumerate(attempts):
            code, parsing_error = extract_python_block(attempt.completion)

            if parsing_error:
                records.append(EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=False,
                    train_details=[],
                    test_details=[{
                        "is_train": False,
                        "pair_idx": 0,
                        "correct": False,
                        "error": parsing_error,
                        "output": None,
                        "expected": expected,
                    }],
                    metadata={"evaluator": self.name, "parsing_error": parsing_error},
                ))
                continue

            exec_result = execute_solve(code, timeout_s=self.timeout_s)

            if exec_result["status"] != "ok":
                records.append(EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=False,
                    train_details=[],
                    test_details=[{
                        "is_train": False,
                        "pair_idx": 0,
                        "correct": False,
                        "error": exec_result.get("error"),
                        "output": None,
                        "expected": expected,
                    }],
                    metadata={"evaluator": self.name, "parsing_error": None,
                              "exec_error": exec_result.get("error")},
                ))
                continue

            output = exec_result["output"]
            # Attempt integer comparison
            try:
                is_correct = int(output) == int(expected)
            except (TypeError, ValueError, OverflowError):
                is_correct = str(output).strip() == str(expected).strip()

            records.append(EvalRecord(
                problem_uid=problem.uid,
                attempt_idx=idx,
                is_correct=is_correct,
                train_details=[],
                test_details=[{
                    "is_train": False,
                    "pair_idx": 0,
                    "correct": is_correct,
                    "error": None,
                    "output": output,
                    "expected": expected,
                }],
                metadata={"evaluator": self.name, "parsing_error": None},
            ))
        return records

    def aggregate(self, ctx: RunContext, records: list[EvalRecord]) -> dict[str, Any]:
        if not records:
            return {
                "accuracy_per_attempt": 0.0,
                "official_score": 0.0,
                "strict_score": 0.0,
                "total_attempts": 0,
                "correct_attempts": 0,
            }

        total = len(records)
        correct = sum(1 for r in records if r.is_correct)

        # Per-problem: solved if any attempt is correct
        solved_by_puzzle: dict[str, bool] = {}
        for rec in records:
            solved_by_puzzle.setdefault(rec.problem_uid, False)
            solved_by_puzzle[rec.problem_uid] = solved_by_puzzle[rec.problem_uid] or rec.is_correct

        n_solved = sum(1 for ok in solved_by_puzzle.values() if ok)
        n_puzzles = len(solved_by_puzzle)

        return {
            "accuracy_per_attempt": correct / total,
            "official_score": float(n_solved),
            "strict_score": float(n_solved),
            "strict_solved_puzzles": n_solved,
            "official_score_sum": float(n_solved),
            "total_attempts": total,
            "correct_attempts": correct,
            "total_puzzles": n_puzzles,
            "solve_rate": n_solved / n_puzzles if n_puzzles > 0 else 0.0,
        }
