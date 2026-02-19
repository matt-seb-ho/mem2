"""LCB execution evaluator.

Runs code with stdin piped, captures stdout, and compares against
expected output per test case.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any

from mem2.core.entities import AttemptRecord, EvalRecord, ProblemSpec, RunContext
from mem2.utils.code_execution import extract_python_block


def _worker_exec_stdin(code: str, stdin_data: str, queue: mp.Queue) -> None:
    """Execute code in an isolated process with stdin piped."""
    try:
        import io
        import sys

        # Capture stdout
        old_stdout = sys.stdout
        old_stdin = sys.stdin
        sys.stdout = captured_out = io.StringIO()
        sys.stdin = io.StringIO(stdin_data)

        try:
            exec(code, {"__builtins__": __builtins__}, {})
        finally:
            output = captured_out.getvalue()
            sys.stdout = old_stdout
            sys.stdin = old_stdin

        queue.put({"status": "ok", "error": None, "output": output})
    except Exception as exc:
        # Restore stdout/stdin in case of error
        import sys
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
        queue.put({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
            "output": None,
        })


def execute_with_stdin(code: str, stdin_data: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """Run code with stdin piped, capture stdout."""
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker_exec_stdin, args=(code, stdin_data, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "error", "error": f"timeout error (> {timeout_s}s)", "output": None}
    if queue.empty():
        return {"status": "error", "error": "execution error: empty worker response", "output": None}
    return queue.get()


class LcbExecutionEvaluator:
    """Evaluate LCB attempts by executing code with stdin and comparing stdout."""

    name = "lcb_exec"

    def __init__(self, timeout_s: float = 30.0, **kwargs):
        self.timeout_s = float(timeout_s)

    def evaluate(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
    ) -> list[EvalRecord]:
        # Collect all test cases (public + private)
        public_tests = problem.metadata.get("public_test_cases", [])
        private_tests = problem.metadata.get("private_test_cases", [])
        all_tests = public_tests + private_tests

        records: list[EvalRecord] = []
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
                        "expected": None,
                    }],
                    metadata={"evaluator": self.name, "parsing_error": parsing_error},
                ))
                continue

            if not all_tests:
                # No test cases available
                records.append(EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=False,
                    train_details=[],
                    test_details=[],
                    metadata={
                        "evaluator": self.name,
                        "error": "no test cases available",
                    },
                ))
                continue

            test_details: list[dict[str, Any]] = []
            all_correct = True

            for tc_idx, tc in enumerate(all_tests):
                stdin_data = tc.get("input", "")
                expected = tc.get("expected_output", "").strip()

                exec_result = execute_with_stdin(
                    code, stdin_data, timeout_s=self.timeout_s
                )

                if exec_result["status"] != "ok":
                    test_details.append({
                        "is_train": tc_idx < len(public_tests),
                        "pair_idx": tc_idx,
                        "correct": False,
                        "error": exec_result.get("error"),
                        "output": None,
                        "expected": expected,
                    })
                    all_correct = False
                    continue

                actual = str(exec_result["output"]).strip()
                correct = actual == expected

                test_details.append({
                    "is_train": tc_idx < len(public_tests),
                    "pair_idx": tc_idx,
                    "correct": correct,
                    "error": None,
                    "output": actual,
                    "expected": expected,
                })

                if not correct:
                    all_correct = False

            records.append(EvalRecord(
                problem_uid=problem.uid,
                attempt_idx=idx,
                is_correct=all_correct,
                train_details=[d for d in test_details if d.get("is_train")],
                test_details=test_details,
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

        solved_by_problem: dict[str, bool] = {}
        for rec in records:
            solved_by_problem.setdefault(rec.problem_uid, False)
            solved_by_problem[rec.problem_uid] = solved_by_problem[rec.problem_uid] or rec.is_correct

        n_solved = sum(1 for ok in solved_by_problem.values() if ok)
        n_problems = len(solved_by_problem)

        # Compute per-test-case pass rate
        total_tests = 0
        passed_tests = 0
        for rec in records:
            for td in rec.test_details:
                total_tests += 1
                if td.get("correct"):
                    passed_tests += 1

        return {
            "accuracy_per_attempt": correct / total,
            "official_score": float(n_solved),
            "strict_score": float(n_solved),
            "strict_solved_puzzles": n_solved,
            "official_score_sum": float(n_solved),
            "total_attempts": total,
            "correct_attempts": correct,
            "total_puzzles": n_problems,
            "solve_rate": n_solved / n_problems if n_problems > 0 else 0.0,
            "test_pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_test_cases": total_tests,
            "passed_test_cases": passed_tests,
        }
