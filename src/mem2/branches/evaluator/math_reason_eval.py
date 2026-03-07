from __future__ import annotations

import re
from typing import Any

from mem2.core.entities import AttemptRecord, EvalRecord, ProblemSpec, RunContext

# Reuse the same regex from the benchmark adapter
_BOXED_RE = re.compile(r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} value from model output."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def parse_integer_answer(answer: str) -> int | None:
    """Try to parse an integer from a boxed answer string.

    Handles: "42", "-7", "1000", " 42 ", "1,000", etc.
    Returns None if the string cannot be interpreted as an integer.
    """
    cleaned = answer.replace(",", "").replace(" ", "").strip()
    # Handle negative sign
    if cleaned.startswith("-"):
        digits = cleaned[1:]
    else:
        digits = cleaned
    if digits.isdigit():
        return int(cleaned)
    return None


class MathReasonEvaluator:
    """Evaluate math reasoning attempts by parsing \\boxed{} answers.

    Extracts the last \\boxed{N} from the model's text output and
    compares it to the ground-truth integer answer. No code execution.
    """

    name = "math_reason_eval"
    DOMAIN_NAME = "math"

    def __init__(self, timeout_s: float = 10.0):
        # timeout_s kept for interface compatibility but unused
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
            completion = attempt.completion or ""

            # Try to extract \boxed{} answer
            boxed = extract_boxed_answer(completion)

            if boxed is None:
                records.append(EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=False,
                    train_details=[],
                    test_details=[{
                        "is_train": False,
                        "pair_idx": 0,
                        "correct": False,
                        "error": "no \\boxed{} answer found in response",
                        "output": None,
                        "expected": expected,
                    }],
                    metadata={
                        "evaluator": self.name,
                        "parsing_error": "no \\boxed{} answer found",
                    },
                ))
                continue

            # Try to parse as integer
            parsed = parse_integer_answer(boxed)

            if parsed is None:
                records.append(EvalRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    is_correct=False,
                    train_details=[],
                    test_details=[{
                        "is_train": False,
                        "pair_idx": 0,
                        "correct": False,
                        "error": f"\\boxed{{{boxed}}} is not an integer",
                        "output": boxed,
                        "expected": expected,
                    }],
                    metadata={
                        "evaluator": self.name,
                        "parsing_error": f"non-integer boxed answer: {boxed}",
                        "raw_boxed": boxed,
                    },
                ))
                continue

            # Compare to ground truth
            try:
                is_correct = parsed == int(expected)
            except (TypeError, ValueError):
                is_correct = str(parsed).strip() == str(expected).strip()

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
                    "output": parsed,
                    "expected": expected,
                }],
                metadata={
                    "evaluator": self.name,
                    "parsing_error": None,
                    "raw_boxed": boxed,
                },
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
