from __future__ import annotations

import re
from typing import Any

from mem2.core.entities import ProblemSpec, RunContext
from mem2.core.errors import DataValidationError

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_boxed_answer(solution: str) -> str | None:
    """Extract the last \\boxed{...} value from a competition_math solution."""
    matches = _BOXED_RE.findall(solution)
    return matches[-1].strip() if matches else None


def _is_integer_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    return answer.replace("-", "").replace(" ", "").isdigit()


def _parse_integer_answer(answer: str) -> int:
    return int(answer.replace(" ", ""))


class CompetitionMathPsBenchmarkAdapter:
    """Load competition_math problems filtered to PS-compatible integer-answer subsets.

    Requires the ``datasets`` library and a local copy saved via
    ``datasets.save_to_disk``.
    """

    name = "competition_math_ps"

    def __init__(
        self,
        data_root: str = "/root/workspace/data/hf/qwedsacf__competition_math",
        split: str = "train",
        types: list[str] | None = None,
        levels: list[str] | None = None,
        limit: int = 0,
        include_ids: list[str] | None = None,
        require_integer_answer: bool = True,
    ):
        self.data_root = str(data_root)
        self.split = str(split)
        self.types = list(types or ["Number Theory", "Counting & Probability"])
        self.levels = list(levels) if levels else None
        self.limit = int(limit)
        self.include_ids = set(include_ids) if include_ids else None
        self.require_integer_answer = bool(require_integer_answer)

    def load(self, ctx: RunContext) -> dict[str, ProblemSpec]:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise DataValidationError(
                "The 'datasets' library is required for competition_math_ps. "
                "Install with: pip install datasets"
            ) from exc

        ds = load_from_disk(self.data_root)
        if self.split not in ds:
            raise DataValidationError(
                f"Split '{self.split}' not found in dataset at {self.data_root}. "
                f"Available: {list(ds.keys())}"
            )
        raw = ds[self.split]

        type_set = set(self.types) if self.types else None
        level_set = set(self.levels) if self.levels else None

        problems: dict[str, ProblemSpec] = {}
        for idx in range(len(raw)):
            row = raw[idx]
            math_type = row.get("type", "")
            level = row.get("level", "")

            if type_set and math_type not in type_set:
                continue
            if level_set and level not in level_set:
                continue

            answer_str = _extract_boxed_answer(row.get("solution", ""))
            if self.require_integer_answer and not _is_integer_answer(answer_str):
                continue

            uid = f"cmath_{idx}"
            if self.include_ids and uid not in self.include_ids:
                continue

            answer_int = _parse_integer_answer(answer_str) if _is_integer_answer(answer_str) else None

            problems[uid] = ProblemSpec(
                uid=uid,
                train_pairs=[],  # Math-PS has no I/O train pairs
                test_pairs=[],   # Verification is answer-match, not grid comparison
                metadata={
                    "problem_text": row["problem"],
                    "solution_text": row["solution"],
                    "answer_str": answer_str,
                    "answer_int": answer_int,
                    "math_type": math_type,
                    "level": level,
                    "dataset_idx": idx,
                },
            )

            if 0 < self.limit <= len(problems):
                break

        return problems

    def validate(self, problems: dict[str, ProblemSpec]) -> None:
        if not problems:
            raise DataValidationError(
                "No competition_math problems loaded. Check data_root, types, "
                "and level filters."
            )
        for uid, problem in problems.items():
            if not problem.metadata.get("problem_text"):
                raise DataValidationError(f"Problem {uid} has no problem text")
            if problem.metadata.get("answer_int") is None:
                raise DataValidationError(
                    f"Problem {uid} has no integer answer "
                    f"(answer_str={problem.metadata.get('answer_str')!r})"
                )
