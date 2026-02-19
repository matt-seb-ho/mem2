"""LiveCodeBench benchmark adapter.

Loads problems from the LCB HuggingFace dataset (saved to disk),
filters by difficulty, and maps to ProblemSpec.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mem2.core.entities import ProblemSpec, RunContext
from mem2.core.errors import DataValidationError

logger = logging.getLogger(__name__)


def _parse_test_cases(raw: str | list | None) -> list[dict[str, str]]:
    """Parse test cases from various LCB formats.

    Returns list of ``{"input": str, "expected_output": str}``.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(raw, list):
        cases: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                inp = str(item.get("input", item.get("stdin", "")))
                out = str(item.get("expected_output", item.get("output", item.get("stdout", ""))))
                cases.append({"input": inp, "expected_output": out})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                cases.append({"input": str(item[0]), "expected_output": str(item[1])})
        return cases
    return []


class LiveCodeBenchAdapter:
    """Load LiveCodeBench problems from a local HF dataset copy.

    Requires the ``datasets`` library and a local copy saved via
    ``datasets.save_to_disk``.
    """

    name = "livecodebench"

    def __init__(
        self,
        data_root: str = "/root/workspace/data/hf/livecodebench",
        split: str = "test",
        difficulties: list[str] | None = None,
        limit: int = 0,
        include_ids: list[str] | None = None,
    ):
        self.data_root = str(data_root)
        self.split = str(split)
        self.difficulties = list(difficulties) if difficulties else None
        self.limit = int(limit)
        self.include_ids = set(include_ids) if include_ids else None

    def load(self, ctx: RunContext) -> dict[str, ProblemSpec]:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise DataValidationError(
                "The 'datasets' library is required for livecodebench. "
                "Install with: pip install datasets"
            ) from exc

        ds = load_from_disk(self.data_root)
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            if self.split not in ds:
                raise DataValidationError(
                    f"Split '{self.split}' not found in dataset at {self.data_root}. "
                    f"Available: {list(ds.keys())}"
                )
            raw = ds[self.split]
        else:
            # Dataset without splits
            raw = ds

        diff_set = set(self.difficulties) if self.difficulties else None

        problems: dict[str, ProblemSpec] = {}
        for idx in range(len(raw)):
            row = raw[idx]

            difficulty = str(row.get("difficulty", row.get("question_difficulty", "")))
            if diff_set and difficulty not in diff_set:
                continue

            uid = str(row.get("question_id", row.get("task_id", f"lcb_{idx}")))
            if self.include_ids and uid not in self.include_ids:
                continue

            question_content = str(
                row.get("question_content", row.get("problem_statement", row.get("prompt", "")))
            )

            # Parse test cases
            public_tests = _parse_test_cases(
                row.get("public_test_cases", row.get("sample_test_cases", None))
            )
            private_tests = _parse_test_cases(
                row.get("private_test_cases", row.get("hidden_test_cases", None))
            )

            starter_code = str(row.get("starter_code", row.get("code_stub", ""))).strip()

            problems[uid] = ProblemSpec(
                uid=uid,
                train_pairs=[],  # Code problems don't use grid pairs
                test_pairs=[],
                metadata={
                    "question_content": question_content,
                    "difficulty": difficulty,
                    "public_test_cases": public_tests,
                    "private_test_cases": private_tests,
                    "starter_code": starter_code,
                    "dataset_idx": idx,
                },
            )

            if 0 < self.limit <= len(problems):
                break

        return problems

    def validate(self, problems: dict[str, ProblemSpec]) -> None:
        if not problems:
            raise DataValidationError(
                "No LiveCodeBench problems loaded. Check data_root and filters."
            )
        for uid, problem in problems.items():
            if not problem.metadata.get("question_content"):
                raise DataValidationError(f"Problem {uid} has no question content")
