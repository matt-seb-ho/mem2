"""LiveCodeBench benchmark adapter.

Loads problems from a local JSONL file or HuggingFace dataset,
filters by difficulty, and maps to ProblemSpec.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    """Load LiveCodeBench problems from a local JSONL file or HF dataset.

    Supports two data formats:
    - JSONL file: ``data_root`` points to a directory containing ``problems.jsonl``
    - HF dataset: ``data_root`` points to a HuggingFace ``save_to_disk`` directory
      (requires the ``datasets`` library)
    """

    name = "livecodebench"

    def __init__(
        self,
        data_root: str = "data/livecodebench_v56",
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

    def _load_jsonl(self, jsonl_path: Path) -> dict[str, ProblemSpec]:
        """Load from a JSONL file."""
        diff_set = set(self.difficulties) if self.difficulties else None

        problems: dict[str, ProblemSpec] = {}
        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                row = json.loads(line)

                difficulty = str(row.get("difficulty", row.get("question_difficulty", "")))
                if diff_set and difficulty not in diff_set:
                    continue

                uid = str(row.get("question_id", row.get("task_id", f"lcb_{idx}")))
                if self.include_ids and uid not in self.include_ids:
                    continue

                question_content = str(
                    row.get("question_content", row.get("problem_statement", row.get("prompt", "")))
                )

                public_tests = _parse_test_cases(
                    row.get("public_test_cases", row.get("sample_test_cases", None))
                )
                private_tests = _parse_test_cases(
                    row.get("private_test_cases", row.get("hidden_test_cases", None))
                )

                starter_code = str(row.get("starter_code", row.get("code_stub", ""))).strip()

                problems[uid] = ProblemSpec(
                    uid=uid,
                    train_pairs=[],
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

    def _load_hf(self) -> dict[str, ProblemSpec]:
        """Load from a HuggingFace dataset saved to disk."""
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise DataValidationError(
                "The 'datasets' library is required when loading from HF format. "
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

            public_tests = _parse_test_cases(
                row.get("public_test_cases", row.get("sample_test_cases", None))
            )
            private_tests = _parse_test_cases(
                row.get("private_test_cases", row.get("hidden_test_cases", None))
            )

            starter_code = str(row.get("starter_code", row.get("code_stub", ""))).strip()

            problems[uid] = ProblemSpec(
                uid=uid,
                train_pairs=[],
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

    def load(self, ctx: RunContext) -> dict[str, ProblemSpec]:
        jsonl_path = Path(self.data_root) / "problems.jsonl"
        if jsonl_path.exists():
            logger.info("Loading LCB problems from JSONL: %s", jsonl_path)
            return self._load_jsonl(jsonl_path)
        else:
            logger.info("Loading LCB problems from HF dataset: %s", self.data_root)
            return self._load_hf()

    def validate(self, problems: dict[str, ProblemSpec]) -> None:
        if not problems:
            raise DataValidationError(
                "No LiveCodeBench problems loaded. Check data_root and filters."
            )
        for uid, problem in problems.items():
            if not problem.metadata.get("question_content"):
                raise DataValidationError(f"Problem {uid} has no question content")
