from __future__ import annotations

import json
from pathlib import Path

from mem2.core.entities import ProblemSpec, RunContext
from mem2.core.errors import DataValidationError


class ArcAgiBenchmarkAdapter:
    name = "arc_agi"

    def __init__(
        self,
        data_root: str = "data/arc_agi/training",
        limit: int = 5,
        include_ids: list[str] | None = None,
    ):
        self.data_root = Path(data_root)
        self.limit = int(limit)
        self.include_ids = list(include_ids or [])
        self._include_ids_set = set(self.include_ids)

    def load(self, ctx: RunContext) -> dict[str, ProblemSpec]:
        if not self.data_root.exists():
            raise DataValidationError(f"ARC data root does not exist: {self.data_root}")

        file_map = {fp.stem: fp for fp in self.data_root.glob("*.json")}
        if self.include_ids:
            files = [file_map[uid] for uid in self.include_ids if uid in file_map]
        else:
            files = sorted(file_map.values())
        if self.limit > 0:
            files = files[: self.limit]

        problems = {}
        for fp in files:
            payload = json.loads(fp.read_text())
            problems[fp.stem] = ProblemSpec(
                uid=fp.stem,
                train_pairs=payload.get("train", []),
                test_pairs=payload.get("test", []),
                metadata={"source": str(fp)},
            )
        return problems

    def validate(self, problems: dict[str, ProblemSpec]) -> None:
        if not problems:
            raise DataValidationError("No ARC problems loaded")
        for uid, problem in problems.items():
            if not problem.train_pairs:
                raise DataValidationError(f"Problem {uid} has no train pairs")
            if not problem.test_pairs:
                raise DataValidationError(f"Problem {uid} has no test pairs")
