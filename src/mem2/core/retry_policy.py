from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ArcMemoRetryPolicy:
    max_passes: int = 3
    criterion: str = "train"  # train | test | all
    error_feedback: str = "all"  # first | all
    num_feedback_passes: int = 1  # -1 means all previous passes
    include_past_outcomes: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ArcMemoRetryPolicy":
        run = config.get("run", {})
        policy_cfg = run.get("retry_policy", {})
        return cls(
            max_passes=int(policy_cfg.get("max_passes", run.get("max_passes", 3))),
            criterion=str(policy_cfg.get("criterion", run.get("retry_criterion", "train"))),
            error_feedback=str(policy_cfg.get("error_feedback", "all")),
            num_feedback_passes=int(policy_cfg.get("num_feedback_passes", 1)),
            include_past_outcomes=bool(policy_cfg.get("include_past_outcomes", True)),
        )

    def needs_retry(self, eval_record: Any) -> bool:
        """Mirror ArcMemo retry criterion semantics on a single attempt result."""
        criterion = self.criterion.lower().strip()
        if criterion == "train":
            details = eval_record.train_details
        elif criterion == "test":
            details = eval_record.test_details
        else:
            details = list(eval_record.train_details) + list(eval_record.test_details)
        if not details:
            return True
        return any(not bool(d.get("correct", False)) for d in details)

    def is_valid(self) -> tuple[bool, str | None]:
        if self.max_passes < 1:
            return False, "max_passes must be >= 1"
        if self.criterion not in {"train", "test", "all"}:
            return False, "criterion must be one of: train, test, all"
        if self.error_feedback not in {"first", "all"}:
            return False, "error_feedback must be one of: first, all"
        if self.num_feedback_passes == 0 or self.num_feedback_passes < -1:
            return False, "num_feedback_passes must be -1 or >= 1"
        return True, None

