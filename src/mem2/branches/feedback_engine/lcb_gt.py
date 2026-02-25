"""Ground-truth feedback engine for LiveCodeBench.

Produces per-test-case feedback showing which *public* tests passed/failed.
Private test results are not exposed to avoid ground-truth leakage.
"""

from __future__ import annotations

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    ProblemSpec,
    RunContext,
)


class LcbGroundTruthFeedbackEngine:
    """Per-test-case feedback for LiveCodeBench attempts."""

    name = "lcb_gt"
    DOMAIN_NAME = "code"

    def __init__(
        self,
        positive_msg: str = "All test cases passed",
        negative_msg: str = "Some test cases failed",
    ):
        self.positive_msg = positive_msg
        self.negative_msg = negative_msg

    async def generate(
        self,
        ctx: RunContext,
        provider,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord] | None,
    ) -> list[FeedbackRecord]:
        eval_records = eval_records or []
        out: list[FeedbackRecord] = []

        for idx, att in enumerate(attempts):
            rec = eval_records[idx] if idx < len(eval_records) else None
            is_correct = rec.is_correct if rec is not None else False

            if is_correct:
                content = self.positive_msg
                errors: list[str] = []
                test_failures: list[dict] = []
            elif rec is not None:
                errors, test_failures, pub_passed, pub_total = self._extract_outcomes(rec)
                content = self._format_feedback(errors, test_failures, pub_passed, pub_total)
            else:
                errors = []
                test_failures = []
                content = self.negative_msg

            out.append(FeedbackRecord(
                problem_uid=problem.uid,
                attempt_idx=idx,
                feedback_type="gt",
                content=content,
                metadata={
                    "is_correct": is_correct,
                    "errors": errors,
                    "test_failures": test_failures,
                },
            ))
        return out

    @staticmethod
    def _extract_outcomes(
        rec: EvalRecord,
    ) -> tuple[list[str], list[dict], int, int]:
        """Extract errors and test failures from public tests only.

        Private test results are withheld to avoid ground-truth leakage.
        Returns (errors, test_failures, public_passed, public_total).
        """
        errors: list[str] = []
        test_failures: list[dict] = []

        parsing_error = rec.metadata.get("parsing_error")
        if parsing_error:
            errors.append(str(parsing_error))
            return errors, test_failures, 0, 0

        public_total = 0
        public_passed = 0
        for detail in rec.test_details:
            if not detail.get("is_train", False):
                continue  # skip private tests
            public_total += 1
            if detail.get("correct"):
                public_passed += 1
                continue
            err = detail.get("error")
            if err:
                errors.append(str(err))
            else:
                test_failures.append({
                    "test_idx": detail.get("pair_idx", 0),
                    "expected": detail.get("expected", ""),
                    "actual": detail.get("output", ""),
                })

        return errors, test_failures, public_passed, public_total

    @staticmethod
    def _format_feedback(
        errors: list[str],
        test_failures: list[dict],
        public_passed: int,
        public_total: int,
    ) -> str:
        sections: list[str] = []
        if errors:
            sections.append("**Execution / Parsing Errors**")
            sections.extend(f"- {e}" for e in errors)
        if test_failures:
            sections.append(
                f"**Failed Example Test Cases** ({public_passed}/{public_total} passed)"
            )
            for tf in test_failures:
                sections.append(
                    f"- Test {tf.get('test_idx', '?')}: "
                    f"expected {tf.get('expected', '?')!r}, "
                    f"got {tf.get('actual', '?')!r}"
                )
        if not sections:
            if public_total > 0:
                sections.append(
                    f"Example tests passed ({public_passed}/{public_total}), "
                    "but some hidden tests failed"
                )
            else:
                sections.append("Incorrect")
        return "\n".join(sections)
