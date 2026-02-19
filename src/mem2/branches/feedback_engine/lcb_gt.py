"""Ground-truth feedback engine for LiveCodeBench.

Produces per-test-case feedback showing which tests passed/failed.
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
                errors, test_failures = self._extract_outcomes(rec)
                content = self._format_feedback(errors, test_failures, rec.test_details)
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
    ) -> tuple[list[str], list[dict]]:
        """Extract errors and test failures from eval record."""
        errors: list[str] = []
        test_failures: list[dict] = []

        parsing_error = rec.metadata.get("parsing_error")
        if parsing_error:
            errors.append(str(parsing_error))
            return errors, test_failures

        for detail in rec.test_details:
            if detail.get("correct"):
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

        return errors, test_failures

    @staticmethod
    def _format_feedback(
        errors: list[str],
        test_failures: list[dict],
        test_details: list[dict],
    ) -> str:
        total = len(test_details)
        passed = sum(1 for d in test_details if d.get("correct"))

        sections: list[str] = []
        if errors:
            sections.append("**Execution / Parsing Errors**")
            sections.extend(f"- {e}" for e in errors)
        if test_failures:
            sections.append(f"**Failed Test Cases** ({passed}/{total} passed)")
            for tf in test_failures:
                sections.append(
                    f"- Test {tf.get('test_idx', '?')}: "
                    f"expected {tf.get('expected', '?')!r}, "
                    f"got {tf.get('actual', '?')!r}"
                )
        if not sections:
            sections.append(f"Incorrect ({passed}/{total} tests passed)")
        return "\n".join(sections)
