from __future__ import annotations

from mem2.core.entities import AttemptRecord, EvalRecord, FeedbackRecord, ProblemSpec, RunContext


class MathPsGroundTruthFeedbackEngine:
    """Ground-truth feedback for math-PS attempts.

    Produces structured feedback from eval records:
    - Parsing errors → "No valid code block found"
    - Execution errors → error message
    - Wrong answer → "Your code returned X, expected Y"
    - Correct → positive message
    """

    name = "math_ps_gt"

    def __init__(
        self,
        positive_msg: str = "Correct",
        negative_msg: str = "Incorrect",
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
        expected = problem.metadata.get("answer_int")
        out: list[FeedbackRecord] = []

        for idx, att in enumerate(attempts):
            rec = eval_records[idx] if idx < len(eval_records) else None
            is_correct = rec.is_correct if rec is not None else False

            if is_correct:
                content = self.positive_msg
                errors: list[str] = []
                mismatches: list[dict] = []
            elif rec is not None:
                errors, mismatches = self._extract_outcomes(rec, expected)
                content = self._format_feedback(errors, mismatches)
            else:
                errors = []
                mismatches = []
                content = self.negative_msg

            out.append(FeedbackRecord(
                problem_uid=problem.uid,
                attempt_idx=idx,
                feedback_type="gt",
                content=content,
                metadata={
                    "is_correct": is_correct,
                    "errors": errors,
                    "mismatches": mismatches,
                },
            ))
        return out

    @staticmethod
    def _extract_outcomes(
        rec: EvalRecord, expected: int | None,
    ) -> tuple[list[str], list[dict]]:
        errors: list[str] = []
        mismatches: list[dict] = []

        parsing_error = rec.metadata.get("parsing_error")
        if parsing_error:
            errors.append(str(parsing_error))
            return errors, mismatches

        exec_error = rec.metadata.get("exec_error")
        if exec_error:
            errors.append(str(exec_error))
            return errors, mismatches

        # Wrong answer
        for detail in rec.test_details:
            if detail.get("correct"):
                continue
            err = detail.get("error")
            if err:
                errors.append(str(err))
            else:
                mismatches.append({
                    "output": detail.get("output"),
                    "expected": detail.get("expected", expected),
                })
        return errors, mismatches

    @staticmethod
    def _format_feedback(errors: list[str], mismatches: list[dict]) -> str:
        sections: list[str] = []
        if errors:
            sections.append("**Execution / Parsing Errors**")
            sections.extend(f"- {e}" for e in errors)
        if mismatches:
            sections.append("**Wrong Answer**")
            for m in mismatches:
                sections.append(
                    f"- Your code returned {m.get('output')}, "
                    f"but the expected answer is {m.get('expected')}"
                )
        if not sections:
            sections.append("Incorrect")
        return "\n".join(sections)
