from __future__ import annotations

from mem2.core.entities import AttemptRecord, EvalRecord, FeedbackRecord, ProblemSpec, RunContext


class MathPsGroundTruthFeedbackEngine:
    """Ground-truth feedback for math-PS attempts.

    Produces structured feedback from eval records:
    - Parsing errors → "No valid code block found"
    - Execution errors → error message
    - Wrong answer → "Incorrect" (no ground-truth leak)
    - Correct → positive message
    """

    name = "math_ps_gt"
    DOMAIN_NAME = "math"

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
        out: list[FeedbackRecord] = []

        for idx, att in enumerate(attempts):
            rec = eval_records[idx] if idx < len(eval_records) else None
            is_correct = rec.is_correct if rec is not None else False

            if is_correct:
                content = self.positive_msg
                errors: list[str] = []
            elif rec is not None:
                errors = self._extract_errors(rec)
                content = self._format_feedback(errors)
            else:
                errors = []
                content = self.negative_msg

            out.append(FeedbackRecord(
                problem_uid=problem.uid,
                attempt_idx=idx,
                feedback_type="gt",
                content=content,
                metadata={
                    "is_correct": is_correct,
                    "errors": errors,
                },
            ))
        return out

    @staticmethod
    def _extract_errors(rec: EvalRecord) -> list[str]:
        """Extract only parsing/execution errors. No ground-truth leak."""
        errors: list[str] = []

        parsing_error = rec.metadata.get("parsing_error")
        if parsing_error:
            errors.append(str(parsing_error))
            return errors

        exec_error = rec.metadata.get("exec_error")
        if exec_error:
            errors.append(str(exec_error))
            return errors

        # Wrong answer — no details, just "Incorrect"
        return errors

    @staticmethod
    def _format_feedback(errors: list[str]) -> str:
        sections: list[str] = []
        if errors:
            sections.append("**Execution / Parsing Errors**")
            sections.extend(f"- {e}" for e in errors)
        else:
            sections.append("Incorrect")
        return "\n".join(sections)
