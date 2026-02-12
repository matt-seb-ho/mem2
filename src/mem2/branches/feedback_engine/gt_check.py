from __future__ import annotations

from mem2.core.entities import AttemptRecord, EvalRecord, FeedbackRecord, ProblemSpec, RunContext


class GroundTruthFeedbackEngine:
    name = "gt_check"

    def __init__(self, positive_msg: str = "Correct", negative_msg: str = "Incorrect"):
        self.positive_msg = positive_msg
        self.negative_msg = negative_msg

    def _extract_outcomes(self, eval_record: EvalRecord) -> tuple[list[str], list[dict]]:
        errors: list[str] = []
        mismatches: list[dict] = []
        parsing_error = eval_record.metadata.get("parsing_error")
        if parsing_error:
            errors.append(parsing_error)
            return errors, mismatches

        # ArcMemo retry feedback only reports training-example outcomes.
        for detail in eval_record.train_details:
            pair_idx = int(detail.get("pair_idx", -1)) + 1
            err = detail.get("error")
            if err:
                errors.append(str(err))
                continue
            if not detail.get("correct", False):
                mismatches.append(
                    {
                        "example_idx": pair_idx,
                        "output": detail.get("output"),
                        "expected": detail.get("expected"),
                    }
                )
        return errors, mismatches

    def _format_failure_feedback(self, eval_record: EvalRecord) -> str:
        parsing_error = eval_record.metadata.get("parsing_error")
        if parsing_error:
            return (
                "**Execution / Parsing Errors**\\n"
                f"- {parsing_error}\\n"
                "- Please return a markdown python code block defining `transform`."
            )

        errors, mismatches_dict = self._extract_outcomes(eval_record)

        sections: list[str] = []
        if errors:
            sections.append("**Execution / Parsing Errors**")
            sections.extend(f"- {e}" for e in errors)
        if mismatches_dict:
            sections.append("**Output Mismatches**")
            for row in mismatches_dict[:3]:
                ex_idx = row.get("example_idx", "?")
                sections.append(
                    f"- Example {ex_idx}: output={row.get('output')} expected={row.get('expected')}"
                )
        if not sections:
            sections.append(self.negative_msg)
        return "\\n".join(sections)

    async def generate(
        self,
        ctx: RunContext,
        provider,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord] | None,
    ) -> list[FeedbackRecord]:
        eval_records = eval_records or []
        out = []
        for idx, att in enumerate(attempts):
            is_correct = eval_records[idx].is_correct if idx < len(eval_records) else False
            record = eval_records[idx] if idx < len(eval_records) else None
            if is_correct:
                content = self.positive_msg
                errors, mismatches = [], []
            else:
                content = (
                    self._format_failure_feedback(record)
                    if record is not None
                    else self.negative_msg
                )
                errors, mismatches = self._extract_outcomes(record) if record is not None else ([], [])
            out.append(
                FeedbackRecord(
                    problem_uid=problem.uid,
                    attempt_idx=idx,
                    feedback_type="gt",
                    content=content,
                    metadata={
                        "is_correct": is_correct,
                        "errors": errors,
                        "mismatches": mismatches,
                    },
                )
            )
        return out
