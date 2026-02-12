from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)


class ArcMemoPsMemoryBuilder:
    name = "arcmemo_ps"

    def __init__(
        self,
        max_entries: int = 200,
        seed_lessons_file: str | None = None,
        seed_lessons_per_problem: int = 5,
    ):
        self.max_entries = int(max_entries)
        self.seed_lessons_file = seed_lessons_file
        self.seed_lessons_per_problem = int(seed_lessons_per_problem)

    @staticmethod
    def _format_seed_hint(lesson: dict[str, Any]) -> str | None:
        situation = str(lesson.get("situation", "")).strip()
        suggestion = str(lesson.get("suggestion", "")).strip()
        if situation and suggestion:
            return f"- situation: {situation}\n  suggestion: {suggestion}"
        if situation:
            return f"- situation: {situation}"
        if suggestion:
            return f"- suggestion: {suggestion}"
        return None

    def _load_seed_entries(self, allowed_uids: set[str]) -> list[dict[str, Any]]:
        if not self.seed_lessons_file:
            return []

        path = Path(self.seed_lessons_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        payload = json.loads(path.read_text())
        entries: list[dict[str, Any]] = []

        # Allow seeding from a previously serialized memory state.
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("payload"), dict)
            and isinstance(payload["payload"].get("entries"), list)
        ):
            all_rows = [row for row in payload["payload"]["entries"] if isinstance(row, dict)]
            has_uid_match = any(
                str(row.get("problem_uid", "")).strip() in allowed_uids for row in all_rows
            )
            for row in payload["payload"]["entries"]:
                if not isinstance(row, dict):
                    continue
                uid = str(row.get("problem_uid", "")).strip()
                if has_uid_match and uid and uid not in allowed_uids:
                    continue
                hint = str(row.get("hint", "")).strip()
                if not hint:
                    continue
                entries.append(
                    {
                        "problem_uid": uid,
                        "pass_idx": int(row.get("pass_idx", -1)),
                        "is_correct": bool(row.get("is_correct", True)),
                        "feedback": str(row.get("feedback", "seeded memory")),
                        "hint": hint,
                    }
                )
            return entries

        # ArcMemo-style parsed lessons: {uid: [ {situation, suggestion}, ... ] }
        if isinstance(payload, dict):
            has_uid_match = any(uid in allowed_uids for uid in payload.keys())
            for uid, lesson_rows in payload.items():
                if has_uid_match and uid not in allowed_uids:
                    continue
                if isinstance(lesson_rows, str):
                    hint = lesson_rows.strip()
                    if hint:
                        entries.append(
                            {
                                "problem_uid": uid,
                                "pass_idx": -1,
                                "is_correct": True,
                                "feedback": "seeded lesson",
                                "hint": hint,
                            }
                        )
                    continue
                if not isinstance(lesson_rows, list):
                    continue
                row_count = 0
                for lesson in lesson_rows:
                    hint: str | None = None
                    if isinstance(lesson, dict):
                        hint = self._format_seed_hint(lesson)
                        if not hint:
                            fallback_hint = str(lesson.get("hint", "")).strip()
                            hint = fallback_hint or None
                    elif isinstance(lesson, str):
                        hint = lesson.strip() or None
                    if not hint:
                        continue
                    entries.append(
                        {
                            "problem_uid": uid,
                            "pass_idx": -1,
                            "is_correct": True,
                            "feedback": "seeded lesson",
                            "hint": hint,
                        }
                    )
                    row_count += 1
                    if self.seed_lessons_per_problem > 0 and row_count >= self.seed_lessons_per_problem:
                        break
        return entries

    def initialize(self, ctx: RunContext, problems: dict[str, ProblemSpec]) -> MemoryState:
        allowed_uids = set(problems.keys())
        seed_entries = self._load_seed_entries(allowed_uids=allowed_uids)
        if len(seed_entries) > self.max_entries:
            seed_entries = seed_entries[-self.max_entries :]
        return MemoryState(
            schema_name="arcmemo_ps",
            schema_version="v1",
            payload={"entries": seed_entries},
            metadata={
                "initialized_problem_count": len(problems),
                "seed_lessons_file": self.seed_lessons_file,
                "seeded_entry_count": len(seed_entries),
            },
        )

    def reflect(
        self,
        ctx: RunContext,
        problem: ProblemSpec,
        attempts: list[AttemptRecord],
        feedback: list[FeedbackRecord],
    ) -> list[dict]:
        items = []
        for idx, att in enumerate(attempts):
            fb_txt = feedback[idx].content if idx < len(feedback) else ""
            items.append(
                {
                    "problem_uid": problem.uid,
                    "attempt_idx": idx,
                    "attempt_preview": att.completion[:200],
                    "feedback": fb_txt[:200],
                }
            )
        return items

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        entries = list(memory.payload.get("entries", []))
        for i, att in enumerate(attempts):
            is_correct = eval_records[i].is_correct if i < len(eval_records) else False
            feedback = feedback_records[i].content if i < len(feedback_records) else ""
            entries.append(
                {
                    "problem_uid": att.problem_uid,
                    "pass_idx": att.pass_idx,
                    "is_correct": is_correct,
                    "feedback": feedback,
                    "hint": "preserve successful transformations" if is_correct else "inspect failure mode",
                }
            )
        if len(entries) > self.max_entries:
            entries = entries[-self.max_entries :]
        memory.payload["entries"] = entries
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        # Keep latest entries only for now. Consolidation strategy can be upgraded additively.
        entries = list(memory.payload.get("entries", []))
        memory.payload["entries"] = entries[-self.max_entries :]
        return memory
