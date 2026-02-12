from __future__ import annotations

from mem2.core.entities import AttemptRecord, MemoryState, ProblemSpec, RetrievalBundle, RunContext


class LessonTopKRetriever:
    name = "lesson_topk"

    def __init__(self, top_k: int = 2):
        self.top_k = int(top_k)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        entries = list(memory.payload.get("entries", []))
        problem_entries = [
            row for row in entries if str(row.get("problem_uid", "")).strip() == problem.uid
        ]
        source_entries = problem_entries if problem_entries else entries
        items = source_entries[-self.top_k :] if self.top_k > 0 else []
        hint_text = "\\n".join(str(x.get("hint", "")) for x in items if x.get("hint")) or None
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text,
            retrieved_items=items,
            metadata={
                "top_k": self.top_k,
                "history_attempts": len(previous_attempts),
                "scoped_to_problem": bool(problem_entries),
            },
        )
