"""ReasoningBank selector retriever thin port."""
from __future__ import annotations

import re

from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ReasoningBankSelectorRetriever:
    name = "reasoning_bank"
    COMPATIBLE_SCHEMAS = {"reasoning_bank"}

    def __init__(self, top_k: int = 1, include_prev_attempt: bool = True):
        self.top_k = int(top_k)
        self.include_prev_attempt = bool(include_prev_attempt)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        experiences = list(memory.payload.get("experiences", [])) if memory else []
        if experiences:
            return self._retrieve_experiences(memory, problem, previous_attempts, experiences)
        return self._retrieve_flat_legacy(memory, problem, previous_attempts)

    def _query_tokens(
        self, problem: ProblemSpec, previous_attempts: list[AttemptRecord]
    ) -> set[str]:
        query_parts = [str(problem.train_pairs)]
        if self.include_prev_attempt and previous_attempts:
            query_parts.append(previous_attempts[-1].completion[:1000])
        return _tokenize(" ".join(query_parts))

    def _empty_bundle(
        self, problem: ProblemSpec, mode: str, memory_item_count: int
    ) -> RetrievalBundle:
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=None,
            retrieved_items=[],
            metadata={"selector_mode": mode, "memory_item_count": memory_item_count},
        )

    def _retrieve_experiences(
        self,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        experiences: list[dict],
    ) -> RetrievalBundle:
        all_memory_items = [
            item for exp in experiences for item in exp.get("memory_items", [])
        ]
        if not all_memory_items or self.top_k <= 0:
            return self._empty_bundle(
                problem, "empty", memory_item_count=len(all_memory_items)
            )

        query_tokens = self._query_tokens(problem, previous_attempts)
        # ReasoningBank port choice: Jaccard-token-overlap proxy for embedding similarity (paper uses gemini-embedding-001; lib-free proxy keeps mem2 dep-free).
        scored = [
            (
                _jaccard(query_tokens, _tokenize(str(exp.get("source_query", "")))),
                exp,
            )
            for exp in experiences
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_experiences = scored[: self.top_k]
        selected_items = []
        for score, exp in top_experiences:
            for item in exp.get("memory_items", []):
                selected_items.append(
                    {
                        **item,
                        "score": score,
                        "source_task_uid": exp.get("source_task_uid", ""),
                        "from_success": bool(exp.get("from_success", False)),
                        "created_at": exp.get("created_at", ""),
                    }
                )

        return self._bundle_from_items(
            problem,
            selected_items,
            metadata={
                "selector_mode": "experience_jaccard",
                "memory_item_count": len(all_memory_items),
                "selected_count": len(selected_items),
                "experience_count": len(experiences),
                "selected_experience_count": len(top_experiences),
            },
        )

    def _retrieve_flat_legacy(
        self,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        items = list(memory.payload.get("memory_items", [])) if memory else []
        if not items or self.top_k <= 0:
            return self._empty_bundle(problem, "empty", memory_item_count=len(items))

        query_tokens = self._query_tokens(problem, previous_attempts)
        # ReasoningBank port choice: Jaccard-token-overlap proxy for embedding similarity (paper uses gemini-embedding-001; lib-free proxy keeps mem2 dep-free).
        scored = []
        for item in items:
            title = str(item.get("title", ""))
            description = str(item.get("description", ""))
            score = _jaccard(query_tokens, _tokenize(f"{title} {description}"))
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[: self.top_k]
        retrieved_items = [{**item, "score": score} for score, item in top]
        return self._bundle_from_items(
            problem,
            retrieved_items,
            metadata={
                "selector_mode": "legacy_flat_jaccard",
                "memory_item_count": len(items),
                "selected_count": len(retrieved_items),
            },
        )

    def _bundle_from_items(
        self,
        problem: ProblemSpec,
        retrieved_items: list[dict],
        metadata: dict,
    ) -> RetrievalBundle:
        intro = (
            "Below are memory items from past interaction that may be helpful to solve the task. "
            "You can use them when you feel relevant. In each step, please first explicitly discuss "
            "if you want to use each memory item or not, then take action.\n\n"
        )
        body = "\n\n".join(
            f"Item {idx + 1}: {item.get('title', '')}\n{item.get('content', '')}"
            for idx, item in enumerate(retrieved_items)
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=intro + body,
            retrieved_items=retrieved_items,
            metadata=metadata,
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        return self.retrieve(ctx, memory, problem, previous_attempts)
