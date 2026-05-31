"""ReasoningBank memory builder thin port."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)
from mem2.core.errors import ConfigurationError

SUCCESSFUL_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**. 

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
"""

FAILED_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**. 

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_EXTRACTION_CONTEXT_CHAR_LIMIT = 4000
_EXTRACTION_QUERY_CHAR_LIMIT = 3000


def parse_memory_items_markdown(text: str) -> list[dict[str, str]]:
    """Parse the SUCCESSFUL_SI / FAILED_SI output. Returns list of items."""
    items: list[dict[str, str]] = []
    blocks = re.split(r"^#\s*Memory Item\s*\d*\s*\n", text, flags=re.MULTILINE)
    for block in blocks[1:]:
        title_m = re.search(r"^##\s*Title\s+(.+?)$", block, flags=re.MULTILINE)
        desc_m = re.search(r"^##\s*Description\s+(.+?)$", block, flags=re.MULTILINE)
        content_m = re.search(
            r"^##\s*Content\s+(.+?)(?=\n#|\Z)",
            block,
            flags=re.MULTILINE | re.DOTALL,
        )
        if title_m and content_m:
            items.append(
                {
                    "title": title_m.group(1).strip(),
                    "description": desc_m.group(1).strip() if desc_m else "",
                    "content": content_m.group(1).strip(),
                }
            )
    return items[:3]


def _format_arc_trajectory(completion: str) -> str:
    # ReasoningBank port choice: ARC attempts are Python completions, not WebArena
    # think/action trajectories; map pre-code reasoning to <think> and code to <action>.
    match = _CODE_BLOCK_RE.search(completion)
    if match is None:
        return completion
    reasoning = completion[: match.start()].strip()
    code = match.group(1).strip()
    return f"<think>\n{reasoning}\n</think>\n<action>\n{code}\n</action>"


def _serialize_problem_for_extraction(attempt: AttemptRecord) -> str:
    """Return the task content used as ReasoningBank's extraction query."""
    prompt = attempt.prompt.strip()
    if prompt:
        return prompt[:_EXTRACTION_QUERY_CHAR_LIMIT]
    problem_payload = attempt.metadata.get("problem") or attempt.metadata.get("problem_spec")
    if isinstance(problem_payload, dict):
        parts = ["Problem train pairs:"]
        train_pairs = problem_payload.get("train_pairs") or problem_payload.get("train") or []
        for idx, pair in enumerate(list(train_pairs)[:5]):
            if isinstance(pair, dict):
                parts.append(
                    f"  pair {idx}: input={pair.get('input')}, output={pair.get('output')}"
                )
            else:
                parts.append(f"  pair {idx}: {pair}")
        test_pairs = problem_payload.get("test_pairs") or problem_payload.get("test") or []
        if test_pairs:
            parts.append("Test inputs:")
            for idx, pair in enumerate(list(test_pairs)[:2]):
                if isinstance(pair, dict):
                    parts.append(f"  test {idx}: input={pair.get('input')}")
                else:
                    parts.append(f"  test {idx}: {pair}")
        serialized = "\n".join(parts).strip()
        if serialized:
            return serialized[:_EXTRACTION_QUERY_CHAR_LIMIT]
    return f"Problem UID: {attempt.problem_uid}"


def _materialize_memory_items(experiences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for exp in experiences:
        for item in exp.get("memory_items", []):
            items.append(
                {
                    **item,
                    "source_task_uid": exp.get("source_task_uid", ""),
                    "from_success": bool(exp.get("from_success", False)),
                    "created_at": exp.get("created_at", ""),
                }
            )
    return items


class ReasoningBankMemoryBuilder:
    name = "reasoning_bank"
    SCHEMA_NAME = "reasoning_bank"

    def __init__(
        self,
        max_items_per_trajectory: int = 3,
        extraction_temperature: float = 1.0,
        freeze: bool = False,
        extraction_model: str = "",
        seed_memory_file: str | None = None,
    ):
        self.max_items_per_trajectory = int(max_items_per_trajectory)
        self.extraction_temperature = float(extraction_temperature)
        self.freeze = bool(freeze)
        self.extraction_model = extraction_model
        self.seed_memory_file = seed_memory_file

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        experiences: list[dict] = []
        memory_items: list[dict] = []
        seed_metadata: dict[str, Any] = {}
        if self.seed_memory_file:
            from pathlib import Path
            import json as _json
            seed_path = Path(self.seed_memory_file).expanduser()
            if not seed_path.is_absolute():
                seed_path = Path.cwd() / seed_path
            if seed_path.exists():
                try:
                    seed_blob = _json.loads(seed_path.read_text(encoding="utf-8"))
                    if isinstance(seed_blob, dict):
                        payload = seed_blob.get("payload") or {}
                        if isinstance(payload.get("experiences"), list):
                            experiences = list(payload["experiences"])
                        if isinstance(payload.get("memory_items"), list):
                            memory_items = list(payload["memory_items"])
                        # If only experiences are present, derive flat memory_items.
                        if experiences and not memory_items:
                            memory_items = _materialize_memory_items(experiences)
                        seed_metadata["seed_source"] = str(self.seed_memory_file)
                        seed_metadata["seed_schema_version"] = str(
                            seed_blob.get("schema_version") or ""
                        )
                        seed_metadata["seed_experience_count"] = len(experiences)
                        seed_metadata["seed_memory_item_count"] = len(memory_items)
                except Exception as exc:  # pragma: no cover - configuration fault
                    seed_metadata["seed_load_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        return MemoryState(
            schema_name=self.SCHEMA_NAME,
            schema_version="v2",
            payload={"experiences": experiences, "memory_items": memory_items},
            metadata={
                "initialized_problem_count": len(problems),
                **seed_metadata,
            },
        )

    def update(
        self,
        ctx: RunContext,
        memory: MemoryState,
        attempts: list[AttemptRecord],
        eval_records: list[EvalRecord],
        feedback_records: list[FeedbackRecord],
    ) -> MemoryState:
        if self.freeze:
            return memory

        experiences = list(memory.payload.get("experiences", []))
        for idx, att in enumerate(attempts):
            # ReasoningBank port choice: use EvalRecord.is_correct directly instead of LLM-as-judge (we have ground truth in mem2; paper does not).
            is_correct = eval_records[idx].is_correct if idx < len(eval_records) else False
            prompt = SUCCESSFUL_SI if is_correct else FAILED_SI
            query = _serialize_problem_for_extraction(att)
            trajectory = _format_arc_trajectory(att.completion)
            response_text = self._extract_memory_items(ctx, prompt, query, trajectory)
            new_items = parse_memory_items_markdown(response_text)[
                : self.max_items_per_trajectory
            ]
            experiences.append(
                {
                    "source_task_uid": att.problem_uid,
                    "source_query": query,
                    "source_trajectory": trajectory,
                    "from_success": is_correct,
                    "created_at": datetime.now(tz=UTC).isoformat(),
                    "memory_items": new_items,
                }
            )

        memory.schema_version = "v2"
        memory.payload["experiences"] = experiences
        memory.payload["memory_items"] = _materialize_memory_items(experiences)
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        # ReasoningBank port choice: MaTTS (memory-aware test-time scaling) is out of scope for this thin port.
        return memory

    def _extract_memory_items(
        self,
        ctx: RunContext,
        prompt: str,
        query: str,
        trajectory: str,
    ) -> str:
        provider = self._resolve_provider(ctx)
        if provider is None:
            raise ConfigurationError(
                "ReasoningBankMemoryBuilder requires _reasoning_bank_provider or "
                "_meta_edit_provider. Use freeze=True or wire a provider."
            )
        trajectory_budget = max(0, _EXTRACTION_CONTEXT_CHAR_LIMIT - len(query))
        user_message = (
            f"{prompt}\n\n"
            f"Query: {query}\n"
            f"Trajectory:\n{trajectory[:trajectory_budget]}"
        )
        completions = provider.generate(
            user_message,
            model=self.extraction_model or getattr(provider, "model", ""),
            n=1,
            temperature=self.extraction_temperature,
        )
        return str(completions[0]) if completions else ""

    @staticmethod
    def _resolve_provider(ctx: RunContext) -> Any:
        config = ctx.config or {}
        return config.get("_reasoning_bank_provider") or config.get("_meta_edit_provider")
