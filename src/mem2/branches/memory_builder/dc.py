"""Dynamic Cheatsheet memory builder.

Default is FROZEN to match the ArcMemo paper baseline (sections/4_experiments.tex:20).
Set freeze: false in experiment YAML overrides for cumulative DC-Cu behavior.

Prompt constants are import-time reads from third_party/dynamic-cheatsheet. This
keeps byte parity with the checked-out upstream prompts, but prompt changes in
third_party can silently change this port.
"""

from __future__ import annotations

from pathlib import Path

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RunContext,
)
from mem2.core.errors import ConfigurationError


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "third_party/dynamic-cheatsheet").exists():
            return parent
    raise FileNotFoundError("Could not locate third_party/dynamic-cheatsheet")


def _read_upstream_prompt(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


CURATOR_PROMPT = _read_upstream_prompt(
    "third_party/dynamic-cheatsheet/prompts/curator_prompt_for_dc_cumulative.txt"
)


def _extract_cheatsheet(response: str, old_cheatsheet: str) -> str:
    response = str(response or "").strip()
    if "<cheatsheet>" not in response:
        return old_cheatsheet
    try:
        text = response.split("<cheatsheet>", 1)[1]
        text = text.split("</cheatsheet>", 1)[0]
        return text.strip()
    except Exception:
        return old_cheatsheet


def _cap_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if max_words <= 0 or len(words) <= max_words:
        return str(text or "")
    return " ".join(words[-max_words:])


class DcMemoryBuilder:
    name = "dc"
    SCHEMA_NAME = "dc"

    def __init__(
        self,
        max_words: int = 2500,
        freeze: bool = True,
        seed_buffer_file: str | None = None,
    ):
        self.max_words = int(max_words)
        self.freeze = bool(freeze)
        self.seed_buffer_file = seed_buffer_file
        self._curator_prompt = CURATOR_PROMPT

    def initialize(
        self, ctx: RunContext, problems: dict[str, ProblemSpec]
    ) -> MemoryState:
        cheatsheet = "(empty)"
        if self.seed_buffer_file:
            seed_path = Path(self.seed_buffer_file).expanduser()
            if not seed_path.is_absolute():
                seed_path = Path.cwd() / seed_path
            if seed_path.exists():
                cheatsheet = seed_path.read_text(encoding="utf-8")
        return MemoryState(
            schema_name="dc",
            schema_version="v1",
            payload={"cheatsheet": cheatsheet},
            metadata={
                "initialized_problem_count": len(problems),
                "seed_buffer_file": self.seed_buffer_file,
                "freeze": self.freeze,
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

        provider = None
        try:
            provider = (ctx.config or {}).get("_meta_edit_provider")
        except Exception:
            provider = None

        if provider is None:
            raise ConfigurationError(
                "DcMemoryBuilder (freeze=false) requires "
                "ctx.config['_meta_edit_provider']. Either set freeze: true for "
                "frozen mode or wire a meta-edit provider in the pipeline."
            )

        current_cheatsheet = str(memory.payload.get("cheatsheet", "") or "")
        completed_attempts = [attempt for attempt in attempts if attempt.completion]

        for attempt in completed_attempts:
            prompt = (
                self._curator_prompt
                .replace("[[QUESTION]]", attempt.prompt)
                .replace("[[MODEL_ANSWER]]", attempt.completion)
                .replace("[[PREVIOUS_CHEATSHEET]]", current_cheatsheet)
            )
            try:
                completions = provider.generate(
                    prompt,
                    model=getattr(provider, "model", ""),
                )
            except Exception as exc:
                memory.metadata["dc_update_error"] = f"{type(exc).__name__}: {exc}"
                continue
            response = str(completions[0]) if completions else ""
            current_cheatsheet = _cap_words(
                _extract_cheatsheet(response, current_cheatsheet),
                self.max_words,
            )
            memory.payload["cheatsheet"] = current_cheatsheet
        return memory

    def consolidate(self, ctx: RunContext, memory: MemoryState) -> MemoryState:
        memory.payload["cheatsheet"] = _cap_words(
            str(memory.payload.get("cheatsheet", "") or ""),
            self.max_words,
        )
        return memory
