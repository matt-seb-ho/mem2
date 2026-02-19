from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mem2.core.entities import AttemptRecord, FeedbackRecord, ProblemSpec, RetrievalBundle
from mem2.prompting.options import ArcMemoPromptOptions

MAX_LINE_WIDTH = 120

ARC_INTRO = """### Introduction
Given a puzzle containing input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Within a puzzle, each pair follows the same transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and is often the background color.
"""

EXAMPLE_GRIDS_INTRO = "Here are the input and output grids for the reference examples:"

HINT_TEMPLATE_MIN = """### Hints
{hints}
"""

HINT_TEMPLATE_SELECTED = """### Hints
We distilled some lessons or takeaways from previously solved ARC puzzles. Here are some lessons we selected that may or may not be relevant to this puzzle:
{hints}
"""

HINT_TEMPLATE_ALL = """### Hints
We distilled some lessons or takeaways from previously solved ARC puzzles. Each lesson is formatted with a "situation" component describing when to apply this lesson, and a "suggestion" component describing what might be a good idea to do in this situation. Here are the lessons:
{hints}
"""

HINT_TEMPLATE_OP3 = """\
### Concepts from Previously Solved Puzzles
We recorded concepts about structures and routines we observed in previously solved puzzles. These concepts may or may not be relevant to this puzzle, but they provide useful context to show examples of what structures may appear in the grids, what operations may be used, and how they might be composed. Concepts are annotated with fields like:
- cues: (short for "relevance cues"), what to look for that might indicate this concept is relevant in this puzzle
- implementation: notes on how this concept was implemented in past solution programs
- output typing: what the output of this routine is (e.g. a grid, a list, a number, a bool, etc.)
- parameters: a list of parameters that describe ways the concept may vary
We also have some recommendations on how to approach problem solving with these concepts in mind:
- We label the grid manipulation routines separately-- these directly affect the grids so they are easier to spot (along with structure concepts)
- You might try to first identify which grid manipulation operations are used, then investigate their parameters
- The non-grid manipulation routines might describe ways we've seen previous puzzles set parameters, so you can look to these for inspiration
- There may not be exact matches to this list, so we encourage you to think about variations, novel ways to recombine existing ideas, as well as completely new concepts
- These concepts and this approach are only suggestions, use them as you see fit

{hints}"""

HINT_TEMPLATES = {
    "min": HINT_TEMPLATE_MIN,
    "selected": HINT_TEMPLATE_SELECTED,
    "all_hints": HINT_TEMPLATE_ALL,
    "op3": HINT_TEMPLATE_OP3,
}

DESCRIPTION_TEMPLATE = """\
### External Puzzle Description
We queried other sources for descriptions of what they observe/speculate about this puzzle. These external sources are not authoritative (they may be wrong or incomplete), but they may provide useful context or insights. Here is the description(s):
{description}"""

CODE_INSTR_DEFAULT = """### Instructions
Write a Python function `transform` that converts a given input grid to its corresponding output grid based on the pattern observed in the reference examples. 
- Your function should have signature: `def transform(input_grid: np.ndarray) -> np.ndarray:`
- The input and output grids are 2D numpy arrays of integers.
- Each grid cell is assigned a **color**, represented by an **integer**. These integers **do not have numerical meaning**; they **should not** be used in arithmetic operations. Think of these integers as **labels** rather than numbers.
- Write your final code response in a markdown python code block (be sure to have "```python" in the line before your code and "```" in the line after your code).
- No need to provide examples of the function call, just the function definition is sufficient."""

HINT_CITATION_EXTRA_INSTRUCTION = """\
- Please also indicate which hints were useful in the solution process
  - The hints are numbered, please provide a comma-separated list inside <hint_citations> </hint_citations> tags.
  - For example, if you found hints 1 and 3 useful, you would write: <hint_citations>1, 3</hint_citations>."""

CODE_INSTR_DICT = {
    "default": CODE_INSTR_DEFAULT,
}

OUTPUT_MISMATCH_FEEDBACK_HEADER = """**Output Mismatches**
The puzzle provides reference examples containing input and output pairs. Here are the outputs from your previous attempt's code that did not match the expected output:"""

RETRY_INSTRUCTION_BODY = (
    "Please reflect on the above issues (code formatting, code execution error, "
    "or grid outputs differing from the expected/correct example outputs), "
    "and revise your reasoning, transformation rule hypothesis, or code accordingly. "
    "Please reflect on the your previous response and consider whether your transformation rule "
    "hypothesis is incorrect or if the code implementation is flawed."
)

_PROBLEM_DATA_CACHE: dict[str, dict[str, Any]] = {}


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_problem_data(problem_data_path: str | None) -> dict[str, Any]:
    if not problem_data_path:
        return {}
    key = str(problem_data_path)
    cached = _PROBLEM_DATA_CACHE.get(key)
    if cached is not None:
        return cached
    path = Path(problem_data_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    _PROBLEM_DATA_CACHE[key] = payload
    return payload


def _problem_data_variant(
    value: Any,
    *,
    variant_key: str | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if variant_key and isinstance(value.get(variant_key), dict):
        return value[variant_key]
    if isinstance(value.get("0"), dict):
        return value["0"]
    for v in value.values():
        if isinstance(v, dict):
            return v
    return {}


def format_grid_numpy(grid: list[list[int]] | np.ndarray) -> str:
    arr = np.array(grid, dtype=int)
    return np.array2string(arr, separator=", ", max_line_width=MAX_LINE_WIDTH)


def format_problem_for_prompt(problem: ProblemSpec) -> str:
    blocks: list[str] = ["### Your Puzzle Grids", EXAMPLE_GRIDS_INTRO]
    for idx, pair in enumerate(problem.train_pairs, start=1):
        blocks.append(f"Example {idx}")
        blocks.append("Input:")
        blocks.append(format_grid_numpy(pair["input"]))
        blocks.append("Output:")
        blocks.append(format_grid_numpy(pair["output"]))
        blocks.append("")
    blocks.append("Here is the input grid for the test example:")
    blocks.append("Input:")
    for pair in problem.test_pairs:
        blocks.append(format_grid_numpy(pair["input"]))
    return "\n".join(blocks)


def make_initial_prompt(
    problem: ProblemSpec,
    retrieval: RetrievalBundle | None,
    options: ArcMemoPromptOptions | None = None,
) -> str:
    options = options or ArcMemoPromptOptions()
    problem_data = _load_problem_data(options.problem_data)
    puzzle_data = _problem_data_variant(
        problem_data.get(problem.uid),
        variant_key=options.problem_data_variant_key,
    )

    configured_hint = puzzle_data.get("hint") if isinstance(puzzle_data.get("hint"), str) else None
    configured_desc = (
        puzzle_data.get("description") if isinstance(puzzle_data.get("description"), str) else None
    )

    effective_hint = configured_hint or (retrieval.hint_text if retrieval else None)
    instructions = CODE_INSTR_DICT.get(options.instruction_key, CODE_INSTR_DEFAULT)
    if effective_hint and options.require_hint_citations:
        instructions = f"{instructions}\n{HINT_CITATION_EXTRA_INSTRUCTION}"

    hint_block: str | None = None
    if effective_hint and options.include_hint:
        tmpl = HINT_TEMPLATES.get(options.hint_template_key, HINT_TEMPLATE_SELECTED)
        hint_block = tmpl.format(hints=effective_hint)

    desc_block: str | None = None
    if configured_desc:
        desc_block = DESCRIPTION_TEMPLATE.format(description=configured_desc)

    components = [
        ARC_INTRO,
        format_problem_for_prompt(problem),
        "",
        instructions,
        "",
        desc_block,
        hint_block,
    ]
    return "\n".join(x for x in components if x)


def _feedback_outcomes(
    fb: FeedbackRecord,
    *,
    error_feedback: str,
) -> tuple[list[str], list[tuple[int, str]]]:
    md = fb.metadata or {}
    errors = list(md.get("errors", []))
    mismatches = list(md.get("mismatches", []))
    if error_feedback == "first":
        errors = errors[:1]
        mismatches = mismatches[:1]

    def _normalize_example_idx(value: object) -> int:
        if isinstance(value, int):
            return max(1, value)
        if isinstance(value, str):
            tokens = value.replace(":", " ").split()
            for tok in reversed(tokens):
                if tok.isdigit():
                    return max(1, int(tok))
        return 1

    pretty_mismatches: list[tuple[int, str]] = []
    for m in mismatches:
        ex_idx_raw: object
        output_obj: object
        if isinstance(m, dict):
            ex_idx_raw = m.get("example_idx", m.get("pair_idx", m.get("label", 1)))
            output_obj = m.get("output", None)
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            ex_idx_raw = m[0]
            output_obj = m[1]
        else:
            ex_idx_raw = 1
            output_obj = m

        if output_obj is None or isinstance(output_obj, str):
            output = str(output_obj)
        else:
            output = format_grid_numpy(output_obj)
        pretty_mismatches.append((_normalize_example_idx(ex_idx_raw), output))
    return errors, pretty_mismatches


def make_retry_prompt(
    initial_prompt: str,
    attempts: list[AttemptRecord],
    feedback: list[FeedbackRecord],
    *,
    error_feedback: str = "all",
    num_feedback_passes: int = 1,
    include_past_outcomes: bool = True,
    new_concepts: str | None = None,
) -> str:
    if num_feedback_passes == -1:
        attempt_slice = attempts
        offset = 0
    else:
        attempt_slice = attempts[-num_feedback_passes:]
        offset = max(0, len(attempts) - len(attempt_slice))

    blocks: list[str] = []
    for local_idx, att in enumerate(attempt_slice, start=offset + 1):
        idx0 = local_idx - 1
        block = [f"#### Attempt {local_idx}", att.completion or ""]
        if idx0 < len(feedback):
            include_outcome = include_past_outcomes or (idx0 == len(attempts) - 1)
            if include_outcome:
                fb = feedback[idx0]
                errors, mismatches = _feedback_outcomes(fb, error_feedback=error_feedback)
                if errors:
                    block.append("**Execution / Parsing Errors**")
                    block.extend(f"- {e}" for e in errors)
                if mismatches:
                    block.append(OUTPUT_MISMATCH_FEEDBACK_HEADER)
                    for ex_idx, out in mismatches:
                        block.append(f"- Example {ex_idx}:\n{out}")
                if not errors and not mismatches and fb.content:
                    block.append(fb.content)
        blocks.append("\n".join(block))

    history = "\n\n---\n\n".join(blocks) if blocks else "No previous attempts recorded."
    components = [
        initial_prompt,
        "",
        "### Your Previous Response(s) and Outcomes",
        history,
        "",
        "### New Instructions",
        RETRY_INSTRUCTION_BODY,
    ]
    if new_concepts:
        components.append("")
        components.append(
            "### Reselected Lessons\n"
            "Here are reselected lessons that may or may not be helpful for solving this puzzle:\n"
            f"{new_concepts}"
        )
    return "\n".join(components)
