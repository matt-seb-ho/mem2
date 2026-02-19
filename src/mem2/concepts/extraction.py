"""Two-stage concept extraction from solved math/code problems.

Follows the ARC pipeline architecture:
  Stage 1: solution code → pseudocode + summary  (per problem, one LLM call)
  Stage 2: pseudocode → concepts                 (batched, with concept repo in prompt)

Stage 2 passes the *current* concept repository into each prompt so the LLM
can reuse existing concept names rather than inventing duplicates.  After each
batch, newly extracted concepts are written into memory, so later batches see
a growing repository — exactly matching the ARC offline extraction flow.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import orjson
import yaml

from mem2.concepts.data import Concept, ParameterSpec
from mem2.concepts.memory import ConceptMemory, ProblemSolution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SolvedProblem:
    uid: str
    problem_text: str
    solution_code: str


# ---------------------------------------------------------------------------
# YAML / tag extraction helpers
# ---------------------------------------------------------------------------
_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>.  Returns '' on failure."""
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _extract_yaml_block(text: str) -> str | None:
    """Extract first ```yaml ... ``` block from text."""
    m = _YAML_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


# =========================================================================
# Stage 1: Solution → Pseudocode + Summary
# =========================================================================

# -- Prompt templates -----------------------------------------------------

_MATH_PSEUDOCODE_PROMPT = """\
# Introduction
We are analyzing correctly solved competition math problems to build a reusable \
concept library.  Your task is to analyze a solution, rewrite it as pseudocode \
that can more easily be abstracted into concepts, and write a one-liner summary \
of the solution approach.

A concept can encode any of:
(a) a mathematical technique or theorem application
(b) how certain parameters or values are determined
(c) properties that are checked or leveraged
(d) an algorithmic pattern used in the computation

# Instructions
Pseudocode:
- write the pseudocode translation inside <pseudocode> and </pseudocode> tags
- be concise without compromising correctness
- emphasize the mathematical reasoning steps and techniques over implementation details
Summary:
- write a one-liner summary of the solution approach inside <summary> and </summary> tags

# Your Problem Solution
Analyze, abstract into pseudocode, and summarize the following solution:

Problem:
{problem_text}

Solution:
```python
{solution_code}
```
"""

_CODE_PSEUDOCODE_PROMPT = """\
# Introduction
We are analyzing correctly solved competitive programming problems to build a \
reusable concept library.  Your task is to analyze a solution, rewrite it as \
pseudocode that can more easily be abstracted into concepts, and write a \
one-liner summary of the solution approach.

A concept can encode any of:
(a) an algorithmic technique or data structure usage
(b) how certain parameters or values are determined
(c) properties that are checked or leveraged
(d) an optimization or implementation pattern

# Instructions
Pseudocode:
- write the pseudocode translation inside <pseudocode> and </pseudocode> tags
- be concise without compromising correctness
- emphasize algorithmic operations and data structure choices over I/O details
Summary:
- write a one-liner summary of the solution approach inside <summary> and </summary> tags

# Your Problem Solution
Analyze, abstract into pseudocode, and summarize the following solution:

Problem:
{problem_text}

Solution:
```python
{solution_code}
```
"""


def build_pseudocode_prompt(problem: SolvedProblem, domain: str) -> str:
    """Stage 1: build prompt that converts solution → pseudocode + summary."""
    template = _MATH_PSEUDOCODE_PROMPT if domain == "math" else _CODE_PSEUDOCODE_PROMPT
    return template.format(
        problem_text=problem.problem_text,
        solution_code=problem.solution_code,
    )


def parse_pseudocode_response(response: str) -> tuple[str, str]:
    """Parse Stage 1 LLM output.  Returns (pseudocode, summary)."""
    pseudocode = _extract_tag(response, "pseudocode")
    summary = _extract_tag(response, "summary")
    if not pseudocode:
        logger.warning("Stage 1: no <pseudocode> tag found in response")
    if not summary:
        logger.warning("Stage 1: no <summary> tag found in response")
    return pseudocode, summary


# =========================================================================
# Stage 2: Pseudocode → Concepts  (with concept repo in prompt)
# =========================================================================

# -- Prompt templates -----------------------------------------------------

_MATH_CONCEPT_PROMPT = """\
# Introduction
We are analyzing correctly solved competition math problems to build a reusable \
concept library.  Your task is to analyze a solution (rendered as pseudocode) \
and abstract out reusable concepts.

A concept encodes one of the following:
(a) a mathematical technique -- a method or approach for solving a class of problems
(b) a theorem or identity -- a known result that can be applied
(c) an algorithmic pattern -- a computational strategy (e.g. enumeration, recursion)
(d) a definition -- a term for a recurring structure or phenomenon

Programs can be viewed as a sequence of reasoning steps.  To construct a solution, \
we compose steps and determine what parameters to use for each.

# Instructions
- Format your final concept list inside a fenced yaml markdown block \
(first line = "```yaml" and last line = "```")
- Feel free to think before writing your final response
- Each concept entry should have these fields:
    concept: technique name / theorem name / pattern name
    kind: discover organically (e.g. "technique", "theorem", "identity", \
"counting method", "algebraic manipulation", "number theory tool", "algorithm", etc.)
    description: (optional) elaborate if the name is not self-evident
    parameters: list of {{name, typing, description}} if the concept has meaningful parameters
    cues: list of problem features that suggest this concept is relevant
    implementation: list of how this concept was applied in this specific solution
- **Reuse concepts whenever possible**
    - check existing concepts in the `Concept Repository` section below
    - if an existing concept matches what you see, reuse its exact name
    - when reusing a concept, you may omit `kind` and `description` (only fill `concept`)
    - you can still add new `cues` and `implementation` entries when reusing
- Distinct concepts must have different names
- Extract 1-5 concepts per problem (only meaningful ones)

# Concept Repository
Here is the current concept repository.  Check for reuse before creating new concepts:
{concept_list}

# Your Problem Solution
Abstract the following solution into a concept list:
```
{pseudocode}
```
"""

_CODE_CONCEPT_PROMPT = """\
# Introduction
We are analyzing correctly solved competitive programming problems to build a \
reusable concept library.  Your task is to analyze a solution (rendered as \
pseudocode) and abstract out reusable concepts.

A concept encodes one of the following:
(a) an algorithm -- a named algorithmic technique (e.g. binary search, BFS)
(b) a data structure -- a data organization used (e.g. segment tree, union-find)
(c) a technique -- an implementation pattern or optimization (e.g. two pointers, prefix sums)
(d) a definition -- a term for a recurring problem structure

Programs can be viewed as a sequence of algorithmic steps.  To construct a solution, \
we compose steps and determine what parameters to use for each.

# Instructions
- Format your final concept list inside a fenced yaml markdown block \
(first line = "```yaml" and last line = "```")
- Feel free to think before writing your final response
- Each concept entry should have these fields:
    concept: algorithm name / data structure name / technique name
    kind: discover organically (e.g. "algorithm", "data structure", "technique", \
"optimization", "graph method", "dp pattern", "string algorithm", etc.)
    description: (optional) elaborate if the name is not self-evident
    parameters: list of {{name, typing, description}} if the concept has meaningful parameters
    cues: list of problem features that suggest this concept is relevant
    implementation: list of how this concept was applied in this specific solution
- **Reuse concepts whenever possible**
    - check existing concepts in the `Concept Repository` section below
    - if an existing concept matches what you see, reuse its exact name
    - when reusing a concept, you may omit `kind` and `description` (only fill `concept`)
    - you can still add new `cues` and `implementation` entries when reusing
- Distinct concepts must have different names
- Extract 1-5 concepts per problem (only meaningful ones)

# Concept Repository
Here is the current concept repository.  Check for reuse before creating new concepts:
{concept_list}

# Your Problem Solution
Abstract the following solution into a concept list:
```
{pseudocode}
```
"""


def render_concept_repo(mem: ConceptMemory) -> str:
    """Render the current concept memory as a text list for the Stage 2 prompt.

    Follows the ARC pattern: list concepts grouped by kind with descriptions,
    so the LLM can see what already exists and reuse names.
    """
    if not mem.concepts:
        return "(empty — no concepts extracted yet)"

    lines: list[str] = []
    for kind in sorted(mem.categories.keys()):
        names = mem.categories[kind]
        if not names:
            continue
        lines.append(f"## {kind}")
        for name in names:
            c = mem.concepts[name]
            line = f"- concept: {c.name}"
            if c.description:
                line += f"\n  description: {c.description}"
            lines.append(line)
        lines.append("")  # spacer
    return "\n".join(lines).rstrip()


def build_concept_prompt(pseudocode: str, domain: str, mem: ConceptMemory) -> str:
    """Stage 2: build prompt that extracts concepts from pseudocode.

    The current concept repository is rendered into the prompt so the LLM
    can reuse existing concept names.
    """
    template = _MATH_CONCEPT_PROMPT if domain == "math" else _CODE_CONCEPT_PROMPT
    return template.format(
        concept_list=render_concept_repo(mem),
        pseudocode=pseudocode,
    )


def parse_concept_response(response: str) -> list[dict]:
    """Parse Stage 2 LLM output.  Returns list of concept annotation dicts."""
    yaml_text = _extract_yaml_block(response)
    if not yaml_text:
        logger.warning("Stage 2: no YAML block found in response")
        return []

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        logger.warning(f"Stage 2 YAML parse error: {exc}")
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "concepts" in parsed:
        return parsed["concepts"]
    logger.warning(f"Stage 2: expected list of concepts, got {type(parsed)}")
    return []


# =========================================================================
# Writing concepts into ConceptMemory (bypassing kind filter)
# =========================================================================

def write_concept(mem: ConceptMemory, problem_uid: str, ann: dict) -> None:
    """Write a single concept annotation into memory.

    Bypasses ConceptMemory.write_concept() which rejects kinds not in
    {"structure", "routine"}.  Directly constructs Concept objects to allow
    organic kind discovery for math/code domains.
    """
    name = ann.get("concept") or ann.get("name")
    if not name:
        logger.info(f"[{problem_uid}] Skipping concept: missing 'concept' field")
        return

    name = name.strip()

    if name in mem.concepts:
        # Merge into existing concept
        mem.concepts[name].update(problem_uid, ann)
    else:
        kind = ann.get("kind", "technique").strip()

        # Build parameters
        params: list[ParameterSpec] = []
        for p in ann.get("parameters") or []:
            if isinstance(p, dict):
                params.append(ParameterSpec(
                    name=p.get("name", ""),
                    typing=p.get("typing"),
                    description=p.get("description"),
                ))

        concept = Concept(
            name=name,
            kind=kind,
            parameters=params,
            description=ann.get("description"),
        )
        concept.update(problem_uid, ann)
        mem.concepts[name] = concept
        mem.categories[kind].append(name)


# =========================================================================
# Loading solved problems from a build run directory
# =========================================================================

def load_solved_problems(run_dir: Path, domain: str) -> list[SolvedProblem]:
    """Read build run artifacts and return correctly solved problems.

    Joins problems.json + attempts.jsonl + eval_records.jsonl on
    (problem_uid, pass_idx) to find correct attempts.
    """
    run_dir = Path(run_dir)

    # Load problems
    problems_path = run_dir / "problems.json"
    problems = orjson.loads(problems_path.read_bytes())

    # Load attempts indexed by (problem_uid, pass_idx)
    attempts: dict[tuple[str, int], dict] = {}
    attempts_path = run_dir / "attempts.jsonl"
    for line in attempts_path.read_bytes().splitlines():
        if not line.strip():
            continue
        a = orjson.loads(line)
        key = (a["problem_uid"], a["pass_idx"])
        attempts[key] = a

    # Load eval records, find correct ones
    eval_path = run_dir / "eval_records.jsonl"
    correct_keys: set[tuple[str, int]] = set()
    for line in eval_path.read_bytes().splitlines():
        if not line.strip():
            continue
        e = orjson.loads(line)
        if e.get("is_correct"):
            pass_idx = e["metadata"].get("pass_idx", e.get("attempt_idx", 0))
            correct_keys.add((e["problem_uid"], pass_idx))

    # Build SolvedProblem list
    solved: list[SolvedProblem] = []
    seen_uids: set[str] = set()
    for (uid, pass_idx) in sorted(correct_keys):
        # Take first correct attempt per problem
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        attempt = attempts.get((uid, pass_idx))
        if attempt is None:
            logger.warning(f"No attempt found for correct eval: {uid} pass_idx={pass_idx}")
            continue

        problem = problems.get(uid)
        if problem is None:
            logger.warning(f"No problem entry for uid: {uid}")
            continue

        metadata = problem.get("metadata", {})
        if domain == "math":
            problem_text = metadata.get("problem_text", "")
        elif domain == "code":
            problem_text = metadata.get("question_content", "")
        else:
            problem_text = metadata.get("problem_text") or metadata.get("question_content", "")

        if not problem_text:
            logger.warning(f"No problem text for uid: {uid}")
            continue

        solved.append(SolvedProblem(
            uid=uid,
            problem_text=problem_text,
            solution_code=attempt["completion"],
        ))

    logger.info(f"Loaded {len(solved)} solved problems from {run_dir}")
    return solved
