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
(a) a mathematical technique or theorem application (e.g. Chinese Remainder Theorem, \
Vieta's Formulas, generating functions)
(b) how certain parameters or values are determined (e.g. finding bounds via AM-GM)
(c) properties that are checked or leveraged (e.g. divisibility, parity, symmetry)
(d) a specific computational strategy (e.g. casework by residues, recursive enumeration)

# Instructions
Pseudocode:
- write the pseudocode translation inside <pseudocode> and </pseudocode> tags
- focus on the **mathematical reasoning** — name the theorems, identities, and \
techniques used rather than describing Python operations
- replace generic code patterns with their mathematical names \
(e.g. "apply Euler's totient" not "loop and count coprimes")
- be concise without compromising correctness
Summary:
- write a one-liner summary of the solution approach inside <summary> and </summary> tags

# Example

Problem:
How many positive integers less than 1000 are divisible by exactly one of 7 and 11?

Solution:
```python
def solve():
    count = 0
    for n in range(1, 1000):
        d7 = n % 7 == 0
        d11 = n % 11 == 0
        if d7 != d11:
            count += 1
    return count
```

<pseudocode>
count = |multiples of 7 below 1000| + |multiples of 11 below 1000| - 2 * |multiples of 77 below 1000|
Apply inclusion-exclusion: |A XOR B| = |A| + |B| - 2|A ∩ B|
Each count via floor division: floor(999/k)
</pseudocode>

<summary>Inclusion-exclusion on divisibility sets with floor-division counting</summary>

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


_MATH_REASONING_PSEUDOCODE_PROMPT = """\
# Introduction
We are analyzing correctly solved competition math problems to build a reusable \
concept library.  Your task is to analyze a mathematical reasoning solution, \
extract its key steps as concise pseudocode, and write a one-liner summary.

A concept can encode any of:
(a) a mathematical technique or theorem application (e.g. Chinese Remainder Theorem, \
Vieta's Formulas, generating functions)
(b) how certain parameters or values are determined (e.g. finding bounds via AM-GM)
(c) properties that are checked or leveraged (e.g. divisibility, parity, symmetry)
(d) a specific computational strategy (e.g. casework by residues, complementary counting)

# Instructions
Pseudocode:
- write the pseudocode translation inside <pseudocode> and </pseudocode> tags
- focus on the **mathematical reasoning steps** — name the theorems, identities, \
and techniques used
- be concise without compromising correctness
Summary:
- write a one-liner summary of the solution approach inside <summary> and </summary> tags

# Your Problem Solution
Analyze, abstract into pseudocode, and summarize the following solution:

Problem:
{problem_text}

Solution:
{solution_code}
"""


def build_pseudocode_prompt(
    problem: SolvedProblem,
    domain: str,
    stage1_mode: str = "code",
) -> str:
    """Stage 1: build prompt that converts solution → pseudocode + summary.

    Args:
        stage1_mode: "code" (default) — solution is Python code, translate to pseudocode.
                     "reasoning" — solution is mathematical reasoning, summarize into pseudocode.
                     "passthrough" — skip Stage 1, use solution text directly as pseudocode.
    """
    if stage1_mode == "passthrough":
        raise ValueError("passthrough mode should skip Stage 1 entirely")
    if domain == "math" and stage1_mode == "reasoning":
        template = _MATH_REASONING_PSEUDOCODE_PROMPT
    elif domain == "math":
        template = _MATH_PSEUDOCODE_PROMPT
    else:
        template = _CODE_PSEUDOCODE_PROMPT
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
(a) a theorem or identity -- a named mathematical result (e.g. "Euler's Totient Theorem", \
"Vieta's Formulas", "Hockey Stick Identity")
(b) a technique -- a specific method for a class of problems (e.g. "Generating Functions", \
"Roots of Unity Filter", "Simon's Favorite Factoring Trick")
(c) a computational strategy -- a concrete approach to a calculation (e.g. "Casework by Residue Class", \
"Complementary Counting", "Recursive Enumeration with Memoization")
(d) a definition -- a term for a recurring structure (e.g. "Primitive Root", "Derangement")

Mathematical solutions are sequences of reasoning steps. Each step applies a technique \
or theorem, possibly with problem-specific parameters.

# Concept Naming Rules
- **Be specific**: use the established mathematical name when one exists
- **Avoid generic catch-alls**: concepts like "algorithm", "algebraic manipulation", \
"constraint satisfaction", "system of equations", or "modular arithmetic" are too broad \
to be useful — they match too many problems and provide no actionable guidance
- A good concept name should suggest a *specific action* a solver should take, not \
merely categorize the problem
- Bad examples: "number theory tool", "optimization", "counting method"
- Good examples: "Chicken McNugget Theorem", "Stars and Bars", "Burnside's Lemma", \
"Binary Search on Answer", "Pigeonhole Principle"
- If no standard name exists, create a descriptive 2-5 word name that captures \
the specific technique (e.g. "Digit Sum Modular Reduction", "Greedy Interval Packing")
- **Extract 2-4 concepts per problem** — only specific, actionable ones

# Instructions
- Format your final concept list inside a fenced yaml markdown block \
(first line = "```yaml" and last line = "```")
- Feel free to think before writing your final response
- Each concept entry should have these fields:
    concept: the specific technique/theorem/pattern name
    kind: one of "theorem", "technique", "strategy", "definition"
    description: (optional) one-sentence elaboration if the name is not self-evident
    parameters: (optional) list of {{name, typing, description}} for the variable \
aspects of this concept
    cues: 2-4 specific problem features that suggest this concept is relevant \
(these should be concrete and observable, not tautological)
    implementation: 1-2 notes on the general procedure for applying this concept \
(describe the method abstractly, not what happened in this particular problem)
- **CRITICAL: No problem-specific values.** Strip ALL concrete numbers, coordinates, \
coefficients, and computed answers from cues and implementation. Write abstract patterns only.
    - BAD cue: "center is at (4,4) for a 7×7 grid"
    - GOOD cue: "problem has a grid with a natural center point"
    - BAD implementation: "compute 160 miles ÷ 5 hours = 32 mph"
    - GOOD implementation: "compute speed = distance ÷ time"
    - BAD implementation: "ACM: base CM = 2, height = 8 → area = 8"
    - GOOD implementation: "compute triangle area = ½ × base × height using perpendicular distance"
- **Reuse concepts whenever possible**
    - check existing concepts in the `Concept Repository` section below
    - if an existing concept matches what you see, reuse its exact name
    - when reusing, you may omit `kind`, `description`, and `parameters`
    - you can still add new `cues` and `implementation` entries when reusing
- Cue quality guidelines:
    - Good cue: "problem asks for count of integers with specific digit properties"
    - Bad cue: "a mathematical technique is needed" (tautological)
    - Good cue: "expression involves product of consecutive integers"
    - Bad cue: "algebraic manipulation is required" (too vague)

# Example

Pseudocode:
```
Count lattice paths from (0,0) to (7,3) avoiding y=x+1
Total unrestricted paths: C(10,3)
Reflected bad paths: paths from (1,-1) to (7,3) = C(10,4) by reflection across y=x+1
Answer = C(10,3) - C(10,4)
```

```yaml
- concept: Lattice Path Counting
  kind: technique
  description: count paths on integer grid using binomial coefficients
  parameters:
    - name: grid dimensions
      typing: tuple[int, int]
      description: the (width, height) of the lattice to traverse
    - name: step set
      typing: list[tuple[int, int]]
      description: allowed moves (e.g. right and up)
  cues:
    - problem involves moving on a grid with restricted steps
    - answer expressible as binomial coefficient C(m+n, n)
  implementation:
    - total paths from origin to target with unit steps = C(width+height, height)

- concept: Reflection Principle for Lattice Paths
  kind: technique
  description: subtract reflected bad paths to enforce boundary constraints on lattice paths
  parameters:
    - name: boundary
      typing: str
      description: the line constraint that paths must not cross (e.g. y = x + 1)
  cues:
    - lattice path problem with a boundary that cannot be crossed
    - problem mentions staying below/above a diagonal line
  implementation:
    - reflect start point across boundary, count unrestricted paths from reflected point
    - bad paths biject to reflected paths via first-touch reflection
```

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
(a) an algorithm -- a named algorithmic technique (e.g. "Dijkstra's Algorithm", \
"KMP String Matching", "Tarjan's SCC")
(b) a data structure -- a specific data organization (e.g. "Segment Tree with Lazy Propagation", \
"DSU with Path Compression", "Monotonic Stack")
(c) a technique -- a concrete optimization or pattern (e.g. "Coordinate Compression", \
"Two Pointers on Sorted Array", "Bitmask DP over Subsets")
(d) a definition -- a term for a recurring problem structure (e.g. "Euler Tour", "Bridge Edge")

# Concept Naming Rules
- **Be specific**: use the established algorithm/DS name when one exists
- **Avoid generic catch-alls**: concepts like "dynamic programming", "graph traversal", \
"greedy algorithm", or "sorting" are too broad — they match too many problems
- A good concept name should suggest a *specific technique* to apply
- Bad: "optimization", "data structure usage", "string processing"
- Good: "Knuth's Optimization for 1D DP", "Centroid Decomposition", "Z-Algorithm"
- **Extract 2-4 concepts per problem** — only specific, actionable ones

# Instructions
- Format your final concept list inside a fenced yaml markdown block \
(first line = "```yaml" and last line = "```")
- Feel free to think before writing your final response
- **CRITICAL: YAML formatting rules**: Do NOT use backtick characters (`) anywhere in your \
YAML output. Use plain text descriptions instead of code formatting.
    - BAD: "implementation: `dp[i]` stores the minimum cost"
    - GOOD: "implementation: dp[i] stores the minimum cost"
- Each concept entry should have these fields:
    concept: the specific algorithm/data structure/technique name
    kind: one of "algorithm", "data structure", "technique", "definition"
    description: (optional) one-sentence elaboration if the name is not self-evident
    parameters: (optional) list of {{name, typing, description}} for the variable \
aspects of this concept
    cues: 2-4 specific problem features that suggest this concept is relevant
    implementation: 1-2 notes on the general procedure for applying this concept \
(describe the method abstractly, not what happened in this particular problem)
- **CRITICAL: No problem-specific values.** Strip ALL concrete numbers, array sizes, \
variable names from cues and implementation. Write abstract patterns only.
    - BAD cue: "array has 200000 elements and queries"
    - GOOD cue: "large array with range queries requiring sublinear time"
    - BAD implementation: "build segment tree over array a of size n"
    - GOOD implementation: "build segment tree for point updates and range queries"
- **Reuse concepts whenever possible**
    - check existing concepts in the `Concept Repository` section below
    - if an existing concept matches what you see, reuse its exact name
    - when reusing, you may omit `kind`, `description`, and `parameters`
    - you can still add new `cues` and `implementation` entries when reusing
- Cue quality guidelines:
    - Good cue: "problem asks for shortest path in weighted graph with non-negative edges"
    - Bad cue: "a graph algorithm is needed" (tautological)

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

    Shows concepts grouped by kind with descriptions and cues, so the LLM
    can judge whether to reuse an existing concept or create a new one.
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
            if c.cues:
                # Show up to 3 cues so LLM can judge reuse
                shown = c.cues[:3]
                cue_str = "; ".join(shown)
                if len(c.cues) > 3:
                    cue_str += f" (+{len(c.cues) - 3} more)"
                line += f"\n  cues: {cue_str}"
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
