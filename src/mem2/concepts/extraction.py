"""Concept extraction from solved math/code problems.

Reads build run artifacts, calls an LLM to extract typed PS-format concepts,
and assembles a ConceptMemory-compatible structure.
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
# YAML extraction regex (same pattern used in concept_selector / arcmemo_selector)
# ---------------------------------------------------------------------------
_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MATH_EXTRACTION_PROMPT = """\
You are analyzing a correctly solved competition math problem to extract reusable problem-solving concepts.

## Problem
{problem_text}

## Correct Solution
{solution_code}

## Task
Analyze this solution and extract the key mathematical concepts, techniques, and patterns used. For each concept, provide structured annotations.

Output a YAML block with this structure:
```yaml
summary: <one-line summary of the solution approach>
pseudocode: |
  <step-by-step pseudocode of the solution logic>
concepts:
  - concept: <concept name - a short descriptive name>
    kind: <category - discover organically, e.g. "technique", "theorem", "identity", "counting method", "algebraic manipulation", "number theory tool", etc.>
    description: <what this concept does or represents>
    parameters:
      - name: <parameter name>
        typing: <parameter type>
        description: <what this parameter controls>
    cues:
      - <when to consider using this concept - what problem features suggest it>
    implementation:
      - <how this concept was applied in this specific solution>
```

Example:
```yaml
summary: Uses combinatorics to count favorable outcomes and compute a probability ratio.
pseudocode: |
  1. Compute total ways to draw 4 slips from 40: C(40,4)
  2. Count favorable outcomes for event p: 10 numbers * C(4,4) each
  3. Count favorable outcomes for event q: C(10,2) pairs * C(4,2)^2
  4. Return q/p = favorable_q / favorable_p
concepts:
  - concept: combinatorial counting
    kind: technique
    description: Count outcomes by decomposing into independent choices and multiplying
    parameters:
      - name: n
        typing: int
        description: total items to choose from
      - name: k
        typing: int
        description: number of items to choose
    cues:
      - Problem asks to count arrangements, selections, or probability of discrete events
      - Multiple independent selection steps can be identified
    implementation:
      - Used C(40,4) for total outcomes, C(10,2) to choose number pairs, C(4,2) to choose slips per number
```

Guidelines:
- Extract 1-5 concepts per problem (only meaningful ones)
- Name concepts clearly and concisely (e.g. "modular arithmetic", "pigeonhole principle", "generating function")
- Discover concept kinds organically - use whatever category naturally fits
- Cues should describe problem features that suggest this concept is relevant
- Implementation notes should be specific to this solution
- Parameters are optional - only include if the concept has meaningful parameters
"""

CODE_EXTRACTION_PROMPT = """\
You are analyzing a correctly solved competitive programming problem to extract reusable algorithmic concepts.

## Problem
{problem_text}

## Correct Solution
{solution_code}

## Task
Analyze this solution and extract the key algorithmic concepts, data structures, and implementation patterns used. For each concept, provide structured annotations.

Output a YAML block with this structure:
```yaml
summary: <one-line summary of the solution approach>
pseudocode: |
  <step-by-step pseudocode of the solution logic>
concepts:
  - concept: <concept name - a short descriptive name>
    kind: <category - discover organically, e.g. "algorithm", "data structure", "technique", "optimization", "graph method", "dp pattern", "string algorithm", etc.>
    description: <what this concept does or represents>
    parameters:
      - name: <parameter name>
        typing: <parameter type>
        description: <what this parameter controls>
    cues:
      - <when to consider using this concept - what problem features suggest it>
    implementation:
      - <how this concept was applied in this specific solution>
```

Example:
```yaml
summary: Uses prefix sums to efficiently compute subarray sums and find the answer.
pseudocode: |
  1. Read array of N integers
  2. Build prefix sum array P where P[i] = sum(A[0..i-1])
  3. For each query (l, r), answer is P[r+1] - P[l]
  4. Output results
concepts:
  - concept: prefix sum array
    kind: technique
    description: Precompute cumulative sums to answer range-sum queries in O(1)
    parameters:
      - name: array
        typing: list[int]
        description: the input array to build prefix sums over
    cues:
      - Problem involves multiple range sum queries over a static array
      - Need to compute sum of contiguous subarray efficiently
    implementation:
      - Built prefix sum array of size N+1, used difference of two prefix values for each query
```

Guidelines:
- Extract 1-5 concepts per problem (only meaningful ones)
- Name concepts clearly and concisely (e.g. "binary search", "union-find", "sliding window", "topological sort")
- Discover concept kinds organically - use whatever category naturally fits
- Cues should describe problem features that suggest this concept is relevant
- Implementation notes should be specific to this solution
- Parameters are optional - only include if the concept has meaningful parameters
"""


# ---------------------------------------------------------------------------
# Loading solved problems from a build run directory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_extraction_prompt(problem: SolvedProblem, domain: str) -> str:
    """Build domain-specific extraction prompt for a solved problem."""
    template = MATH_EXTRACTION_PROMPT if domain == "math" else CODE_EXTRACTION_PROMPT
    return template.format(
        problem_text=problem.problem_text,
        solution_code=problem.solution_code,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_extraction_response(response: str) -> dict | None:
    """Extract and parse YAML block from LLM response.

    Returns dict with 'summary', 'pseudocode', 'concepts' or None on failure.
    """
    m = _YAML_BLOCK_RE.search(response)
    if not m:
        # Try parsing the whole response as YAML
        yaml_text = response.strip()
    else:
        yaml_text = m.group(1).strip()

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        logger.warning(f"YAML parse error: {exc}")
        return None

    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict from YAML, got {type(parsed)}")
        return None

    # Validate required fields
    if "concepts" not in parsed or not isinstance(parsed.get("concepts"), list):
        logger.warning("Missing or invalid 'concepts' field in extraction response")
        return None

    return parsed


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_concept_memory(
    extractions: list[tuple[str, dict]],
) -> ConceptMemory:
    """Assemble a ConceptMemory from parsed extraction results.

    Parameters
    ----------
    extractions : list of (problem_uid, parsed_extraction_dict) tuples

    Returns
    -------
    ConceptMemory with concepts and solutions populated.

    Note: bypasses ConceptMemory.write_concept() which rejects kinds not in
    {"structure", "routine"}. We directly construct Concept objects to allow
    organic kind discovery.
    """
    mem = ConceptMemory()

    for problem_uid, extraction in extractions:
        # Record solution
        mem.solutions[problem_uid] = ProblemSolution(
            problem_id=problem_uid,
            solution=None,
            summary=extraction.get("summary"),
            pseudocode=extraction.get("pseudocode"),
        )

        # Process each concept annotation
        for ann in extraction.get("concepts", []):
            name = ann.get("concept")
            if not name:
                logger.info(f"[{problem_uid}] Skipping concept: missing 'concept' field")
                continue

            name = name.strip()
            kind = ann.get("kind", "technique").strip()

            if name in mem.concepts:
                # Merge into existing concept
                mem.concepts[name].update(problem_uid, ann)
            else:
                # Build parameters
                params: list[ParameterSpec] = []
                for p in ann.get("parameters") or []:
                    if isinstance(p, dict):
                        params.append(ParameterSpec(
                            name=p.get("name", ""),
                            typing=p.get("typing"),
                            description=p.get("description"),
                        ))

                # Construct concept directly (bypassing write_concept kind filter)
                concept = Concept(
                    name=name,
                    kind=kind,
                    parameters=params,
                    description=ann.get("description"),
                )
                concept.update(problem_uid, ann)
                mem.concepts[name] = concept
                mem.categories[kind].append(name)

    return mem
