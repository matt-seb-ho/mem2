"""On-policy corpus-global concept induction (the new system).

Turns dsv4f's OWN correct solves of the BARC seeds into a typed concept library,
with every judgement made by the model — no BARC annotations, no human few-shot
concept examples. See docs/onpolicy_concept_induction_plan.md.

Pipeline (each stage = LLM calls):
  A  per-solve   solution code   -> pseudocode + one-line summary
  B  per-solve   pseudocode      -> free-form concept tags + NL descriptions + kind
  C  corpus      all tags        -> canonical vocabulary  (map/reduce + bounded loop)
  D  per-concept member solves   -> typed Concept (ConceptMemory schema)

This module holds the ZERO-SHOT prompt builders + parsers + a thin async LLM
caller with token accounting. Orchestration lives in scripts/induce_library.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def extract_tag(text: str, tag: str) -> str:
    m = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL).search(text or "")
    return m.group(1).strip() if m else ""


def extract_yaml_block(text: str) -> Any | None:
    """Return parsed YAML from the first ```yaml block (or whole text as fallback)."""
    m = _YAML_BLOCK_RE.search(text or "")
    raw = m.group(1).strip() if m else (text or "").strip()
    if not raw:
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return None


# ===========================================================================
# Stage A — solution code -> pseudocode + summary  (zero-shot)
# ===========================================================================
# Derived from arc_memo's op3 pseudocode_instr.txt INTENT, but with the few-shot
# {examples}/{concepts} blocks removed — the model abstracts on its own.
STAGE_A_PROMPT = """\
# Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule \
that maps input grids to output grids. Each puzzle presents several input-output grid \
pairs as reference examples and the task is to predict the transformation rule. Grids \
are 2D integer arrays with integers representing colors; 0 is the black background.

We are learning from previously solved puzzles to improve future puzzle solving. Your \
task is to analyze one correct solution program, rewrite it as pseudocode that can be \
abstracted into reusable concepts, and write a one-line summary of the transformation \
rule. A concept can encode any of:
(a) grid manipulation: an operation that directly changes the output grid
(b) helper routine: specialized logic for parameterizing more abstract operations
(c) criteria: checked properties / logic for conditional operations
(d) structure: shapes/objects to look for in the pixel grids

# Instructions
- Write the pseudocode inside <pseudocode> and </pseudocode> tags.
- Focus on the broader ideas, not Python minutiae; name operations meaningfully.
- Prefer an explicit grid-object view (e.g. object.color / .shape / .size / .position) \
over raw array indexing where it clarifies intent.
- Be concise without losing correctness.
- Write a one-line summary of the transformation rule inside <summary> and </summary>.

# Your Puzzle Solution
Analyze, abstract into pseudocode, and summarize the following solution:
```python
{solution}
```"""


def build_stage_a_prompt(solution_code: str) -> str:
    return STAGE_A_PROMPT.format(solution=solution_code)


def parse_stage_a(completion: str) -> dict[str, str]:
    return {
        "pseudocode": extract_tag(completion, "pseudocode"),
        "summary": extract_tag(completion, "summary"),
    }


# ===========================================================================
# Stage B — pseudocode -> free-form concept tags + descriptions  (zero-shot)
# ===========================================================================
# Deliberately divergent: the model invents its OWN vocabulary. No shared concept
# list is provided. Global unification happens later in Stage C.
STAGE_B_PROMPT = """\
# Introduction
Consider "ARC" puzzles: a hidden rule maps input grids to output grids (2D integer \
color arrays, 0 = black background). We are building a library of reusable ideas by \
analyzing solved puzzles one at a time.

Below is the pseudocode and summary of ONE solved puzzle. Identify the reusable \
concepts it embodies. A concept is either:
- a "routine": an operation or piece of logic (e.g. tiling a sprite, flood fill, \
finding connected components, recoloring by size), or
- a "structure": a class of visual entities to look for in the grids (e.g. rectangle \
frame, line, symmetric object, noise pixels).

Invent clear, general names — do not tie them to this one puzzle. Favor names that \
could plausibly recur across many puzzles.

# Instructions
Output a YAML list inside a ```yaml fenced block. Each entry has:
  - tag:         a short, general concept name (lower case, words separated by spaces)
    kind:        routine | structure
    description: one sentence explaining the concept in general terms
    role:        one short phrase on how it was used in THIS puzzle
Only include genuinely reusable ideas; skip puzzle-specific trivia. 3-8 entries is typical.

# Puzzle
Summary: {summary}

Pseudocode:
{pseudocode}"""


def build_stage_b_prompt(pseudocode: str, summary: str) -> str:
    return STAGE_B_PROMPT.format(pseudocode=pseudocode or "(none)", summary=summary or "(none)")


def parse_stage_b(completion: str) -> list[dict[str, str]]:
    data = extract_yaml_block(completion)
    out: list[dict[str, str]] = []
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "")).strip().lower()
        if not tag:
            continue
        kind = str(item.get("kind", "")).strip().lower()
        out.append({
            "tag": tag,
            "kind": "structure" if kind.startswith("struct") else "routine",
            "description": str(item.get("description", "")).strip(),
            "role": str(item.get("role", "")).strip(),
        })
    return out


# ===========================================================================
# Stage C — corpus-global vocabulary unification (LLM-driven, map/reduce + loop)
# ===========================================================================
_STOPWORDS = {"a", "an", "the", "of", "to", "by", "via", "for", "with", "on", "in"}


def _singularize(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalize_tag(tag: str) -> str:
    """Lexical (NOT semantic) canonical key: lowercase, strip hyphens/punctuation,
    drop stopwords, singularize, sort words. Collapses pure surface variants like
    'separator line'/'separator lines'/'anti-diagonal line' — leaves semantic
    synonyms (e.g. 'connected components' vs 'object detection') for the LLM.
    """
    import re as _re
    t = _re.sub(r"[^a-z0-9\s]", " ", tag.lower())
    words = [_singularize(w) for w in t.split() if w and w not in _STOPWORDS]
    return " ".join(sorted(words))


def aggregate_tags(stage_b: dict[str, dict]) -> list[dict]:
    """Collapse Stage-B output into unique tag records, merging pure surface-form
    variants via normalize_tag. Returns [{tag, kind, count, uids[], descriptions[],
    surface_forms[]}] sorted by count desc. Semantic synonymy is left to the LLM.
    """
    from collections import defaultdict
    rec: dict[str, dict] = {}
    kind_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    surface_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for uid, v in stage_b.items():
        for t in v.get("tags", []):
            surface = t["tag"]
            key = normalize_tag(surface) or surface
            r = rec.setdefault(key, {"uids": [], "descriptions": []})
            if uid not in r["uids"]:
                r["uids"].append(uid)
            d = (t.get("description") or "").strip()
            if d and d not in r["descriptions"]:
                r["descriptions"].append(d)
            kind_votes[key][t.get("kind", "routine")] += 1
            surface_counts[key][surface] += 1
    out = []
    for key, r in rec.items():
        votes = kind_votes[key]
        r["kind"] = max(votes, key=votes.get) if votes else "routine"
        # display tag = most common surface form
        r["tag"] = max(surface_counts[key], key=surface_counts[key].get)
        r["surface_forms"] = sorted(surface_counts[key])
        r["count"] = len(r["uids"])
        out.append(r)
    out.sort(key=lambda r: (-r["count"], r["tag"]))
    return out


STAGE_C_MAP_PROMPT = """\
# Task
You are unifying the vocabulary of an ARC-puzzle concept library. Below is a list of \
candidate concept tags (each invented independently while analyzing one solved puzzle), \
with a representative description. Many are synonyms or near-duplicates of each other.

Group them into CANONICAL concepts. Be AGGRESSIVE about merging tags that denote the same \
underlying idea, even if worded differently — singular/plural, verb/noun, or more/less \
specific phrasings of the same thing (e.g. "connected components" / "connected component \
extraction" / "object detection"; "separator line" / "separator lines"; "tiling" / \
"pattern tiling" / "grid replication"). Aim to collapse this list substantially.

# Rules
- Pick a clear, general canonical name for each group (lower case, spaces).
- Still keep genuinely distinct ideas separate (e.g. "flood fill" vs "bounding box").
- Every input tag must belong to exactly one group.
- kind is routine (an operation/logic) or structure (a visual entity to look for).

# Output
A YAML list inside a ```yaml fenced block. Each entry:
  - canonical: canonical concept name
    kind: routine | structure
    gloss: one sentence describing the canonical concept
    members: [exact input tag strings that map to this concept]

# Candidate tags
{tag_block}"""


STAGE_C_REDUCE_PROMPT = """\
# Task
You are consolidating the vocabulary of an ARC-puzzle concept library. Below are \
concepts proposed independently while analyzing different puzzles. MANY denote the SAME \
underlying idea under different wording (singular/plural, verb/noun, more/less specific, \
synonyms). Aggressively merge these into a smaller set of general, reusable concepts.

# Guidance
- Merge whenever two entries describe the same operation or the same kind of visual \
structure, even if the wording differs. Examples that SHOULD merge: \
"separator line"/"separator lines"/"divider line"; "connected components"/"connected \
component extraction"/"object detection"; "recolor"/"color mapping"/"color substitution".
- Prefer a concise, general canonical name (lower case, spaces).
- Still keep genuinely different ideas apart (e.g. "flood fill" vs "bounding box").
- Every input concept must map to exactly one output concept.

# Output
A YAML list inside a ```yaml fenced block. Each entry:
  - canonical: final concept name
    kind: routine | structure
    gloss: one sentence describing the concept
    members: [exact input concept names that map here]

# Input concepts
{concept_block}"""


STAGE_C_CRITIQUE_PROMPT = """\
# Task
This is the core concept vocabulary for an ARC-puzzle solving library. It still contains \
redundant entries that describe the SAME underlying idea. Merge them so each idea appears \
once. This is the final cleanup — be decisive.

# What to merge (examples)
- a concept and its "computation"/"detection"/"extraction" variant: \
"bounding box" + "bounding box computation"; "connected components" + "connected \
component extraction"; "color" + "color detection".
- noun/verb or specific/general phrasings of one operation: "recolor" + "color mapping" \
+ "color substitution"; "fill" + "flood fill" when they mean the same thing.
- near-identical structures: "rectangle" + "uniform rectangle" if they refer to the same \
visual entity.
Keep truly distinct ideas apart (e.g. "bounding box" vs "flood fill"). When merging, keep \
the clearest/most general name as the target (usually the higher-frequency one).

# Output
A YAML mapping inside a ```yaml fenced block:
  merges: [[target_name, dup1, dup2, ...], ...]   # dups merge INTO target_name (first item)
  renames: {{old_name: new_name}}                   # optional cleaner names
Aim to remove the clear redundancies; output empty lists only if truly nothing overlaps.

# Vocabulary (name [kind] (freq): gloss)
{vocab_block}"""


def build_stage_c_map_prompt(tag_records: list[dict]) -> str:
    lines = []
    for r in tag_records:
        desc = (r["descriptions"][0] if r["descriptions"] else "").replace("\n", " ")
        lines.append(f"- {r['tag']} [{r['kind']}] (x{r['count']}): {desc}")
    return STAGE_C_MAP_PROMPT.format(tag_block="\n".join(lines))


def build_stage_c_reduce_prompt(groups: list[dict]) -> str:
    lines = []
    for g in groups:
        lines.append(f"- {g['canonical']} [{g.get('kind','routine')}]: {g.get('gloss','')}")
    return STAGE_C_REDUCE_PROMPT.format(concept_block="\n".join(lines))


def build_stage_c_critique_prompt(vocab: list[dict]) -> str:
    lines = []
    for c in vocab:
        lines.append(f"- {c['canonical']} [{c['kind']}] (freq {c['frequency']}): {c.get('gloss','')}")
    return STAGE_C_CRITIQUE_PROMPT.format(vocab_block="\n".join(lines))


def parse_stage_c_groups(completion: str) -> list[dict]:
    data = extract_yaml_block(completion)
    out = []
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical", "")).strip().lower()
        if not canonical:
            continue
        members = item.get("members", []) or []
        members = [str(m).strip().lower() for m in members if str(m).strip()]
        kind = str(item.get("kind", "routine")).strip().lower()
        out.append({
            "canonical": canonical,
            "kind": "structure" if kind.startswith("struct") else "routine",
            "gloss": str(item.get("gloss", "")).strip(),
            "members": members,
        })
    return out


def parse_stage_c_critique(completion: str) -> dict:
    data = extract_yaml_block(completion)
    if not isinstance(data, dict):
        return {"merges": [], "renames": {}}
    merges = []
    for grp in data.get("merges", []) or []:
        if isinstance(grp, list) and len(grp) >= 2:
            merges.append([str(x).strip().lower() for x in grp if str(x).strip()])
    renames = {}
    for k, v in (data.get("renames", {}) or {}).items():
        if str(k).strip() and str(v).strip():
            renames[str(k).strip().lower()] = str(v).strip().lower()
    return {"merges": merges, "renames": renames}


# ===========================================================================
# Stage D — per-concept typed synthesis  (ConceptMemory schema)
# ===========================================================================
STAGE_D_PROMPT = """\
# Introduction
You are writing one entry of a reusable concept library for solving ARC puzzles (hidden \
rules mapping input grids to output grids; 2D integer color arrays, 0 = black background). \
A concept is either a "routine" (an operation / piece of logic) or a "structure" (a class \
of visual entities to look for in the grids).

Below is a concept that recurred across several solved puzzles, followed by the evidence: \
the per-puzzle descriptions and the pseudocode of solutions that used it. Synthesize a \
single, general, well-typed concept entry that captures the shared idea — not the \
specifics of any one puzzle.

# Concept
name: {name}
kind: {kind}
gloss: {gloss}

# Evidence from solutions that used it
{evidence}

# Output
Output ONE concept as a YAML mapping inside a ```yaml fenced block, with fields:
  concept: {name}
  kind: {kind}
  routine_subtype: (routines only) "grid manipulation" if it directly edits the output \
grid, else "intermediate"; omit for structures
  output_typing: a python-like return type (e.g. grid, list[object], color, bool, int); \
you may define a custom type with "Name := definition"
  parameters: list of {{name, typing, description}} describing how the concept varies \
(omit or [] if none)
  description: 1-2 sentences, general (not puzzle-specific)
  cues: list of short notes — what to look for in a puzzle's grids that signals this \
concept is relevant
  implementation: list of short notes — how to implement it programmatically
Keep it concise and general. Reuse the given name exactly."""


def build_stage_d_prompt(name: str, kind: str, gloss: str, evidence: str) -> str:
    return STAGE_D_PROMPT.format(name=name, kind=kind, gloss=gloss, evidence=evidence)


def parse_stage_d(completion: str, fallback_name: str, fallback_kind: str) -> dict | None:
    data = extract_yaml_block(completion)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None
    name = str(data.get("concept") or data.get("name") or fallback_name).strip().lower()
    kind = str(data.get("kind", fallback_kind)).strip().lower()
    kind = "structure" if kind.startswith("struct") else "routine"
    ann: dict = {"concept": name, "kind": kind}
    if kind == "routine" and data.get("routine_subtype"):
        ann["routine_subtype"] = str(data["routine_subtype"]).strip()
    if data.get("output_typing"):
        ann["output_typing"] = str(data["output_typing"]).strip()
    params = []
    for p in data.get("parameters", []) or []:
        if isinstance(p, dict) and p.get("name"):
            params.append({
                "name": str(p["name"]).strip(),
                "typing": (str(p["typing"]).strip() if p.get("typing") else None),
                "description": (str(p["description"]).strip() if p.get("description") else None),
            })
        elif isinstance(p, str) and p.strip():
            params.append({"name": p.strip()})
    if params:
        ann["parameters"] = params
    if data.get("description"):
        ann["description"] = str(data["description"]).strip()
    def _as_list(x):
        if isinstance(x, str):
            return [x.strip()] if x.strip() else []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        return []
    cues = _as_list(data.get("cues"))
    impl = _as_list(data.get("implementation"))
    if cues:
        ann["cues"] = cues
    if impl:
        ann["implementation"] = impl
    return ann


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
@dataclass
class StageUsage:
    """Per-stage token rollup (tokens only — decision D0)."""
    stage: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    completions: int = 0

    @classmethod
    def from_snapshot(cls, stage: str, before: dict, after: dict, model: str) -> "StageUsage":
        b = (before or {}).get(model, {})
        a = (after or {}).get(model, {})
        def d(k: str) -> int:
            return int(a.get(k, 0)) - int(b.get(k, 0))
        return cls(stage, d("input_tokens"), d("output_tokens"),
                   d("reasoning_tokens"), d("requests"), d("completions"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "requests": self.requests,
            "completions": self.completions,
        }
