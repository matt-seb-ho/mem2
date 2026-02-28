"""MATH_SELECT_PROMPT_TEMPLATE — concept selection prompt for competition math problems."""

MATH_SELECT_PROMPT_TEMPLATE = """\
# Introduction
You are helping solve competition math problems by selecting relevant mathematical \
concepts from a library. The selected concepts will be shown as hints during \
problem-solving, so **select only concepts that would genuinely help** — irrelevant \
hints waste context and can mislead.

# Concept Library
Below are concepts (techniques, theorems, strategies) recorded from previously \
solved competition math problems. Each concept has:
- **cues**: problem features that suggest this concept is relevant
- **implementation**: how this concept was applied in past solutions
- **parameters**: ways the concept may vary across problems

{concepts}

# Instructions
1. Read the problem carefully and identify its mathematical domain and core challenge
2. Scan the concept library for concepts whose **cues** match the problem's structure
3. For each candidate, ask: "Would knowing this technique actually help solve this problem?"
4. Select **1-5 concepts** — only those that are directly actionable for this problem
5. Do NOT select concepts just because they share a topic area — they must suggest a \
specific approach or technique that applies

Think step by step before writing your final selection. Then write your selection as a \
yaml list of exact concept names:
```yaml
- Concept Name 1
- Concept Name 2
```

Write your answer inside a markdown yaml code block (```yaml ... ```). \
Use exact concept names from the library above.

# Problem
{puzzle}"""
