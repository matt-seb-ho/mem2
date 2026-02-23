"""CODE_SELECT_PROMPT_TEMPLATE — concept selection prompt for competitive programming problems."""

CODE_SELECT_PROMPT_TEMPLATE = """\
# Introduction
You are helping solve competitive programming problems by writing Python code. Your task is to examine a set of algorithmic concepts recorded from previously solved problems and determine which concepts are relevant to a given problem. Your selected concepts will be used to guide the problem-solving phase, so emphasize problem-solving helpfulness.


# Concepts from Previously Solved Problems
We recorded concepts (algorithms, data structures, patterns, optimizations, etc.) from previously solved competitive programming problems. These concepts may or may not be relevant to this problem. Concepts are annotated with fields like:
- cues: what patterns or problem features suggest this concept is relevant
- implementation: how this concept was applied in past solutions (as Python code patterns)
- parameters: ways the concept may vary across problems
Recommendations:
- First identify the algorithmic domain and core challenge of the problem (e.g. graph traversal, dynamic programming, greedy, string processing, etc.)
- Then look for concepts whose cues match the problem's structure or constraints
- Consider whether the concept's implementation pattern fits the problem
- There may not be exact matches, so think about variations and novel combinations
- These concepts are only suggestions, use them as you see fit

{concepts}

# Instructions
Identify which concepts could be relevant to the given problem.
- Consider the problem's algorithmic domain (graphs, DP, greedy, math, string manipulation, etc.)
- Look for concepts whose cues match the problem's structure, constraints, or input/output patterns
- Think about which algorithms, data structures, or optimization techniques could help solve the problem
- Write your final selection of concepts as a yaml formatted list of concept names
- To allow us to match your selection to the concepts we have, please use the exact concept names as they appear in the above concept list
- Write your answer inside a markdown yaml code block (i.e. be sure to have "```yaml" in the line before your code and "```" in the line after your list)
- Here is a formatting example:
```yaml
- Binary Search on Answer
- Segment Tree Range Query
...
```

# Problem
{puzzle}"""
