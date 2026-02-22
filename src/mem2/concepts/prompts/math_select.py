"""MATH_SELECT_PROMPT_TEMPLATE — concept selection prompt for competition math problems."""

MATH_SELECT_PROMPT_TEMPLATE = """\
# Introduction
You are helping solve competition math problems (Number Theory, Counting & Probability, etc.) by writing Python code. Your task is to examine a set of mathematical concepts recorded from previously solved problems and determine which concepts are relevant to a given problem. Your selected concepts will be used to guide the problem-solving phase, so emphasize problem-solving helpfulness.


# Concepts from Previously Solved Problems
We recorded concepts (techniques, theorems, algorithms, formulas, etc.) from previously solved competition math problems. These concepts may or may not be relevant to this problem. Concepts are annotated with fields like:
- cues: what patterns or problem features suggest this concept is relevant
- implementation: how this concept was applied in past solutions (as Python code patterns)
- parameters: ways the concept may vary across problems
Recommendations:
- First identify the mathematical domain and core challenge of the problem
- Then look for concepts whose cues match the problem's structure
- Consider whether the concept's implementation pattern fits the problem
- There may not be exact matches, so think about variations and novel combinations
- These concepts are only suggestions, use them as you see fit

{concepts}

# Instructions
Identify which concepts could be relevant to the given problem.
- Consider the problem's mathematical domain (number theory, combinatorics, modular arithmetic, etc.)
- Look for concepts whose cues match the problem's structure or constraints
- Think about which techniques or theorems could help solve the problem
- Write your final selection of concepts as a yaml formatted list of concept names
- To allow us to match your selection to the concepts we have, please use the exact concept names as they appear in the above concept list
- Write your answer inside a markdown yaml code block (i.e. be sure to have "```yaml" in the line before your code and "```" in the line after your list)
- Here is a formatting example:
```yaml
- GCD Pair Parametrization
- Euler's Totient Computation
...
```

# Problem
{puzzle}"""
