"""MATH_HINT_TEMPLATE — hint framing for competition math concept memory."""

MATH_HINT_TEMPLATE = """\
### Concepts from Previously Solved Problems
We recorded concepts (techniques, theorems, algorithms, formulas, etc.) from previously solved competition math problems. These concepts may or may not be relevant to this problem. Concepts are annotated with:
- cues: what patterns or problem features suggest this concept is relevant
- implementation: how this concept was applied in past solutions (as Python code patterns)
- parameters: ways the concept may vary across problems
Recommendations:
- First identify the mathematical domain and core challenge of the problem
- Consider whether the concept's implementation pattern fits this problem
- There may not be exact matches, so think about variations and novel combinations
- These concepts are only suggestions, use them as you see fit

{hints}"""
