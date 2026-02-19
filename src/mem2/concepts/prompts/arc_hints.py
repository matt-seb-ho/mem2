"""HINT_TEMPLATE_OP3 — ported from arc_memo/concept_mem/evaluation/prompts.py."""

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
