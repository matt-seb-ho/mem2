"""SELECT_PROMPT_TEMPLATE — ported from arc_memo/concept_mem/memory/v4/select.py."""

SELECT_PROMPT_TEMPLATE = """\
# Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and solving the puzzle means predicting the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

Your task is to analyze a puzzle's reference examples, examine a set of concepts recorded from previously solved puzzles, and determine which concepts are relevant to this puzzle. Your selected concepts and notes will be used for the puzzle solving phase, so emphasize problem solving helpfulness.


# Concepts from Previously Solved Puzzles
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

{concepts}

# Instructions
Identify which concepts could be relevant to the given puzzle.
- We suggest first investigating more "visible" concepts first (e.g. structures and grid manipulation routines)
- After identifying these concepts, you can investigate what logic/criteria/intermediate routines might be useful for these initially identified concepts
- You can also select any other concept you think might be relevant even if it's not directly related to the grid manipulation routines
- Write your final selection of concepts as a yaml formatted list of concept names
- To allow us to match your selection to the concepts we have, please use the exact concept names as they appear in the above concept list
- Write your answer inside a markdown yaml code block (i.e. be sure to have "```yaml" in the line before your code and "```" in the line after your list)
- Here is a formatting example:
```yaml
- line drawing
- intersection of lines
...
```

# Your Given Puzzle
Analyze the following puzzle:
{puzzle}"""
