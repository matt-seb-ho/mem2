# ARC-AGI-1

**Source:** [ARC-AGI-1](https://github.com/fchollet/ARC-AGI) (Chollet, 2019)

Each puzzle is a JSON file encoding a transformation rule over colored pixel grids.
The objective is to infer the rule from input-output example pairs and produce the output for held-out test inputs.

## Contents

| Directory | Count | Description |
|-----------|-------|-------------|
| `training/` | 400 | Public training puzzles (easier difficulty) |
| `evaluation/` | 400 | Public validation puzzles (matches private eval difficulty) |
| `barc_seeds/` | 164 | Handwritten Python solutions from [BARC](https://github.com/xu3kev/BARC) (Li et al., 2024). Each `.py` imports `common.py` and implements `def main(input_grid)`. Used to seed concept memory. |
| `concept_memory/` | — | Pre-built concept memory artifacts from ArcMemo (Ho et al., 2025) |

## Concept Memory

| File | Description |
|------|-------------|
| `compressed_v1.json` | 270 PS-format concepts abstracted from BARC seed solutions. Main seed memory used in the paper. Fields: name, kind, routine_subtype, output_typing, parameters, description, cues, implementation, used_in. |
| `cue_impl_compressed.json` | 93 concept entries with compressed cue + implementation fields. |
| `parsed_lessons.json` | 160 lessons extracted from solution traces (keyed by puzzle UID). |

## Splits (`splits.json`)

| Split | Count | Description |
|-------|-------|-------------|
| `eval_100` | 100 | Uniform random subset of evaluation set, same 100 puzzles used in the ArcMemo paper. |
| `barc_seeds` | 164 | Training puzzles with BARC Python solutions (for memory initialization). |

## Puzzle JSON Format

```json
{
  "train": [
    {"input": [[0,0,1],[1,0,0]], "output": [[1,1,0],[0,1,1]]}
  ],
  "test": [
    {"input": [[0,1,0],[0,0,1]], "output": [[1,0,1],[1,1,0]]}
  ]
}
```

Grid values are integers 0-9 representing colors.
