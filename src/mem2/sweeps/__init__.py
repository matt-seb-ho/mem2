"""mem2.sweeps — shared utilities for the ablation sweep driver + aggregator.

The sweep tools (`scripts/sweeps/ablation_matrix.py`,
`scripts/sweeps/ablation_matrix_parallel.py`, `scripts/sweeps/aggregate_axis.py`)
are thin CLIs around the data structures in this package. All per-axis data
(condition catalogs, kill thresholds, baseline labels, spec-only flags)
lives in `configs/axes/*.yaml`; this package loads and validates them.
"""
