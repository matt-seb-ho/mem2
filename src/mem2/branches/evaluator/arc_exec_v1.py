"""Backward-compatible shim; prefer non-versioned module path."""

from .arc_exec import ArcExecutionEvaluator

__all__ = ["ArcExecutionEvaluator"]
