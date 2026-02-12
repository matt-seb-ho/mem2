"""Backward-compatible shim; prefer non-versioned module path."""

from .arcmemo_ps import ArcMemoPsMemoryBuilder

__all__ = ["ArcMemoPsMemoryBuilder"]
