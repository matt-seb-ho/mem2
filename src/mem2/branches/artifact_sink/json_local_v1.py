"""Backward-compatible shim; prefer non-versioned module path."""

from .json_local import JsonLocalArtifactSink

__all__ = ["JsonLocalArtifactSink"]
