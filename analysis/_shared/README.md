# Shared Analysis Utilities

This folder contains importable helpers that analysis modules can share. Utilities must be deterministic, file-local, and free of provider calls. They should accept explicit paths instead of discovering global state silently.
