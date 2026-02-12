from __future__ import annotations

from pathlib import Path


def run_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "memory": root / "memory",
        "analysis": root / "analysis",
    }
