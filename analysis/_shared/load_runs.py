from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def iter_case_traces(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for iter_dir in sorted((root / "problems").glob("*/iter_*")):
        rows.append(
            {
                "run_id": root.name,
                "task_id": iter_dir.parent.name,
                "iter": iter_dir.name.removeprefix("iter_"),
                "trace_dir": str(iter_dir),
                "call_meta": load_json(iter_dir / "call_meta.json", default={}) or {},
                "retrieval_bundle": load_json(iter_dir / "retrieval_bundle.json", default={}) or {},
                "eval": load_json(iter_dir / "eval.json", default={}) or {},
                "parsed": load_json(iter_dir / "parsed.json", default={}) or {},
            }
        )
    return rows


def load_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    return {
        "run_dir": str(root),
        "run_id": root.name,
        "meta": load_json(root / "meta.json", default={}) or {},
        "traces": iter_case_traces(root),
    }


def traces_dataframe(run_dir: str | Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for traces_dataframe()") from exc
    return pd.DataFrame(iter_case_traces(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a case-study run and print a JSON summary")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = load_run(args.run_dir)
    print(json.dumps({"run_id": run["run_id"], "trace_count": len(run["traces"])}, indent=2))


if __name__ == "__main__":
    main()
