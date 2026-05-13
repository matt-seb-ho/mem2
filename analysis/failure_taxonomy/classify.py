from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis._shared.load_runs import load_run


def classify_run(run_dir: Path, out_path: Path) -> Path:
    run = load_run(run_dir)
    payload = {
        "module": "failure_taxonomy",
        "status": "pending_no_llm_classifier",
        "run_id": run["run_id"],
        "trace_count": len(run["traces"]),
        "classes": [
            "single-cell discriminative",
            "relative-offset",
            "small-mask",
            "color-role",
            "region-boundary",
        ],
        "records": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a no-LLM failure taxonomy placeholder")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("failure_taxonomy_placeholder.json"))
    args = parser.parse_args()
    print(classify_run(args.run_dir, args.out))


if __name__ == "__main__":
    main()
