from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis._shared.load_runs import load_run


def extract_growth(run_dir: Path, out_path: Path) -> Path:
    run = load_run(run_dir)
    payload = {
        "module": "memory_growth",
        "status": "placeholder",
        "run_id": run["run_id"],
        "trace_count": len(run["traces"]),
        "memory_size_over_time": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a placeholder memory growth extraction")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("memory_growth_placeholder.json"))
    args = parser.parse_args()
    print(extract_growth(args.run_dir, args.out))


if __name__ == "__main__":
    main()
