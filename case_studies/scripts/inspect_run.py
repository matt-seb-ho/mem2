from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import iter_trace_dirs, load_json, relative_link


def inspect_run(run_dir: Path) -> str:
    run_dir = run_dir.resolve()
    meta = load_json(run_dir / "meta.json", default={}) or {}
    lines = [
        f"Run: {meta.get('run_id', run_dir.name)}",
        f"Port: {meta.get('port', 'unknown')}",
        f"Label: {meta.get('label', 'unknown')}",
        f"Problems: {meta.get('n_problems', 'unknown')}",
        f"LLM calls: {meta.get('llm_call_count', 'unknown')}",
        "",
        "Traces:",
    ]
    trace_dirs = iter_trace_dirs(run_dir)
    if not trace_dirs:
        lines.append("- none")
    for iter_dir in trace_dirs:
        eval_data = load_json(iter_dir / "eval.json", default={}) or {}
        call_meta = load_json(iter_dir / "call_meta.json", default={}) or {}
        correct = "true" if eval_data.get("correct", False) else "false"
        latency = call_meta.get("latency_s", "unknown")
        lines.append(
            f"- {relative_link(iter_dir, run_dir)} correct={correct} latency_s={latency}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a case-study run")
    parser.add_argument("run_dir", type=Path, help="Path to case_studies/runs/<run_id>")
    args = parser.parse_args()
    print(inspect_run(args.run_dir))


if __name__ == "__main__":
    main()
