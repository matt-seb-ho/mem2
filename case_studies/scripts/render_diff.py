from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import RUNS_ROOT, iter_trace_dirs, load_json


def resolve_run_dir(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    candidate = RUNS_ROOT / str(value)
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Run not found: {value}")


def _trace_map(run_dir: Path) -> dict[str, Path]:
    return {
        f"{trace.parent.name}/{trace.name}": trace
        for trace in iter_trace_dirs(run_dir)
    }


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _json_text(path: Path) -> str:
    data = load_json(path, default={})
    return json.dumps(data, indent=2, sort_keys=True)


def _unified_diff(left: str, right: str, left_name: str, right_name: str) -> str:
    lines = list(
        difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=left_name,
            tofile=right_name,
            lineterm="",
        )
    )
    return "\n".join(lines) if lines else "No differences."


def render_diff(left_run: Path, right_run: Path) -> str:
    left_run = left_run.resolve()
    right_run = right_run.resolve()
    left_traces = _trace_map(left_run)
    right_traces = _trace_map(right_run)
    keys = sorted(set(left_traces) | set(right_traces))

    lines = [
        f"# Case Study Diff: {left_run.name} vs {right_run.name}",
        "",
        f"- Left: `{left_run}`",
        f"- Right: `{right_run}`",
        "",
        "## Trace Coverage",
    ]
    for key in keys:
        status = "both"
        if key not in left_traces:
            status = "right-only"
        elif key not in right_traces:
            status = "left-only"
        lines.append(f"- {key}: {status}")

    for key in keys:
        if key not in left_traces or key not in right_traces:
            continue
        left_trace = left_traces[key]
        right_trace = right_traces[key]
        lines.extend(["", f"## {key}", "", "### Retrieval Bundle", "```diff"])
        lines.append(
            _unified_diff(
                _json_text(left_trace / "retrieval_bundle.json"),
                _json_text(right_trace / "retrieval_bundle.json"),
                f"{left_run.name}/{key}/retrieval_bundle.json",
                f"{right_run.name}/{key}/retrieval_bundle.json",
            )
        )
        lines.extend(["```", "", "### Prompt", "```diff"])
        lines.append(
            _unified_diff(
                _read_text(left_trace / "prompt.txt"),
                _read_text(right_trace / "prompt.txt"),
                f"{left_run.name}/{key}/prompt.txt",
                f"{right_run.name}/{key}/prompt.txt",
            )
        )
        lines.extend(["```", "", "### Response", "```diff"])
        lines.append(
            _unified_diff(
                _read_text(left_trace / "response.txt"),
                _read_text(right_trace / "response.txt"),
                f"{left_run.name}/{key}/response.txt",
                f"{right_run.name}/{key}/response.txt",
            )
        )
        lines.append("```")

    lines.append("")
    return "\n".join(lines)


def write_diff(left_run: Path, right_run: Path, out_path: Path | None = None) -> Path:
    left_run = resolve_run_dir(left_run)
    right_run = resolve_run_dir(right_run)
    out_path = out_path or (left_run / "retrieval_diff.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_diff(left_run, right_run), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a diff between two case-study runs")
    parser.add_argument("left_run", help="Run ID or path for the left run")
    parser.add_argument("right_run", help="Run ID or path for the right run")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = write_diff(args.left_run, args.right_run, args.out)
    print(path)


if __name__ == "__main__":
    main()
