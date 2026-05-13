from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts.modes._shared.trace_loader import (
    CaseRun,
    append_summary_link,
    extraction_preview,
    load_case_run,
    metadata_summary,
    response_preview,
    write_analysis_report,
)


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


def render_comparative(run_values: list[str | Path]) -> str:
    runs = [load_case_run(value) for value in run_values]
    if len(runs) < 2:
        raise ValueError("Comparative mode requires at least two runs")

    maps = [run.trace_map() for run in runs]
    keys = sorted(set().union(*(trace_map.keys() for trace_map in maps)))

    lines = [
        "# Comparative Case Study",
        "",
        "## Runs",
        "| run | port | label | traces |",
        "|---|---|---|---:|",
    ]
    for run in runs:
        lines.append(f"| {run.run_id} | {run.port} | {run.label} | {len(run.traces)} |")

    lines.extend(["", "## Outcome Matrix", "| task_iter | " + " | ".join(run.run_id for run in runs) + " |", "|---" + "|---" * len(runs) + "|"])
    for key in keys:
        cells = []
        for trace_map in maps:
            trace = trace_map.get(key)
            if trace is None:
                cells.append("missing")
            else:
                cells.append("correct" if trace.correct else "wrong")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Per-Task Deltas"])
    baseline: CaseRun = runs[0]
    baseline_map = maps[0]
    for key in keys:
        if key not in baseline_map:
            continue
        base_trace = baseline_map[key]
        lines.extend(["", f"### {key}", ""])
        for run, trace_map in zip(runs[1:], maps[1:], strict=True):
            trace = trace_map.get(key)
            lines.append(f"#### {baseline.run_id} vs {run.run_id}")
            if trace is None:
                lines.append("- Comparison trace missing.")
                continue
            lines.extend(
                [
                    f"- Correctness: {base_trace.correct} vs {trace.correct}",
                    f"- Retrieval metadata: `{metadata_summary(base_trace.retrieval_bundle)}` vs `{metadata_summary(trace.retrieval_bundle)}`",
                    f"- Eval delta anchor: `{extraction_preview(base_trace.eval_data)}` vs `{extraction_preview(trace.eval_data)}`",
                    f"- Response preview baseline: {response_preview(base_trace.response)}",
                    f"- Response preview comparison: {response_preview(trace.response)}",
                    "",
                    "Prompt diff:",
                    "```diff",
                    _unified_diff(base_trace.prompt, trace.prompt, f"{baseline.run_id}/{key}/prompt", f"{run.run_id}/{key}/prompt"),
                    "```",
                ]
            )

    lines.append("")
    return "\n".join(lines)


def write_comparative(run_values: list[str | Path], out_run: str | Path | None = None) -> Path:
    runs = [load_case_run(value) for value in run_values]
    if len(runs) < 2:
        raise ValueError("Comparative mode requires at least two runs")
    target_run = load_case_run(out_run) if out_run is not None else runs[0]
    report_path = write_analysis_report(target_run.run_dir, "comparative", render_comparative([run.run_dir for run in runs]))
    append_summary_link(target_run.run_dir, "comparative", report_path, "Comparative analysis")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write side-by-side case-study analysis for two or more runs")
    parser.add_argument("runs", nargs="+", help="Run IDs or paths")
    parser.add_argument("--out-run", default=None, help="Run ID or path that receives analyses/comparative.md")
    args = parser.parse_args()
    path = write_comparative(args.runs, args.out_run)
    print(path)


if __name__ == "__main__":
    main()
