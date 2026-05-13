from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts.modes._shared.trace_loader import append_summary_link, load_case_run, metadata_summary, write_analysis_report


def render_phase_shift_envelope(run_values: list[str | Path]) -> str:
    """Render a repeated-run stability scaffold.

    Success means this mode can quantify correctness variance, retrieval
    variance, and response-shape variance across repeated runs for the same
    problem set and provider settings.
    """
    runs = [load_case_run(value) for value in run_values]
    if len(runs) < 2:
        raise ValueError("Phase-shift envelope requires at least two runs")

    grouped: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
    for run in runs:
        for trace in run.traces:
            grouped[trace.key].append((run.run_id, trace.correct, metadata_summary(trace.retrieval_bundle)))

    lines = [
        "# Phase-Shift Envelope",
        "",
        "## Runs",
        "| run | port | label | traces |",
        "|---|---|---|---:|",
    ]
    for run in runs:
        lines.append(f"| {run.run_id} | {run.port} | {run.label} | {len(run.traces)} |")

    lines.extend(["", "## Stability Matrix", "| task_iter | observed_runs | correct_count | wrong_count | retrieval_metadata |", "|---|---:|---:|---:|---|"])
    for key, observations in sorted(grouped.items()):
        correct_count = sum(1 for _, correct, _ in observations if correct)
        metadata = "; ".join(f"{run_id}: {summary}" for run_id, _, summary in observations)
        lines.append(f"| {key} | {len(observations)} | {correct_count} | {len(observations) - correct_count} | {metadata.replace('|', '/')} |")

    lines.extend(["", "## TODO", "", "- Add prompt-level and response-level similarity once stable normalization is chosen.", ""])
    return "\n".join(lines)


def write_phase_shift_envelope(run_values: list[str | Path], out_run: str | Path | None = None) -> Path:
    runs = [load_case_run(value) for value in run_values]
    if len(runs) < 2:
        raise ValueError("Phase-shift envelope requires at least two runs")
    target_run = load_case_run(out_run) if out_run is not None else runs[0]
    report_path = write_analysis_report(target_run.run_dir, "phase_shift_envelope", render_phase_shift_envelope([run.run_dir for run in runs]))
    append_summary_link(target_run.run_dir, "phase_shift_envelope", report_path, "Phase-shift envelope")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write repeated-run phase-shift scaffold")
    parser.add_argument("runs", nargs="+", help="Run IDs or paths")
    parser.add_argument("--out-run", default=None, help="Run ID or path that receives analyses/phase_shift_envelope.md")
    args = parser.parse_args()
    path = write_phase_shift_envelope(args.runs, args.out_run)
    print(path)


if __name__ == "__main__":
    main()
