from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts.modes._shared.trace_loader import (
    append_summary_link,
    extraction_preview,
    load_case_run,
    metadata_summary,
    response_preview,
    retrieval_items,
    write_analysis_report,
)


def render_error_analysis(run_id_or_path: str | Path) -> str:
    run = load_case_run(run_id_or_path)
    failed = [trace for trace in run.traces if not trace.correct]

    lines = [
        f"# Error Analysis: {run.run_id}",
        "",
        f"- Port: {run.port}",
        f"- Label: {run.label}",
        f"- Traces: {len(run.traces)}",
        f"- Failed traces: {len(failed)}",
        "",
        "## Failure Table",
        "| task_id | iter | trace | retrieval_metadata | response_preview |",
        "|---|---|---|---|---|",
    ]
    if failed:
        for trace in failed:
            lines.append(
                "| {task} | {iter_name} | [{trace_key}]({link}/) | {metadata} | {response} |".format(
                    task=trace.task_id,
                    iter_name=trace.iter_name.removeprefix("iter_"),
                    trace_key=trace.key,
                    link=trace.trace_link,
                    metadata=metadata_summary(trace.retrieval_bundle).replace("|", "/"),
                    response=response_preview(trace.response).replace("|", "/"),
                )
            )
    else:
        lines.append("| none | none | none | none | none |")

    lines.extend(["", "## Per-Failure Notes"])
    if failed:
        for trace in failed:
            lines.extend(
                [
                    "",
                    f"### {trace.key}",
                    "",
                    f"- Correct: {trace.correct}",
                    f"- Eval: `{extraction_preview(trace.eval_data)}`",
                    f"- Parsed: `{extraction_preview(trace.parsed)}`",
                    f"- Retrieval metadata: {metadata_summary(trace.retrieval_bundle)}",
                    "- Retrieved item previews:",
                ]
            )
            items = retrieval_items(trace.retrieval_bundle)
            if items:
                lines.extend(f"  - {item}" for item in items)
            else:
                lines.append("  - none detected by generic loader")
            lines.extend(
                [
                    "- Failure label: TODO",
                    "- Hypothesis: TODO",
                ]
            )
    else:
        lines.append("No failed traces found. Use provenance or phase-shift modes for correct-case inspection.")

    lines.extend(["", "## Human Follow-Up", "", "- TODO: Assign failure class and decide whether retrieval, prompt format, parsing, or evaluation caused the miss.", ""])
    return "\n".join(lines)


def write_error_analysis(run_id_or_path: str | Path) -> Path:
    run = load_case_run(run_id_or_path)
    report_path = write_analysis_report(run.run_dir, "error_analysis", render_error_analysis(run.run_dir))
    append_summary_link(run.run_dir, "error_analysis", report_path, "Error analysis")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-failure case-study analysis for one run")
    parser.add_argument("run", help="Run ID or path")
    args = parser.parse_args()
    path = write_error_analysis(args.run)
    print(path)


if __name__ == "__main__":
    main()
