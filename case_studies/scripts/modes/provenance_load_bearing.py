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
    load_case_run,
    metadata_summary,
    retrieval_items,
    write_analysis_report,
)


def render_provenance_load_bearing(run_id_or_path: str | Path) -> str:
    """Render a provenance scaffold for correct traces.

    Success means this mode can name the retrieved concept that was necessary
    for a correct answer, then connect it to the task or episode that first
    introduced that concept.
    """
    run = load_case_run(run_id_or_path)
    correct = [trace for trace in run.traces if trace.correct]
    lines = [
        f"# Provenance Load-Bearing Review: {run.run_id}",
        "",
        f"- Port: {run.port}",
        f"- Correct traces: {len(correct)}",
        "",
        "## Correct Trace Candidates",
    ]
    if not correct:
        lines.append("- No correct traces found.")
    for trace in correct:
        lines.extend(["", f"### {trace.key}", "", f"- Retrieval metadata: {metadata_summary(trace.retrieval_bundle)}", "- Retrieved item candidates:"])
        items = retrieval_items(trace.retrieval_bundle)
        lines.extend(f"  - {item}" for item in items) if items else lines.append("  - none detected by generic loader")
        lines.extend(["- Load-bearing concept: TODO", "- Provenance source: TODO"])
    lines.append("")
    return "\n".join(lines)


def write_provenance_load_bearing(run_id_or_path: str | Path) -> Path:
    run = load_case_run(run_id_or_path)
    report_path = write_analysis_report(run.run_dir, "provenance_load_bearing", render_provenance_load_bearing(run.run_dir))
    append_summary_link(run.run_dir, "provenance_load_bearing", report_path, "Provenance load-bearing review")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write provenance load-bearing scaffold")
    parser.add_argument("run", help="Run ID or path")
    args = parser.parse_args()
    path = write_provenance_load_bearing(args.run)
    print(path)


if __name__ == "__main__":
    main()
