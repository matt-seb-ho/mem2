from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts.modes._shared.trace_loader import append_summary_link, load_case_run, write_analysis_report


def render_mechanistic_attribution(run_id_or_path: str | Path) -> str:
    """Render a placeholder for future retrieval-to-response attribution.

    Success means this mode can align retrieved concept spans, rendered prompt
    spans, response spans, and final prediction changes without claiming causal
    influence from co-occurrence alone.
    """
    run = load_case_run(run_id_or_path)
    return "\n".join(
        [
            f"# Mechanistic Attribution Plan: {run.run_id}",
            "",
            f"- Port: {run.port}",
            f"- Traces: {len(run.traces)}",
            "",
            "## Status",
            "",
            "TODO: implement token-span and prompt-span attribution once trace format stores stable prompt section offsets.",
            "",
            "## Intended Checks",
            "",
            "- Map retrieved concept IDs to prompt sections.",
            "- Map answer claims or code spans back to prompt sections.",
            "- Distinguish evidence present from evidence used.",
            "",
        ]
    )


def write_mechanistic_attribution(run_id_or_path: str | Path) -> Path:
    run = load_case_run(run_id_or_path)
    report_path = write_analysis_report(run.run_dir, "mechanistic_attribution", render_mechanistic_attribution(run.run_dir))
    append_summary_link(run.run_dir, "mechanistic_attribution", report_path, "Mechanistic attribution plan")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write mechanistic attribution scaffold")
    parser.add_argument("run", help="Run ID or path")
    args = parser.parse_args()
    path = write_mechanistic_attribution(args.run)
    print(path)


if __name__ == "__main__":
    main()
