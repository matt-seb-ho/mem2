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


def render_adversarial(run_id_or_path: str | Path) -> str:
    """Render a placeholder for future ARC perturbation studies.

    Success means this mode can generate controlled grid perturbations,
    re-render prompts, replay the same provider settings, and report which
    perturbations changed correctness or retrieval behavior.
    """
    run = load_case_run(run_id_or_path)
    return "\n".join(
        [
            f"# Adversarial Perturbation Plan: {run.run_id}",
            "",
            f"- Port: {run.port}",
            f"- Traces: {len(run.traces)}",
            "",
            "## Status",
            "",
            "TODO: implement controlled ARC input-grid perturbations after raw problem grids are saved in every case-study trace.",
            "",
            "## Intended Checks",
            "",
            "- Preserve task semantics when perturbing distractor cells.",
            "- Track whether retrieval changes after the perturbation.",
            "- Track whether parsed output and correctness change.",
            "",
        ]
    )


def write_adversarial(run_id_or_path: str | Path) -> Path:
    run = load_case_run(run_id_or_path)
    report_path = write_analysis_report(run.run_dir, "adversarial", render_adversarial(run.run_dir))
    append_summary_link(run.run_dir, "adversarial", report_path, "Adversarial perturbation plan")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write adversarial perturbation scaffold")
    parser.add_argument("run", help="Run ID or path")
    args = parser.parse_args()
    path = write_adversarial(args.run)
    print(path)


if __name__ == "__main__":
    main()
