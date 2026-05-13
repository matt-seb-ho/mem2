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
    retrieval_items,
    write_analysis_report,
)


def render_counterfactual(run_id_or_path: str | Path, *, drop_top_k: int = 1, inject_text: str | None = None) -> str:
    run = load_case_run(run_id_or_path)
    lines = [
        f"# Counterfactual Bundle Plan: {run.run_id}",
        "",
        f"- Port: {run.port}",
        f"- Label: {run.label}",
        f"- Drop top retrieved items: {drop_top_k}",
        f"- Inject concept text: {inject_text or 'none'}",
        "",
        "This report does not regenerate model responses. It records offline bundle edits that a later controlled run can replay.",
        "",
        "## Candidate Edits",
    ]

    if not run.traces:
        lines.append("- No traces found.")
    for trace in run.traces:
        items = retrieval_items(trace.retrieval_bundle, limit=max(drop_top_k, 6))
        removed = items[:drop_top_k] if drop_top_k > 0 else []
        kept = items[drop_top_k:] if drop_top_k > 0 else items
        lines.extend(
            [
                "",
                f"### {trace.key}",
                "",
                f"- Correct: {trace.correct}",
                f"- Retrieval metadata: {metadata_summary(trace.retrieval_bundle)}",
                f"- Eval: `{extraction_preview(trace.eval_data)}`",
                "- Simulated removal:",
            ]
        )
        lines.extend(f"  - {item}" for item in removed) if removed else lines.append("  - none")
        lines.append("- Remaining retrieved item preview:")
        lines.extend(f"  - {item}" for item in kept[:6]) if kept else lines.append("  - none")
        if inject_text:
            lines.extend(["- Simulated injection:", f"  - {inject_text}"])
        lines.append("- Replay status: TODO")

    lines.extend(["", "## Replay Contract", "", "- Re-render the prompt with the edited bundle.", "- Run the same provider settings as the source run.", "- Compare response and parsed output against the original trace.", ""])
    return "\n".join(lines)


def write_counterfactual(run_id_or_path: str | Path, *, drop_top_k: int = 1, inject_text: str | None = None) -> Path:
    run = load_case_run(run_id_or_path)
    report_path = write_analysis_report(
        run.run_dir,
        "counterfactual",
        render_counterfactual(run.run_dir, drop_top_k=drop_top_k, inject_text=inject_text),
    )
    append_summary_link(run.run_dir, "counterfactual", report_path, "Counterfactual bundle plan")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a dry-run counterfactual retrieval-bundle plan")
    parser.add_argument("run", help="Run ID or path")
    parser.add_argument("--drop-top-k", type=int, default=1)
    parser.add_argument("--inject-text", default=None)
    args = parser.parse_args()
    path = write_counterfactual(args.run, drop_top_k=args.drop_top_k, inject_text=args.inject_text)
    print(path)


if __name__ == "__main__":
    main()
