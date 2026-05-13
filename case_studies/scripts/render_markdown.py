from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import iter_trace_dirs, load_json, relative_link


def _metadata_summary(bundle: dict[str, Any]) -> str:
    metadata = bundle.get("metadata") or {}
    parts = []
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            parts.append(f"{key}={value}")
        if len(parts) >= 5:
            break
    return ", ".join(parts) if parts else "none"


def _result_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for iter_dir in iter_trace_dirs(run_dir):
        task_id = iter_dir.parent.name
        iter_id = iter_dir.name.removeprefix("iter_")
        call_meta = load_json(iter_dir / "call_meta.json", default={}) or {}
        eval_data = load_json(iter_dir / "eval.json", default={}) or {}
        retrieval = load_json(iter_dir / "retrieval_bundle.json", default={}) or {}
        rows.append(
            {
                "task_id": task_id,
                "iter": iter_id,
                "correct": bool(eval_data.get("correct", False)),
                "latency_s": call_meta.get("latency_s"),
                "retrieval_metadata": _metadata_summary(retrieval),
                "trace_path": relative_link(iter_dir, run_dir),
            }
        )
    return rows


def render_summary(run_dir: Path) -> str:
    run_dir = run_dir.resolve()
    meta = load_json(run_dir / "meta.json", default={}) or {}
    port = meta.get("port") or "unknown"
    label = meta.get("label") or "unlabeled"
    timestamp = meta.get("timestamp_utc") or "unknown-date"
    rows = _result_rows(run_dir)

    lines = [
        f"# Case Study: {port} - {label} - {timestamp}",
        "",
        "## Config",
        f"- Port: {port}",
        f"- N problems: {meta.get('n_problems', 'unknown')}",
        f"- Seed: {meta.get('seed', 'unknown')}",
        f"- Model: {meta.get('model', 'unknown')}",
        f"- Total LLM calls: {meta.get('llm_call_count', 'unknown')}",
        f"- Total cost USD: ${meta.get('total_cost_usd', 'unknown')}",
        "",
        "## Results",
        "| task_id | iter | correct | latency_s | retrieval_metadata |",
        "|---|---:|---|---:|---|",
    ]
    for row in rows:
        latency = row["latency_s"]
        latency_text = f"{latency:.3f}" if isinstance(latency, (float, int)) else "unknown"
        lines.append(
            "| {task_id} | {iter} | {correct} | {latency} | {metadata} |".format(
                task_id=row["task_id"],
                iter=row["iter"],
                correct="true" if row["correct"] else "false",
                latency=latency_text,
                metadata=row["retrieval_metadata"].replace("|", "/"),
            )
        )

    lines.extend(["", "## Per-problem traces"])
    if rows:
        for row in rows:
            lines.append(
                f"- [{row['task_id']}/iter_{row['iter']}]({row['trace_path']}/) - "
                "prompt | response | bundle | eval"
            )
    else:
        lines.append("- No problem traces found.")

    lines.extend(
        [
            "",
            "## Notable observations",
            "TODO: Fill after human inspection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(run_dir: Path) -> Path:
    out_path = run_dir / "summary.md"
    out_path.write_text(render_summary(run_dir), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a case-study summary.md")
    parser.add_argument("run_dir", type=Path, help="Path to case_studies/runs/<run_id>")
    args = parser.parse_args()
    path = write_summary(args.run_dir)
    print(path)


if __name__ == "__main__":
    main()
