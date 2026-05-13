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

from case_studies.scripts._shared.grid_render import (
    render_pair_markdown,
    render_palette_legend,
    validate_grid,
)
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


def _candidate_problem_files(run_dir: Path, meta: dict[str, Any]) -> list[Path]:
    candidates = [run_dir / "problems.json"]
    if meta.get("problems_path"):
        candidates.append(Path(str(meta["problems_path"])))

    output_dir = meta.get("output_dir")
    if output_dir:
        output_path = Path(str(output_dir))
        if not output_path.is_absolute():
            candidates.append(_REPO_ROOT / output_path / "problems.json")
            candidates.append(run_dir / output_path / "problems.json")
        else:
            candidates.append(output_path / "problems.json")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def _normalize_problem_lookup(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict):
        if "uid" in payload and ("train_pairs" in payload or "test_pairs" in payload):
            return {str(payload["uid"]): payload}
        lookup: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                value.setdefault("uid", key)
                lookup[str(key)] = value
        return lookup
    if isinstance(payload, list):
        lookup = {}
        for item in payload:
            if isinstance(item, dict) and item.get("uid"):
                lookup[str(item["uid"])] = item
        return lookup
    return {}


def _load_problem_lookup(run_dir: Path, meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for path in _candidate_problem_files(run_dir, meta):
        data = load_json(path, default=None)
        if data is not None:
            lookup = _normalize_problem_lookup(data)
            if lookup:
                return lookup
    return {}


def _load_problem_for_task(run_dir: Path, task_id: str, lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for path in [
        run_dir / "problems" / task_id / "problem.json",
        run_dir / "problems" / task_id / "problem_spec.json",
    ]:
        data = load_json(path, default=None)
        if isinstance(data, dict):
            return data
    return lookup.get(task_id)


def _pair_has_renderable_grid(pair: Any) -> bool:
    if not isinstance(pair, dict):
        return False
    for key in ("input", "output"):
        if key in pair:
            try:
                validate_grid(pair[key])
            except ValueError:
                return False
    return "input" in pair or "output" in pair


def _cell_px_for_problem(problem: dict[str, Any]) -> int:
    max_dim = 0
    for pair in list(problem.get("train_pairs") or []) + list(problem.get("test_pairs") or []):
        if not isinstance(pair, dict):
            continue
        for key in ("input", "output"):
            if key not in pair:
                continue
            try:
                grid = validate_grid(pair[key])
            except ValueError:
                continue
            max_dim = max(max_dim, len(grid), len(grid[0]))
    if max_dim >= 25:
        return 5
    if max_dim >= 16:
        return 8
    if max_dim >= 10:
        return 12
    return 18


def _render_grid_section(run_dir: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    lookup = _load_problem_lookup(run_dir, meta)
    rendered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in seen:
            continue
        seen.add(task_id)
        problem = _load_problem_for_task(run_dir, task_id, lookup)
        if not problem:
            continue
        train_pairs = [pair for pair in (problem.get("train_pairs") or []) if _pair_has_renderable_grid(pair)]
        test_pairs = [pair for pair in (problem.get("test_pairs") or []) if _pair_has_renderable_grid(pair)]
        if not train_pairs and not test_pairs:
            continue
        cell_px = _cell_px_for_problem(problem)
        rendered.extend(["", f"### {task_id}", ""])
        for idx, pair in enumerate(train_pairs[:3], start=1):
            rendered.append(render_pair_markdown(pair, label=f"train {idx}", cell_px=cell_px))
            rendered.append("")
        for idx, pair in enumerate(test_pairs[:2], start=1):
            rendered.append(render_pair_markdown(pair, label=f"test {idx}", cell_px=cell_px))
            rendered.append("")
        if len(train_pairs) > 3 or len(test_pairs) > 2:
            rendered.append("- Additional pairs omitted from summary view.")
            rendered.append("")

    if not rendered:
        return []
    return [
        "",
        "## ARC grids",
        "",
        "Palette:",
        "",
        render_palette_legend(),
        "",
        "Rendered grids are included when the case-study run can resolve ARC `problems.json` from `meta.json.output_dir` or per-problem trace files.",
        "",
        *rendered,
    ]


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

    lines.extend(_render_grid_section(run_dir, rows, meta))

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
