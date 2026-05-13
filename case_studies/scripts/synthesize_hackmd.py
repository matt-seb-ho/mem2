from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import CASE_STUDIES_ROOT, RUNS_ROOT, load_json, slugify
from case_studies.scripts.modes._shared.trace_loader import resolve_run_dir

SYNTHESIS_ROOT = CASE_STUDIES_ROOT / "synthesis"


def _read_meta(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "meta.json", default={}) or {}


def _relative_link(target: Path, source_file: Path) -> str:
    return Path(os.path.relpath(target.resolve(), source_file.parent.resolve())).as_posix()


def _resolve_run_dirs(run_values: list[str | Path], glob_pattern: str | None = None) -> list[Path]:
    run_dirs: list[Path] = []
    for value in run_values:
        run_dirs.append(resolve_run_dir(value))
    if glob_pattern:
        run_dirs.extend(sorted(path.resolve() for path in RUNS_ROOT.glob(glob_pattern) if path.is_dir()))

    unique: list[Path] = []
    seen: set[Path] = set()
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("At least one run ID, run path, or glob match is required")
    return unique


def _analysis_files(run_dir: Path, modes: list[str] | None) -> list[Path]:
    analyses_dir = run_dir / "analyses"
    if modes:
        return [analyses_dir / f"{mode}.md" for mode in modes if (analyses_dir / f"{mode}.md").exists()]
    if not analyses_dir.exists():
        return []
    return sorted(analyses_dir.glob("*.md"))


def default_output_path(title: str, *, now: datetime | None = None) -> Path:
    when = now or datetime.now(UTC)
    return SYNTHESIS_ROOT / f"{when.strftime('%Y-%m-%d')}_{slugify(title)}.md"


def render_synthesis(
    run_dirs: list[Path],
    *,
    title: str,
    intro: str,
    modes: list[str] | None,
    out_path: Path,
    now: datetime | None = None,
) -> str:
    when = now or datetime.now(UTC)
    lines = [
        f"# {title}",
        "",
        intro.strip() or "TODO: Add synthesis introduction.",
        "",
        f"- Generated UTC: {when.isoformat(timespec='seconds')}",
        f"- Runs included: {len(run_dirs)}",
        f"- Mode filter: {', '.join(modes) if modes else 'all available analyses'}",
        "",
        "## Runs",
        "| run_id | port | label | seed | summary |",
        "|---|---|---|---|---|",
    ]
    for run_dir in run_dirs:
        meta = _read_meta(run_dir)
        run_id = str(meta.get("run_id") or run_dir.name)
        summary_path = run_dir / "summary.md"
        summary_link = _relative_link(summary_path, out_path) if summary_path.exists() else ""
        summary_cell = f"[summary]({summary_link})" if summary_link else "missing"
        lines.append(
            "| {run_id} | {port} | {label} | {seed} | {summary} |".format(
                run_id=run_id,
                port=meta.get("port", "unknown"),
                label=meta.get("label", "unlabeled"),
                seed=meta.get("seed", "unknown"),
                summary=summary_cell,
            )
        )

    lines.extend(["", "## Included Analyses"])
    for run_dir in run_dirs:
        meta = _read_meta(run_dir)
        run_id = str(meta.get("run_id") or run_dir.name)
        lines.extend(["", f"### {run_id}", ""])
        files = _analysis_files(run_dir, modes)
        if not files:
            lines.append("- No matching analyses found.")
            continue
        for path in files:
            lines.append(f"- [{path.stem}]({_relative_link(path, out_path)})")

    lines.extend(
        [
            "",
            "## Cross-Run Notes",
            "",
            "- TODO: Compare error classes, retrieval differences, load-bearing concepts, and phase-shift behavior.",
            "- TODO: Promote paper-grade cases into a curated advisor-facing note.",
            "",
        ]
    )
    return "\n".join(lines)


def synthesize_hackmd(
    run_values: list[str | Path],
    *,
    glob_pattern: str | None = None,
    modes: list[str] | None = None,
    title: str = "Case Study Synthesis",
    intro: str = "",
    out_path: Path | None = None,
    now: datetime | None = None,
) -> Path:
    run_dirs = _resolve_run_dirs(run_values, glob_pattern)
    out = out_path or default_output_path(title, now=now)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_synthesis(run_dirs, title=title, intro=intro, modes=modes, out_path=out, now=now),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a HackMD-compatible cross-run case-study synthesis draft")
    parser.add_argument("runs", nargs="*", help="Run IDs or paths")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="Glob under case_studies/runs/")
    parser.add_argument("--mode", dest="modes", action="append", default=None, help="Analysis mode to include, repeatable")
    parser.add_argument("--title", default="Case Study Synthesis")
    parser.add_argument("--intro", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = synthesize_hackmd(
        args.runs,
        glob_pattern=args.glob_pattern,
        modes=args.modes,
        title=args.title,
        intro=args.intro,
        out_path=args.out,
    )
    print(path)


if __name__ == "__main__":
    main()
