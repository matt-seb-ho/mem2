from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import CASE_STUDIES_ROOT, RUNS_ROOT, load_json
from case_studies.scripts.render_diff import resolve_run_dir


def _port_from_meta(run_dir: Path) -> str:
    meta = load_json(run_dir / "meta.json", default={}) or {}
    port = meta.get("port")
    if not port:
        raise ValueError(f"Missing port in {run_dir / 'meta.json'}")
    return str(port)


def _runs_lines(runs_dir: Path) -> list[str]:
    entries = sorted(path.name for path in runs_dir.iterdir() if path.name != ".gitkeep")
    if not entries:
        return ["No linked runs yet."]
    return [f"- [{name}](runs/{name}/)" for name in entries]


def update_method_readme(method_dir: Path) -> Path:
    readme = method_dir / "README.md"
    runs_dir = method_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        prefix = text.split("## Runs", 1)[0].rstrip()
    else:
        prefix = f"# {method_dir.name} Case Studies"
    body = "\n".join(["## Runs", "", *_runs_lines(runs_dir), ""])
    readme.write_text(prefix + "\n\n" + body, encoding="utf-8")
    return readme


def link_run_to_method(
    run: str | Path,
    port: str | None = None,
    *,
    case_root: Path = CASE_STUDIES_ROOT,
) -> Path:
    run_dir = resolve_run_dir(run)
    method = port or _port_from_meta(run_dir)
    method_dir = case_root / "by_method" / method
    runs_dir = method_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    link_path = runs_dir / run_dir.name

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            raise FileExistsError(f"Refusing to replace non-symlink: {link_path}")

    target = os.path.relpath(run_dir.resolve(), start=runs_dir.resolve())
    link_path.symlink_to(target)
    update_method_readme(method_dir)
    return link_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Link a run into by_method/<port>/runs")
    parser.add_argument("run", help="Run ID or path")
    parser.add_argument("--port", default=None)
    args = parser.parse_args()
    path = link_run_to_method(args.run, port=args.port)
    print(path)


if __name__ == "__main__":
    main()
