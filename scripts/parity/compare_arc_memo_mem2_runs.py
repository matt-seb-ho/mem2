from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mem2.io.json_io import write_json


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_arc_iteration_dirs(arc_run_dir: Path) -> list[Path]:
    dirs = []
    for child in arc_run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            idx = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        dirs.append((idx, child))
    return [d for _, d in sorted(dirs, key=lambda x: x[0])]


def _load_arc_prompts(arc_run_dir: Path) -> list[str]:
    prompts: list[str] = []
    for it_dir in _iter_arc_iteration_dirs(arc_run_dir):
        p = it_dir / "prompts.json"
        if not p.exists():
            continue
        vals = json.loads(p.read_text(encoding="utf-8"))
        prompts.extend(str(v) for v in vals)
    return prompts


def _load_arc_models(arc_run_dir: Path) -> list[str]:
    it_dirs = _iter_arc_iteration_dirs(arc_run_dir)
    if not it_dirs:
        return []
    token_usage_path = it_dirs[-1] / "token_usage.json"
    if not token_usage_path.exists():
        return []
    data = json.loads(token_usage_path.read_text(encoding="utf-8"))
    after = data.get("after", {}) if isinstance(data, dict) else {}
    if not isinstance(after, dict):
        return []
    return sorted(str(k) for k in after.keys())


def _load_mem2_prompts(mem2_run_dir: Path) -> list[str]:
    attempts_path = mem2_run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        return []
    rows = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda r: int((r.get("metadata") or {}).get("global_step_idx", 0)))
    return [str(r.get("prompt", "")) for r in rows]


def _load_mem2_models(mem2_run_dir: Path) -> list[str]:
    usage_path = mem2_run_dir / "provider_usage.json"
    if not usage_path.exists():
        return []
    data = json.loads(usage_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    return sorted(str(k) for k in data.keys())


def _header_sequence(prompt: str) -> list[str]:
    return [line.strip() for line in prompt.splitlines() if line.startswith("### ")]


def _retry_static_suffix(prompt: str) -> str:
    marker = "### New Instructions"
    idx = prompt.find(marker)
    return prompt[idx:].strip() if idx >= 0 else ""


def _has_reselected_lessons(prompt: str) -> bool:
    return "### Reselected Lessons" in prompt


def compare_runs(arc_run_dir: Path, mem2_run_dir: Path) -> dict[str, Any]:
    arc_prompts = _load_arc_prompts(arc_run_dir)
    mem_prompts = _load_mem2_prompts(mem2_run_dir)

    max_len = max(len(arc_prompts), len(mem_prompts))
    step_rows: list[dict[str, Any]] = []
    for i in range(max_len):
        arc_p = arc_prompts[i] if i < len(arc_prompts) else ""
        mem_p = mem_prompts[i] if i < len(mem_prompts) else ""
        step_rows.append(
            {
                "step_idx": i,
                "arc_exists": i < len(arc_prompts),
                "mem2_exists": i < len(mem_prompts),
                "prompt_equal": arc_p == mem_p if arc_p and mem_p else False,
                "arc_prompt_sha256": _sha256(arc_p) if arc_p else None,
                "mem2_prompt_sha256": _sha256(mem_p) if mem_p else None,
                "header_sequence_equal": _header_sequence(arc_p) == _header_sequence(mem_p)
                if arc_p and mem_p
                else False,
                "retry_static_suffix_equal": _retry_static_suffix(arc_p)
                == _retry_static_suffix(mem_p)
                if arc_p and mem_p
                else False,
                "arc_has_reselected_lessons": _has_reselected_lessons(arc_p),
                "mem2_has_reselected_lessons": _has_reselected_lessons(mem_p),
            }
        )

    initial_equal = (
        arc_prompts[0] == mem_prompts[0]
        if arc_prompts and mem_prompts
        else False
    )

    return {
        "arc_run_dir": str(arc_run_dir),
        "mem2_run_dir": str(mem2_run_dir),
        "arc_prompt_count": len(arc_prompts),
        "mem2_prompt_count": len(mem_prompts),
        "initial_prompt_equal": initial_equal,
        "arc_models_seen": _load_arc_models(arc_run_dir),
        "mem2_models_seen": _load_mem2_models(mem2_run_dir),
        "step_comparison": step_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prompt-level parity between arc_memo and mem2 completed runs."
    )
    parser.add_argument("--arc-run-dir", required=True, help="Path to arc_memo run output dir")
    parser.add_argument("--mem2-run-dir", required=True, help="Path to mem2 run output dir")
    parser.add_argument(
        "--out",
        default="outputs/parity/cross_repo_prompt_parity_report.json",
        help="Where to write the comparison JSON report",
    )
    args = parser.parse_args()

    arc_run_dir = Path(args.arc_run_dir).expanduser().resolve()
    mem2_run_dir = Path(args.mem2_run_dir).expanduser().resolve()
    report = compare_runs(arc_run_dir=arc_run_dir, mem2_run_dir=mem2_run_dir)
    write_json(args.out, report)
    print(f"report -> {args.out}")
    print("initial_prompt_equal:", report["initial_prompt_equal"])


if __name__ == "__main__":
    main()

