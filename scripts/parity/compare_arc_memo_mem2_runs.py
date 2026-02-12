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


def _uid_from_metadata_item(item: object) -> str | None:
    """Extract problem UID from an arc_memo metadata entry.

    Iteration-1 entries are lists like [uid, "0"].
    Iteration-2+ entries are dicts like {"puzzle_id": uid, ...}.
    """
    if isinstance(item, dict):
        uid = item.get("puzzle_id") or item.get("problem_uid")
        if uid is None:
            return None
        return str(uid)
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, str):
        return item
    return None


def _load_arc_prompt_pairs(
    arc_run_dir: Path,
) -> list[dict[str, Any]]:
    """Load (uid, pass_idx, prompt) triples from arc_memo iteration dirs."""
    pairs: list[dict[str, Any]] = []
    for pass_idx, it_dir in enumerate(_iter_arc_iteration_dirs(arc_run_dir)):
        prompts_path = it_dir / "prompts.json"
        metadata_path = it_dir / "metadata.json"
        if not prompts_path.exists():
            continue
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        metadata: list[Any] = []
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for i, prompt in enumerate(prompts):
            uid = _uid_from_metadata_item(metadata[i]) if i < len(metadata) else None
            pairs.append({
                "problem_uid": uid,
                "pass_idx": pass_idx,
                "prompt": str(prompt),
            })
    return pairs


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


def _load_mem2_prompt_pairs(
    mem2_run_dir: Path,
) -> list[dict[str, Any]]:
    """Load (uid, pass_idx, prompt) triples from mem2 attempts.jsonl."""
    attempts_path = mem2_run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        return []
    rows = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda r: int((r.get("metadata") or {}).get("global_step_idx", 0)))
    pairs: list[dict[str, Any]] = []
    for r in rows:
        uid = r.get("problem_uid")
        md = r.get("metadata") or {}
        pass_idx = int(md.get("pass_idx", md.get("global_step_idx", 0)))
        pairs.append({
            "problem_uid": uid,
            "pass_idx": pass_idx,
            "prompt": str(r.get("prompt", "")),
        })
    return pairs


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


def _compare_prompt_pair(
    *,
    problem_uid: str | None,
    arc_prompt: str | None,
    mem2_prompt: str | None,
) -> dict[str, Any]:
    arc_p = arc_prompt or ""
    mem_p = mem2_prompt or ""
    both_exist = bool(arc_prompt) and bool(mem2_prompt)
    return {
        "problem_uid": problem_uid,
        "arc_exists": arc_prompt is not None,
        "mem2_exists": mem2_prompt is not None,
        "prompt_equal": (arc_p == mem_p) if both_exist else False,
        "arc_prompt_sha256": _sha256(arc_p) if arc_p else None,
        "mem2_prompt_sha256": _sha256(mem_p) if mem_p else None,
        "header_sequence_equal": (
            _header_sequence(arc_p) == _header_sequence(mem_p) if both_exist else False
        ),
        "retry_static_suffix_equal": (
            _retry_static_suffix(arc_p) == _retry_static_suffix(mem_p) if both_exist else False
        ),
        "arc_has_reselected_lessons": _has_reselected_lessons(arc_p),
        "mem2_has_reselected_lessons": _has_reselected_lessons(mem_p),
    }


def compare_runs(arc_run_dir: Path, mem2_run_dir: Path) -> dict[str, Any]:
    arc_pairs = _load_arc_prompt_pairs(arc_run_dir)
    mem2_pairs = _load_mem2_prompt_pairs(mem2_run_dir)

    # --- UID-based comparison (primary) ---
    # Group by (problem_uid, occurrence_index) to handle multiple passes per UID.
    arc_by_uid: dict[str, list[str]] = {}
    for p in arc_pairs:
        uid = p["problem_uid"] or "__unknown__"
        arc_by_uid.setdefault(uid, []).append(p["prompt"])

    mem2_by_uid: dict[str, list[str]] = {}
    for p in mem2_pairs:
        uid = p["problem_uid"] or "__unknown__"
        mem2_by_uid.setdefault(uid, []).append(p["prompt"])

    all_uids = sorted(set(arc_by_uid.keys()) | set(mem2_by_uid.keys()))

    uid_comparison: list[dict[str, Any]] = []
    uid_match_count = 0
    uid_total = 0
    arc_only_uids: list[str] = []
    mem2_only_uids: list[str] = []

    for uid in all_uids:
        arc_list = arc_by_uid.get(uid, [])
        mem2_list = mem2_by_uid.get(uid, [])
        if not mem2_list:
            arc_only_uids.append(uid)
        if not arc_list:
            mem2_only_uids.append(uid)
        max_len = max(len(arc_list), len(mem2_list))
        for i in range(max_len):
            arc_p = arc_list[i] if i < len(arc_list) else None
            mem2_p = mem2_list[i] if i < len(mem2_list) else None
            row = _compare_prompt_pair(
                problem_uid=uid,
                arc_prompt=arc_p,
                mem2_prompt=mem2_p,
            )
            row["occurrence_idx"] = i
            uid_comparison.append(row)
            uid_total += 1
            if row["prompt_equal"]:
                uid_match_count += 1

    # --- Positional comparison (legacy, for backward compat) ---
    arc_prompts = [p["prompt"] for p in arc_pairs]
    mem_prompts = [p["prompt"] for p in mem2_pairs]

    initial_equal = (
        arc_prompts[0] == mem_prompts[0] if arc_prompts and mem_prompts else False
    )

    return {
        "arc_run_dir": str(arc_run_dir),
        "mem2_run_dir": str(mem2_run_dir),
        "arc_prompt_count": len(arc_prompts),
        "mem2_prompt_count": len(mem_prompts),
        "initial_prompt_equal": initial_equal,
        "arc_models_seen": _load_arc_models(arc_run_dir),
        "mem2_models_seen": _load_mem2_models(mem2_run_dir),
        # UID-based parity summary
        "uid_match_count": uid_match_count,
        "uid_total_compared": uid_total,
        "uid_match_rate": uid_match_count / uid_total if uid_total > 0 else 0.0,
        "arc_only_uids": arc_only_uids,
        "mem2_only_uids": mem2_only_uids,
        "uid_comparison": uid_comparison,
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
    print(f"initial_prompt_equal: {report['initial_prompt_equal']}")
    print(f"uid_match_rate: {report['uid_match_count']}/{report['uid_total_compared']}")
    if report["arc_only_uids"]:
        print(f"arc_only_uids ({len(report['arc_only_uids'])}): {report['arc_only_uids'][:5]}...")
    if report["mem2_only_uids"]:
        print(f"mem2_only_uids ({len(report['mem2_only_uids'])}): {report['mem2_only_uids'][:5]}...")


if __name__ == "__main__":
    main()
