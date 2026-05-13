from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in [_REPO_ROOT, _REPO_ROOT / "src"]:
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from case_studies.scripts._common import REPO_ROOT, RUNS_ROOT, load_json, load_yaml, relative_link, write_json
from case_studies.scripts.render_markdown import write_summary
from case_studies.scripts.run_case_study import build_case_study_config
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components


@dataclass(frozen=True)
class AxisCondition:
    axis: str
    label: str
    condition: dict[str, Any]

    @property
    def is_baseline(self) -> bool:
        return bool(self.condition.get("baseline", False))


def load_axis_conditions(axis_dir: Path | None = None) -> list[AxisCondition]:
    root = axis_dir or (REPO_ROOT / "configs" / "axes")
    conditions: list[AxisCondition] = []
    for axis_path in sorted(root.glob("*.yaml")):
        if axis_path.name.startswith("_"):
            continue
        data = load_yaml(axis_path)
        axis = str(data.get("axis") or axis_path.stem)
        for condition in data.get("conditions", []) or []:
            label = condition.get("label")
            if label:
                conditions.append(AxisCondition(axis=axis, label=str(label), condition=copy.deepcopy(condition)))
    return conditions


def select_conditions(conditions: list[AxisCondition], ports: list[str] | None) -> list[AxisCondition]:
    if not ports:
        return conditions
    wanted = set(ports)
    selected = [condition for condition in conditions if condition.label in wanted]
    missing = sorted(wanted - {condition.label for condition in selected})
    if missing:
        raise ValueError(f"Unknown ports requested: {', '.join(missing)}")
    return selected


def configure_smoke_run(
    cfg: dict[str, Any],
    *,
    max_workers: int,
    model: str,
    max_tokens: int,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    provider_cfg = cfg.setdefault("components", {}).setdefault("provider", {})
    provider_cfg["default_max_concurrency"] = max_workers
    if dotenv_path is not None:
        provider_cfg["dotenv_path"] = str(dotenv_path)

    meta_provider = cfg.setdefault("components", {}).setdefault("meta_edit_provider", {})
    meta_provider["model"] = model
    meta_provider["gen_cfg"] = {"temperature": 0.3, "max_tokens": max_tokens}

    inference_cfg = cfg.setdefault("components", {}).setdefault("inference_engine", {})
    inference_cfg["model"] = model
    gen_cfg = inference_cfg.setdefault("gen_cfg", {})
    gen_cfg["batch_size"] = max_workers
    gen_cfg["max_tokens"] = max_tokens
    gen_cfg.setdefault("temperature", 0.3)
    gen_cfg.setdefault("top_p", 1.0)
    gen_cfg.setdefault("n", 1)
    return cfg


def _iter_trace_dirs(run_dir: Path) -> list[Path]:
    problems_dir = run_dir / "problems"
    if not problems_dir.exists():
        return []
    return sorted(problems_dir.glob("*/iter_*"))


def _text_preview(text: str, *, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _sample_prompt(run_dir: Path) -> str:
    for trace_dir in _iter_trace_dirs(run_dir):
        path = trace_dir / "prompt.txt"
        if path.exists():
            return _text_preview(path.read_text(encoding="utf-8"))
    return ""


def _retrieval_summaries(run_dir: Path) -> tuple[int, list[str], str]:
    hit_count = 0
    metadata_parts: list[str] = []
    sample = ""
    for trace_dir in _iter_trace_dirs(run_dir):
        bundle = load_json(trace_dir / "retrieval_bundle.json", default={}) or {}
        metadata = bundle.get("metadata") or {}
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata_parts.append(f"{key}={value}")
        items = bundle.get("retrieved_items") or bundle.get("items") or bundle.get("concepts") or []
        if isinstance(items, list):
            hit_count += len(items)
        hint = str(bundle.get("hint_text") or "")
        if hint:
            hit_count += 1
        if not sample:
            if isinstance(items, list) and items:
                sample = _text_preview(json.dumps(items[0], sort_keys=True, default=str), limit=260)
            elif hint:
                sample = _text_preview(hint, limit=260)
            elif bundle:
                sample = _text_preview(json.dumps(bundle, sort_keys=True, default=str), limit=260)
    deduped = []
    seen = set()
    for part in metadata_parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)
    return hit_count, deduped[:8], sample


def _llm_call_count(run_dir: Path) -> int:
    return sum(1 for _ in run_dir.glob("problems/*/iter_*/llm_calls/call_*/call_meta.json"))


def _cost_from_meta(run_dir: Path) -> float | None:
    meta = load_json(run_dir / "meta.json", default={}) or {}
    value = meta.get("total_cost_usd")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def engagement_verdict(condition: AxisCondition, *, retrieval_hits: int, metadata: list[str], prompt: str) -> str:
    if condition.is_baseline:
        return "N/A baseline"
    metadata_text = " ".join(metadata).lower()
    prompt_text = prompt.lower()
    if retrieval_hits <= 0:
        return "NO - empty retrieval"
    if "fallback_flat" in metadata_text or "fallback" in metadata_text:
        return "NO - fallback metadata"
    if "disabled_problem_data" in metadata_text or "hybrid_skip_initial" in metadata_text:
        return "NO - retrieval disabled"
    if condition.label.lower() in metadata_text or condition.label.lower() in prompt_text:
        return "YES - label visible"
    return "CHECK - retrieved content present"


def summarize_run(condition: AxisCondition, run_dir: Path, *, success: bool, error: str | None = None, wall_time_s: float | None = None) -> dict[str, Any]:
    retrieval_hits, metadata, sample_retrieval = _retrieval_summaries(run_dir)
    prompt = _sample_prompt(run_dir)
    return {
        "axis": condition.axis,
        "condition": condition.label,
        "baseline": condition.is_baseline,
        "success": success,
        "error": error,
        "run_id": run_dir.name,
        "trace_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.md"),
        "llm_calls": _llm_call_count(run_dir),
        "retrieval_hits": retrieval_hits,
        "retrieval_metadata": metadata,
        "sample_retrieval": sample_retrieval,
        "sample_prompt_snippet": prompt,
        "engagement_verdict": engagement_verdict(condition, retrieval_hits=retrieval_hits, metadata=metadata, prompt=prompt),
        "cost_usd": _cost_from_meta(run_dir),
        "wall_time_s": wall_time_s,
    }


def run_one_condition(args: argparse.Namespace) -> dict[str, Any]:
    condition_map = {condition.label: condition for condition in load_axis_conditions()}
    condition = condition_map[args.port]
    start = time.monotonic()
    cfg, trace_dir = build_case_study_config(
        port=args.port,
        n_problems=args.n_problems,
        seed=args.seed,
        iters=args.iters,
        base_config=args.base_config,
        label=args.label,
    )
    cfg = configure_smoke_run(
        cfg,
        max_workers=args.max_workers,
        model=args.model,
        max_tokens=args.max_tokens,
        dotenv_path=args.dotenv_path,
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        components = resolve_components(cfg)
        run_sync(cfg, components)
        write_summary(trace_dir)
        result = summarize_run(condition, trace_dir, success=True, wall_time_s=time.monotonic() - start)
    except Exception as exc:
        result = summarize_run(condition, trace_dir, success=False, error=str(exc), wall_time_s=time.monotonic() - start)
    write_json(trace_dir / "condition_result.json", result)
    print("SWEEP_RESULT_JSON=" + json.dumps(result, sort_keys=False))
    return result


def _child_command(condition: AxisCondition, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        "--port",
        condition.label,
        "--n-problems",
        str(args.n_problems),
        "--seed",
        str(args.seed),
        "--iters",
        str(args.iters),
        "--base-config",
        str(args.base_config),
        "--label",
        args.label,
        "--max-workers",
        str(args.max_workers),
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.dotenv_path is not None:
        command.extend(["--dotenv-path", str(args.dotenv_path)])
    return command


def _run_condition_subprocess(condition: AxisCondition, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        _child_command(condition, args),
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=args.condition_timeout_s,
    )
    if proc.returncode != 0:
        return {
            "axis": condition.axis,
            "condition": condition.label,
            "baseline": condition.is_baseline,
            "success": False,
            "error": _text_preview((proc.stderr or proc.stdout or "").strip(), limit=500),
            "run_id": "",
            "trace_dir": "",
            "summary_path": "",
            "llm_calls": 0,
            "retrieval_hits": 0,
            "retrieval_metadata": [],
            "sample_retrieval": "",
            "sample_prompt_snippet": "",
            "engagement_verdict": "NO - process failed",
            "cost_usd": None,
            "wall_time_s": time.monotonic() - started,
        }
    result_line = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("SWEEP_RESULT_JSON="):
            result_line = line.removeprefix("SWEEP_RESULT_JSON=")
            break
    try:
        result = json.loads(result_line)
    except Exception:
        result = {
            "axis": condition.axis,
            "condition": condition.label,
            "baseline": condition.is_baseline,
            "success": False,
            "error": "child completed but result JSON was not parseable",
            "run_id": "",
            "trace_dir": "",
            "summary_path": "",
            "llm_calls": 0,
            "retrieval_hits": 0,
            "retrieval_metadata": [],
            "sample_retrieval": _text_preview(proc.stdout, limit=500),
            "sample_prompt_snippet": "",
            "engagement_verdict": "NO - result parse failed",
            "cost_usd": None,
            "wall_time_s": time.monotonic() - started,
        }
    return result


def render_aggregate(results: list[dict[str, Any]], *, started_at: datetime, wall_time_s: float) -> str:
    success_count = sum(1 for row in results if row.get("success"))
    engaged_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("YES"))
    total_calls = sum(int(row.get("llm_calls") or 0) for row in results)
    known_costs = [float(row["cost_usd"]) for row in results if row.get("cost_usd") is not None]
    total_cost = sum(known_costs) if known_costs else None

    lines = [
        "# Smoke Sweep Validation - 2026-05-13",
        "",
        "## Configuration",
        "- 3 problems per condition x 1 seed x iters=1",
        "- Tracer enabled",
        "- Model: deepseek/deepseek-v4-flash via OpenRouter",
        f"- Conditions attempted: {len(results)}",
        f"- Conditions succeeded: {success_count}/{len(results)}",
        f"- Adapted memory engaged verdicts: {engaged_count}/{len(results)} YES",
        f"- Started UTC: {started_at.isoformat(timespec='seconds')}",
        "",
        "## Per-condition engagement verdict",
        "",
        "| Axis | Condition | Adapted memory engaged? | Retrieval metadata | Sample prompt snippet | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(results, key=lambda item: (str(item.get("axis")), str(item.get("condition")))):
        metadata = "; ".join(row.get("retrieval_metadata") or []) or "none"
        notes = []
        if not row.get("success"):
            notes.append(f"ERROR: {row.get('error') or 'unknown'}")
        if row.get("sample_retrieval"):
            notes.append(f"sample retrieval: {row['sample_retrieval']}")
        note_text = " ".join(notes) or "OK"
        lines.append(
            "| {axis} | {condition} | {verdict} | {metadata} | {prompt} | {notes} |".format(
                axis=row.get("axis", ""),
                condition=row.get("condition", ""),
                verdict=str(row.get("engagement_verdict", "")).replace("|", "/"),
                metadata=_text_preview(metadata, limit=180).replace("|", "/"),
                prompt=_text_preview(str(row.get("sample_prompt_snippet") or ""), limit=180).replace("|", "/"),
                notes=_text_preview(note_text, limit=220).replace("|", "/"),
            )
        )

    failures = [row for row in results if not row.get("success")]
    fallback = [row for row in results if "fallback" in str(row.get("engagement_verdict", "")).lower()]
    empty = [row for row in results if "empty retrieval" in str(row.get("engagement_verdict", "")).lower()]
    lines.extend(["", "## Critical findings", ""])
    if not failures and not fallback and not empty:
        lines.append("- No process failures, fallback metadata, or empty retrieval verdicts detected by the generic checker.")
    for row in failures:
        lines.append(f"- FAIL {row.get('condition')}: {row.get('error')}")
    for row in fallback:
        lines.append(f"- FALLBACK {row.get('condition')}: {row.get('retrieval_metadata')}")
    for row in empty:
        lines.append(f"- EMPTY {row.get('condition')}: no retrieval items or hint text found.")

    cost_text = f"${total_cost:.4f}" if total_cost is not None else "unknown"
    lines.extend(
        [
            "",
            "## Cost",
            f"- Total LLM calls: {total_calls}",
            f"- Total spend: {cost_text}",
            f"- Wall time: {wall_time_s / 60:.2f} minutes",
            "",
            "## Per-run trace links",
            "",
        ]
    )
    for row in sorted(results, key=lambda item: (str(item.get("axis")), str(item.get("condition")))):
        if row.get("trace_dir"):
            run_dir = Path(str(row["trace_dir"]))
            try:
                link = relative_link(run_dir, REPO_ROOT)
            except ValueError:
                link = Path(os.path.relpath(run_dir, REPO_ROOT)).as_posix()
            lines.append(f"- Axis {row.get('axis')} `{row.get('condition')}`: [{run_dir.name}]({link}/)")
        else:
            lines.append(f"- Axis {row.get('axis')} `{row.get('condition')}`: no trace directory")

    lines.append("")
    return "\n".join(lines)


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions = select_conditions(load_axis_conditions(), args.ports)
    started_at = datetime.now(UTC)
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_conditions) as pool:
        future_to_condition = {
            pool.submit(_run_condition_subprocess, condition, args): condition
            for condition in conditions
        }
        for future in concurrent.futures.as_completed(future_to_condition):
            condition = future_to_condition[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "axis": condition.axis,
                    "condition": condition.label,
                    "baseline": condition.is_baseline,
                    "success": False,
                    "error": str(exc),
                    "run_id": "",
                    "trace_dir": "",
                    "summary_path": "",
                    "llm_calls": 0,
                    "retrieval_hits": 0,
                    "retrieval_metadata": [],
                    "sample_retrieval": "",
                    "sample_prompt_snippet": "",
                    "engagement_verdict": "NO - harness failed",
                    "cost_usd": None,
                    "wall_time_s": None,
                }
            results.append(result)
            print(f"[{len(results)}/{len(conditions)}] {condition.label}: {result.get('engagement_verdict')} success={result.get('success')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_aggregate(results, started_at=started_at, wall_time_s=time.monotonic() - start), encoding="utf-8")
    write_json(args.out.with_suffix(".json"), {"results": results})
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trace-enabled smoke sweep across all axis conditions")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", help="Internal single-condition port label")
    parser.add_argument("--ports", nargs="*", default=None, help="Optional subset of condition labels")
    parser.add_argument("--n-problems", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--base-config", type=Path, default=Path("configs/experiments/phase1_arc_base.yaml"))
    parser.add_argument("--label", default="smoke-2026-05-13")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--parallel-conditions", type=int, default=4)
    parser.add_argument("--condition-timeout-s", type=int, default=900)
    parser.add_argument("--dotenv-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("case_studies/synthesis/2026-05-13_smoke_sweep_validation.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_one:
        if not args.port:
            raise ValueError("--run-one requires --port")
        run_one_condition(args)
        return
    results = run_sweep(args)
    success_count = sum(1 for row in results if row.get("success"))
    engaged_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("YES"))
    print(f"Wrote {args.out} with {success_count}/{len(results)} successful runs and {engaged_count} YES engagement verdicts.")


if __name__ == "__main__":
    main()
