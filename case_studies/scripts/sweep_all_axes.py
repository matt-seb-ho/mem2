from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import statistics
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

from case_studies.scripts._common import REPO_ROOT, RUNS_ROOT, load_json, load_yaml, relative_link, slugify, write_json
from case_studies.scripts.render_markdown import write_summary
from case_studies.scripts.run_case_study import build_case_study_config
from mem2.core.entities import RunContext, to_primitive
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components
from mem2.registry.benchmark import BENCHMARKS


_STAGED_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def select_conditions(
    conditions: list[AxisCondition],
    ports: list[str] | None,
    axes: list[str] | None = None,
) -> list[AxisCondition]:
    if axes:
        wanted_axes = set(axes)
        conditions = [condition for condition in conditions if condition.axis in wanted_axes]
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
    ignore_cache: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    provider_cfg = cfg.setdefault("components", {}).setdefault("provider", {})
    provider_cfg["default_max_concurrency"] = max_workers
    if dotenv_path is not None:
        provider_cfg["dotenv_path"] = str(dotenv_path)

    meta_provider = cfg.setdefault("components", {}).setdefault("meta_edit_provider", {})
    meta_provider["model"] = model
    meta_provider["gen_cfg"] = {
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "ignore_cache": ignore_cache,
    }

    inference_cfg = cfg.setdefault("components", {}).setdefault("inference_engine", {})
    inference_cfg["model"] = model
    gen_cfg = inference_cfg.setdefault("gen_cfg", {})
    gen_cfg["batch_size"] = max_workers
    gen_cfg["max_tokens"] = max_tokens
    gen_cfg.setdefault("temperature", 0.3)
    gen_cfg.setdefault("top_p", 1.0)
    gen_cfg.setdefault("n", 1)
    gen_cfg["ignore_cache"] = ignore_cache
    return cfg


def _staged_artifact_slug(value: str) -> str:
    return _STAGED_SLUG_RE.sub("_", value.strip()).strip("_") or "variant"


def _staged_root(args: argparse.Namespace) -> Path:
    return REPO_ROOT / "case_studies" / "staged_artifacts" / slugify(args.label)


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _condition_cfg_for_stage(
    condition: AxisCondition,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, Any], Path]:
    return build_case_study_config(
        port=condition.label,
        n_problems=args.n_problems,
        seed=seed,
        iters=args.iters,
        base_config=args.base_config,
        label=args.label,
        output_root=getattr(args, "output_root", None),
    )


def _staged_bank_path(args: argparse.Namespace, condition: AxisCondition, seed: int) -> Path:
    return (
        _staged_root(args)
        / "banks"
        / f"bank_reorg_{_staged_artifact_slug(condition.label)}_seed{seed}.json"
    )


def _staged_retrieval_path(args: argparse.Namespace, condition: AxisCondition, seed: int) -> Path:
    return (
        _staged_root(args)
        / "retrieval"
        / f"{_staged_artifact_slug(condition.label)}_seed{seed}.json"
    )


def _stage_problem_path(args: argparse.Namespace, seed: int) -> Path:
    return _staged_root(args) / "problems" / f"problems_n{args.n_problems}_seed{seed}.json"


def _write_stage_problem_file(
    args: argparse.Namespace,
    conditions: list[AxisCondition],
    *,
    seed: int,
) -> Path:
    out_path = _stage_problem_path(args, seed)
    if out_path.exists():
        return out_path
    if not conditions:
        raise ValueError("No conditions selected")
    cfg, _ = _condition_cfg_for_stage(conditions[0], args, seed=seed)
    benchmark_key = str(cfg.get("pipeline", {}).get("benchmark", ""))
    if benchmark_key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark for staged problems: {benchmark_key}")
    benchmark_cfg = cfg.get("components", {}).get("benchmark", {}) or {}
    benchmark = BENCHMARKS[benchmark_key](**benchmark_cfg)
    ctx = RunContext(
        run_id=f"staged_problems_seed{seed}",
        seed=seed,
        config=cfg,
        output_dir=str(out_path.parent),
        tags={"stage": "problems"},
    )
    problems = benchmark.load(ctx)
    write_json(out_path, to_primitive(problems))
    return out_path


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


def _condition_parity_grade(condition: AxisCondition) -> str:
    if condition.is_baseline:
        return "baseline"
    grade = condition.condition.get("parity_grade")
    if grade:
        return str(grade)
    candidate = condition.condition.get("candidate") or {}
    if isinstance(candidate, dict) and candidate.get("parity_grade"):
        return str(candidate["parity_grade"])
    return "unknown"


def _condition_parity_note(condition: AxisCondition) -> str:
    note = condition.condition.get("parity_note")
    if note:
        return str(note)
    candidate = condition.condition.get("candidate") or {}
    if isinstance(candidate, dict) and candidate.get("parity_note"):
        return str(candidate["parity_note"])
    return ""


def _score_from_trace(run_dir: Path) -> dict[str, Any]:
    meta = load_json(run_dir / "meta.json", default={}) or {}
    run_summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    per_problem: list[dict[str, Any]] = []
    for trace_dir in _iter_trace_dirs(run_dir):
        eval_data = load_json(trace_dir / "eval.json", default={}) or {}
        records = eval_data.get("records") if isinstance(eval_data.get("records"), list) else []
        first_record = records[0] if records and isinstance(records[0], dict) else {}
        problem_uid = first_record.get("problem_uid") or trace_dir.parent.name
        correct = bool(eval_data.get("correct", first_record.get("is_correct", False)))
        per_problem.append(
            {
                "problem_uid": str(problem_uid),
                "task_id": trace_dir.parent.name,
                "iter": trace_dir.name.removeprefix("iter_"),
                "correct": correct,
                "attempt_idx": first_record.get("attempt_idx"),
                "metadata": first_record.get("metadata") or {},
            }
        )
    per_problem.sort(key=lambda item: (str(item["problem_uid"]), str(item["iter"])))
    n_total = len(per_problem)
    n_correct = sum(1 for item in per_problem if item["correct"])
    trace_score = n_correct / n_total if n_total else None
    official_total = run_summary.get("total_attempts") or run_summary.get("problem_count") or n_total
    official_correct = run_summary.get("correct_attempts")
    official_score = run_summary.get("accuracy_per_attempt")
    return {
        "score": trace_score if trace_score is not None else official_score,
        "n_correct": n_correct if n_total else official_correct,
        "n_total": n_total or official_total,
        "per_problem": per_problem,
        "official_score": official_score,
        "official_score_sum": run_summary.get("official_score_sum") or run_summary.get("official_score"),
        "strict_score": run_summary.get("strict_score"),
    }


def write_score_summary(condition: AxisCondition, run_dir: Path, *, success: bool, error: str | None = None) -> dict[str, Any]:
    meta = load_json(run_dir / "meta.json", default={}) or {}
    score_data = _score_from_trace(run_dir)
    summary = {
        "run_id": run_dir.name,
        "condition": condition.label,
        "axis": condition.axis,
        "parity_grade": _condition_parity_grade(condition),
        "parity_note": _condition_parity_note(condition),
        "success": success,
        "error": error,
        "seed": meta.get("seed"),
        "n_problems": meta.get("n_problems"),
        "model": meta.get("model"),
        "llm_calls": _llm_call_count(run_dir),
        "cost_usd": _cost_from_meta(run_dir),
        **score_data,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


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
    score_summary = write_score_summary(condition, run_dir, success=success, error=error)
    return {
        "axis": condition.axis,
        "condition": condition.label,
        "baseline": condition.is_baseline,
        "parity_grade": _condition_parity_grade(condition),
        "parity_note": _condition_parity_note(condition),
        "success": success,
        "error": error,
        "run_id": run_dir.name,
        "trace_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.md"),
        "score_summary_path": str(run_dir / "summary.json"),
        "seed": score_summary.get("seed"),
        "score": score_summary.get("score"),
        "n_correct": score_summary.get("n_correct"),
        "n_total": score_summary.get("n_total"),
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
        output_root=getattr(args, "output_root", None),
    )
    cfg = configure_smoke_run(
        cfg,
        max_workers=args.max_workers,
        model=args.model,
        max_tokens=args.max_tokens,
        dotenv_path=args.dotenv_path,
        ignore_cache=args.ignore_cache,
    )
    if getattr(args, "staged_bank", None):
        bank_path = _resolve_repo_path(args.staged_bank)
        cfg.setdefault("components", {}).setdefault("memory_builder", {})[
            "seed_memory_file"
        ] = str(bank_path)
        cfg.setdefault("components", {}).setdefault("memory_builder", {})[
            "freeze_memory"
        ] = True
        cfg.setdefault("case_studies", {}).setdefault("staged", {})[
            "bank"
        ] = str(bank_path)
    if getattr(args, "retrieval_from_file", None):
        retrieval_path = _resolve_repo_path(args.retrieval_from_file)
        cfg["retrieval_from_file"] = str(retrieval_path)
        cfg.setdefault("pipeline", {})["memory_retriever"] = "none"
        cfg.setdefault("components", {})["memory_retriever"] = {}
        cfg.setdefault("case_studies", {}).setdefault("staged", {})[
            "retrieval_from_file"
        ] = str(retrieval_path)
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
        "--cache",
        "false" if args.ignore_cache else "true",
    ]
    if getattr(args, "output_root", None) is not None:
        command.extend(["--output-root", str(args.output_root)])
    if args.dotenv_path is not None:
        command.extend(["--dotenv-path", str(args.dotenv_path)])
    if getattr(args, "staged_bank", None):
        command.extend(["--staged-bank", str(args.staged_bank)])
    if getattr(args, "retrieval_from_file", None):
        command.extend(["--retrieval-from-file", str(args.retrieval_from_file)])
    return command


def _failed_result(condition: AxisCondition, *, error: str, wall_time_s: float, seed: int | None = None) -> dict[str, Any]:
    return {
        "axis": condition.axis,
        "condition": condition.label,
        "baseline": condition.is_baseline,
        "parity_grade": _condition_parity_grade(condition),
        "parity_note": _condition_parity_note(condition),
        "seed": seed,
        "success": False,
        "error": error,
        "run_id": "",
        "trace_dir": "",
        "summary_path": "",
        "score_summary_path": "",
        "score": None,
        "n_correct": None,
        "n_total": None,
        "llm_calls": 0,
        "retrieval_hits": 0,
        "retrieval_metadata": [],
        "sample_retrieval": "",
        "sample_prompt_snippet": "",
        "engagement_verdict": "NO - process failed",
        "cost_usd": None,
        "wall_time_s": wall_time_s,
    }


def _run_condition_subprocess_once(condition: AxisCondition, args: argparse.Namespace) -> dict[str, Any]:
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
        return _failed_result(
            condition,
            error=_text_preview((proc.stderr or proc.stdout or "").strip(), limit=500),
            wall_time_s=time.monotonic() - started,
            seed=args.seed,
        )
    result_line = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("SWEEP_RESULT_JSON="):
            result_line = line.removeprefix("SWEEP_RESULT_JSON=")
            break
    try:
        result = json.loads(result_line)
    except Exception:
        result = _failed_result(
            condition,
            error="child completed but result JSON was not parseable",
            wall_time_s=time.monotonic() - started,
            seed=args.seed,
        )
        result["sample_retrieval"] = _text_preview(proc.stdout, limit=500)
    return result


def _run_condition_subprocess(condition: AxisCondition, args: argparse.Namespace) -> dict[str, Any]:
    attempts = int(getattr(args, "retries", 0) or 0) + 1
    last_result: dict[str, Any] | None = None
    for attempt_idx in range(attempts):
        result = _run_condition_subprocess_once(condition, args)
        result["attempt"] = attempt_idx + 1
        if result.get("success"):
            return result
        last_result = result
    return last_result or _failed_result(condition, error="no subprocess attempt ran", wall_time_s=0.0, seed=args.seed)


def render_aggregate(results: list[dict[str, Any]], *, started_at: datetime, wall_time_s: float) -> str:
    success_count = sum(1 for row in results if row.get("success"))
    explicit_yes_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("YES"))
    check_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("CHECK"))
    baseline_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("N/A"))
    no_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("NO"))
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
        f"- Explicit adapter/source engagement: {explicit_yes_count}/{len(results)} YES",
        f"- Retrieved content present, needs manual builder-level confirmation: {check_count}/{len(results)} CHECK",
        f"- Baseline or intentionally memory-free controls: {baseline_count}/{len(results)} N/A",
        f"- Empty retrieval or failed engagement: {no_count}/{len(results)} NO",
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


def _axis_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (int(text), text)
    except ValueError:
        return (999, text)


def _score_text(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def group_phase_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in results:
        key = str(row.get("condition"))
        entry = grouped.setdefault(
            key,
            {
                "axis": row.get("axis"),
                "condition": row.get("condition"),
                "parity_grade": row.get("parity_grade"),
                "parity_note": row.get("parity_note"),
                "runs": [],
            },
        )
        entry["runs"].append(row)
    groups = []
    for entry in grouped.values():
        successful_scores = [
            float(row["score"])
            for row in entry["runs"]
            if row.get("success") and row.get("score") is not None
        ]
        entry["mean"] = statistics.fmean(successful_scores) if successful_scores else None
        entry["std"] = statistics.stdev(successful_scores) if len(successful_scores) >= 2 else 0.0 if successful_scores else None
        entry["success_count"] = sum(1 for row in entry["runs"] if row.get("success"))
        entry["llm_calls"] = sum(int(row.get("llm_calls") or 0) for row in entry["runs"])
        known_costs = [float(row["cost_usd"]) for row in entry["runs"] if row.get("cost_usd") is not None]
        entry["cost_usd"] = sum(known_costs) if known_costs else None
        entry["n_per_seed"] = max((int(row.get("n_total") or 0) for row in entry["runs"]), default=0)
        groups.append(entry)
    groups.sort(key=lambda item: (_axis_sort_key(item.get("axis")), -(item.get("mean") if item.get("mean") is not None else -1.0), str(item.get("condition"))))
    return groups


def render_phase_g_lite(
    results: list[dict[str, Any]],
    *,
    started_at: datetime,
    wall_time_s: float,
    seeds: list[int],
    n_problems: int,
    iters: int,
    max_workers: int,
    ignore_cache: bool,
    model: str,
) -> str:
    groups = group_phase_results(results)
    total_calls = sum(int(row.get("llm_calls") or 0) for row in results)
    known_costs = [float(row["cost_usd"]) for row in results if row.get("cost_usd") is not None]
    total_cost = sum(known_costs) if known_costs else None
    failures = [row for row in results if not row.get("success")]
    anomalies = [
        row
        for row in results
        if row.get("success")
        and row.get("score") is not None
        and (float(row["score"]) < 0.05 or float(row["score"]) > 0.95)
    ]
    surface_groups = [
        group
        for group in groups
        if "surface" in str(group.get("parity_grade", "")).lower()
    ]

    lines = [
        "# Phase G-Lite Results - 2026-05-13",
        "",
        "## Configuration",
        f"- Conditions: {len(groups)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Problems per seed: {n_problems}",
        f"- Iters: {iters}",
        f"- Cache: {'disabled' if ignore_cache else 'enabled'}",
        f"- Max workers: {max_workers}",
        f"- Model: {model}",
        f"- Tracer: enabled",
        f"- Started UTC: {started_at.isoformat(timespec='seconds')}",
        f"- Wall time: {wall_time_s / 60:.2f} minutes",
        f"- Total LLM calls: {total_calls}",
        f"- Total spend: {f'${total_cost:.4f}' if total_cost is not None else 'unknown'}",
        "",
        "## Per-condition results",
        "",
        "| Axis | Condition | Parity grade | n per seed | " + " | ".join(f"seed {seed}" for seed in seeds) + " | Mean | Std | LLM calls | Cost | Notes |",
        "|---|---|---|---:|" + "---:|" * len(seeds) + "---:|---:|---:|---:|---|",
    ]
    for group in groups:
        runs_by_seed = {int(row["seed"]): row for row in group["runs"] if row.get("seed") is not None}
        score_cells = []
        notes = []
        for seed in seeds:
            run = runs_by_seed.get(seed)
            if run is None:
                score_cells.append("missing")
                notes.append(f"seed {seed} missing")
            elif not run.get("success"):
                score_cells.append("fail")
                notes.append(f"seed {seed} failed")
            else:
                score_cells.append(_score_text(run.get("score")))
        if group.get("success_count") != len(seeds):
            notes.append(f"{group.get('success_count')}/{len(seeds)} seeds succeeded")
        note_text = "; ".join(notes) if notes else "OK"
        cost = group.get("cost_usd")
        lines.append(
            "| {axis} | {condition} | {parity} | {n_per_seed} | {scores} | {mean} | {std} | {calls} | {cost} | {notes} |".format(
                axis=group.get("axis", ""),
                condition=str(group.get("condition", "")).replace("|", "/"),
                parity=str(group.get("parity_grade", "")).replace("|", "/"),
                n_per_seed=group.get("n_per_seed", 0),
                scores=" | ".join(score_cells),
                mean=_score_text(group.get("mean")),
                std=_score_text(group.get("std")),
                calls=group.get("llm_calls", 0),
                cost=f"${cost:.4f}" if isinstance(cost, (int, float)) else "unknown",
                notes=note_text.replace("|", "/"),
            )
        )

    lines.extend(["", "## Findings to inspect", ""])
    if failures:
        for row in failures:
            lines.append(f"- FAIL `{row.get('condition')}` seed {row.get('seed')}: {row.get('error')}")
    else:
        lines.append("- No failed condition-seed runs.")
    if anomalies:
        for row in anomalies:
            lines.append(f"- ANOMALY `{row.get('condition')}` seed {row.get('seed')}: score={_score_text(row.get('score'))}")
    else:
        lines.append("- No per-seed scores below 0.05 or above 0.95.")

    lines.extend(["", "## Surface-tier footnotes", ""])
    if surface_groups:
        for group in surface_groups:
            note = _text_preview(str(group.get("parity_note") or "See adapter README for disclosure."), limit=240)
            lines.append(f"- `{group.get('condition')}`: {note}")
    else:
        lines.append("- No surface-tier rows detected in this aggregate.")

    lines.extend(["", "## Per-run trace links", ""])
    for row in sorted(results, key=lambda item: (_axis_sort_key(item.get("axis")), str(item.get("condition")), int(item.get("seed") or 0))):
        if row.get("trace_dir"):
            run_dir = Path(str(row["trace_dir"]))
            try:
                link = relative_link(run_dir, REPO_ROOT)
            except ValueError:
                link = Path(os.path.relpath(run_dir, REPO_ROOT)).as_posix()
            lines.append(
                f"- Axis {row.get('axis')} `{row.get('condition')}` seed {row.get('seed')}: "
                f"[{run_dir.name}]({link}/), score={_score_text(row.get('score'))}"
            )
        else:
            lines.append(f"- Axis {row.get('axis')} `{row.get('condition')}` seed {row.get('seed')}: no trace directory")

    lines.append("")
    return "\n".join(lines)


def _parse_seeds(seed_arg: str | None, legacy_seed: int) -> list[int]:
    if not seed_arg:
        return [legacy_seed]
    seeds = []
    for part in seed_arg.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [legacy_seed]


def _parse_cache_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid --cache value: {value}")


def _parse_condition_filter(ports: list[str] | None, conditions: str | None) -> list[str] | None:
    values: list[str] = []
    for item in ports or []:
        values.extend(part.strip() for part in item.split(","))
    if conditions:
        values.extend(part.strip() for part in conditions.split(","))
    return [value for value in values if value] or None


def _parse_axes_filter(axes: str | None) -> list[str] | None:
    if not axes:
        return None
    values = [part.strip() for part in axes.split(",") if part.strip()]
    return values or None


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "integrated":
        args.mode = "smoke"
    if args.mode == "phase-g-lite":
        if args.n_problems == 3:
            args.n_problems = 50
        if args.seeds is None:
            args.seeds = "42,43"
        if args.label == "smoke-2026-05-13":
            args.label = "phase-g-lite-2026-05-13"
        default_smoke_out = Path("case_studies/synthesis/2026-05-13_smoke_sweep_validation.md")
        if args.out == default_smoke_out:
            args.out = Path("case_studies/synthesis/2026-05-13_phase_g_lite_results.md")
        if args.parallel_conditions == 4:
            args.parallel_conditions = 5
        if args.retries == 0:
            args.retries = 1
    if args.mode == "staged":
        if args.stage is None:
            raise ValueError("--mode staged requires --stage {2,2.5,3,all}")
        default_smoke_out = Path("case_studies/synthesis/2026-05-13_smoke_sweep_validation.md")
        if args.out == default_smoke_out:
            args.out = Path("case_studies/synthesis/staged_pipeline_results.md")
    args.seed_list = _parse_seeds(args.seeds, args.seed)
    args.cache_enabled = _parse_cache_flag(args.cache)
    args.ignore_cache = not args.cache_enabled
    args.ports = _parse_condition_filter(args.ports, args.conditions)
    args.axis_filter = _parse_axes_filter(args.axes)
    if args.output is not None:
        args.output_root = args.output
        if args.label == "smoke-2026-05-13":
            args.label = args.output.name
        default_smoke_out = Path("case_studies/synthesis/2026-05-13_smoke_sweep_validation.md")
        if args.out == default_smoke_out:
            args.out = args.output / "summary.md"
    else:
        args.output_root = args.output_root
    return args


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions = select_conditions(load_axis_conditions(), args.ports, args.axis_filter)
    seeds = list(getattr(args, "seed_list", [args.seed]))
    started_at = datetime.now(UTC)
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_conditions) as pool:
        future_to_target: dict[concurrent.futures.Future[dict[str, Any]], tuple[AxisCondition, int]] = {}
        for condition in conditions:
            for seed in seeds:
                child_args = copy.copy(args)
                child_args.seed = seed
                future = pool.submit(_run_condition_subprocess, condition, child_args)
                future_to_target[future] = (condition, seed)
        for future in concurrent.futures.as_completed(future_to_target):
            condition, seed = future_to_target[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _failed_result(condition, error=str(exc), wall_time_s=0.0, seed=seed)
                result["engagement_verdict"] = "NO - harness failed"
            results.append(result)
            print(
                f"[{len(results)}/{len(conditions) * len(seeds)}] {condition.label} seed={seed}: "
                f"score={_score_text(result.get('score'))} verdict={result.get('engagement_verdict')} "
                f"success={result.get('success')}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wall_time_s = time.monotonic() - start
    if args.mode == "phase-g-lite":
        args.out.write_text(
            render_phase_g_lite(
                results,
                started_at=started_at,
                wall_time_s=wall_time_s,
                seeds=seeds,
                n_problems=args.n_problems,
                iters=args.iters,
                max_workers=args.max_workers,
                ignore_cache=args.ignore_cache,
                model=args.model,
            ),
            encoding="utf-8",
        )
        write_json(
            args.out.with_suffix(".json"),
            {
                "mode": args.mode,
                "config": {
                    "n_problems": args.n_problems,
                    "seeds": seeds,
                    "iters": args.iters,
                    "cache": args.cache_enabled,
                    "ignore_cache": args.ignore_cache,
                    "max_workers": args.max_workers,
                    "model": args.model,
                    "started_at_utc": started_at.isoformat(timespec="seconds"),
                    "wall_time_s": wall_time_s,
                },
                "by_condition": group_phase_results(results),
                "results": results,
            },
        )
    else:
        args.out.write_text(render_aggregate(results, started_at=started_at, wall_time_s=wall_time_s), encoding="utf-8")
        write_json(args.out.with_suffix(".json"), {"results": results})
    return results


def _run_stage_script(
    command: list[str],
    *,
    result_prefix: str,
    timeout_s: int,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "error": f"stage script timed out after {timeout_s}s: {_text_preview(str(exc), limit=400)}",
            "wall_time_s": time.monotonic() - started,
            "command": command,
        }
    wall_time_s = time.monotonic() - started
    if proc.returncode != 0:
        return {
            "success": False,
            "error": _text_preview((proc.stderr or proc.stdout or "").strip(), limit=700),
            "wall_time_s": wall_time_s,
            "command": command,
        }
    result_line = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(result_prefix):
            result_line = line.removeprefix(result_prefix)
            break
    try:
        result = json.loads(result_line)
    except Exception:
        result = {
            "success": False,
            "error": "stage script completed but result JSON was not parseable",
            "stdout": _text_preview(proc.stdout, limit=700),
        }
    result.setdefault("success", True)
    result["wall_time_s"] = wall_time_s
    result["command"] = command
    return result


def run_staged_stage2(args: argparse.Namespace, conditions: list[AxisCondition], seeds: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    banks_dir = _staged_root(args) / "banks"
    for condition in conditions:
        for seed in seeds:
            cfg, _ = _condition_cfg_for_stage(condition, args, seed=seed)
            builder_cfg = cfg.get("components", {}).get("memory_builder", {}) or {}
            input_bank = _resolve_repo_path(
                builder_cfg.get("seed_memory_file")
                or "data/arc_agi/concept_memory/compressed_v1.json"
            )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "apply_reorg.py"),
                "--input-bank",
                str(input_bank),
                "--variant",
                condition.label,
                "--seed",
                str(seed),
                "--output-dir",
                str(banks_dir),
                "--builder-cfg",
                json.dumps(builder_cfg, sort_keys=False),
                "--provider",
                str(cfg.get("components", {}).get("provider", {}).get("profile_name", "llmplus_openrouter")),
                "--model",
                args.model,
                "--concurrency",
                str(max(1, min(args.max_workers, 8))),
                "--max-tokens",
                str(args.max_tokens),
            ]
            if args.dotenv_path is not None:
                command.extend(["--dotenv-path", str(args.dotenv_path)])
            if args.ignore_cache:
                command.append("--ignore-cache")
            result = _run_stage_script(
                command,
                result_prefix="APPLY_REORG_RESULT_JSON=",
                timeout_s=args.condition_timeout_s,
            )
            result.update(
                {
                    "stage": "2",
                    "axis": condition.axis,
                    "condition": condition.label,
                    "seed": seed,
                    "expected_output": str(_staged_bank_path(args, condition, seed)),
                }
            )
            results.append(result)
            print(
                f"[stage 2 {len(results)}/{len(conditions) * len(seeds)}] "
                f"{condition.label} seed={seed}: success={result.get('success')} "
                f"output={result.get('output') or result.get('expected_output')}"
            )
    return results


def run_staged_stage25(args: argparse.Namespace, conditions: list[AxisCondition], seeds: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in seeds:
            cfg, _ = _condition_cfg_for_stage(condition, args, seed=seed)
            retriever_key = str(cfg.get("pipeline", {}).get("memory_retriever", "ps_topk"))
            retriever_cfg = cfg.get("components", {}).get("memory_retriever", {}) or {}
            bank_path = _staged_bank_path(args, condition, seed)
            output_path = _staged_retrieval_path(args, condition, seed)
            problems_path = _write_stage_problem_file(args, conditions, seed=seed)
            if not bank_path.exists():
                result = _failed_result(
                    condition,
                    error=f"missing Stage-2 bank: {bank_path}",
                    wall_time_s=0.0,
                    seed=seed,
                )
                result["stage"] = "2.5"
                result["expected_output"] = str(output_path)
                results.append(result)
                continue
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "precompute_retrieval.py"),
                "--input-bank",
                str(bank_path),
                "--retriever",
                retriever_key,
                "--top-k",
                str(args.top_k),
                "--problems",
                str(problems_path),
                "--seed",
                str(seed),
                "--output",
                str(output_path),
                "--retriever-cfg",
                json.dumps(retriever_cfg, sort_keys=False),
            ]
            result = _run_stage_script(
                command,
                result_prefix="PRECOMPUTE_RETRIEVAL_RESULT_JSON=",
                timeout_s=args.condition_timeout_s,
            )
            result.update(
                {
                    "stage": "2.5",
                    "axis": condition.axis,
                    "condition": condition.label,
                    "seed": seed,
                    "input_bank": str(bank_path),
                    "problems": str(problems_path),
                    "expected_output": str(output_path),
                }
            )
            results.append(result)
            print(
                f"[stage 2.5 {len(results)}/{len(conditions) * len(seeds)}] "
                f"{condition.label} seed={seed}: success={result.get('success')} "
                f"hints={result.get('non_null_hints')}/{result.get('problem_count')}"
            )
    return results


def run_staged_stage3(args: argparse.Namespace, conditions: list[AxisCondition], seeds: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_conditions) as pool:
        future_to_target: dict[concurrent.futures.Future[dict[str, Any]], tuple[AxisCondition, int]] = {}
        for condition in conditions:
            for seed in seeds:
                bank_path = _staged_bank_path(args, condition, seed)
                retrieval_path = _staged_retrieval_path(args, condition, seed)
                if not bank_path.exists() or not retrieval_path.exists():
                    missing = bank_path if not bank_path.exists() else retrieval_path
                    result = _failed_result(
                        condition,
                        error=f"missing staged artifact: {missing}",
                        wall_time_s=0.0,
                        seed=seed,
                    )
                    result["stage"] = "3"
                    results.append(result)
                    print(
                        f"[stage 3 {len(results)}/{len(conditions) * len(seeds)}] "
                        f"{condition.label} seed={seed}: missing artifact"
                    )
                    continue
                child_args = copy.copy(args)
                child_args.seed = seed
                child_args.staged_bank = bank_path
                child_args.retrieval_from_file = retrieval_path
                future = pool.submit(_run_condition_subprocess, condition, child_args)
                future_to_target[future] = (condition, seed)
        for future in concurrent.futures.as_completed(future_to_target):
            condition, seed = future_to_target[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _failed_result(condition, error=str(exc), wall_time_s=0.0, seed=seed)
                result["engagement_verdict"] = "NO - harness failed"
            result["stage"] = "3"
            result["staged_bank"] = str(_staged_bank_path(args, condition, seed))
            result["retrieval_from_file"] = str(_staged_retrieval_path(args, condition, seed))
            results.append(result)
            print(
                f"[stage 3 {len(results)}/{len(conditions) * len(seeds)}] "
                f"{condition.label} seed={seed}: score={_score_text(result.get('score'))} "
                f"success={result.get('success')}"
            )
    return results


def render_staged(
    results: list[dict[str, Any]],
    *,
    started_at: datetime,
    wall_time_s: float,
    stages: list[str],
    seeds: list[int],
    n_problems: int,
    model: str,
) -> str:
    success_count = sum(1 for row in results if row.get("success"))
    lines = [
        "# Staged Pipeline Results",
        "",
        "## Configuration",
        f"- Stages: {', '.join(stages)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Problems per seed: {n_problems}",
        f"- Model: {model}",
        f"- Started UTC: {started_at.isoformat(timespec='seconds')}",
        f"- Wall time: {wall_time_s / 60:.2f} minutes",
        f"- Successful rows: {success_count}/{len(results)}",
        "",
        "## Stage Rows",
        "",
        "| Stage | Axis | Condition | Seed | Success | Score | Output | Notes |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for row in sorted(results, key=lambda item: (str(item.get("stage")), _axis_sort_key(item.get("axis")), str(item.get("condition")), int(item.get("seed") or 0))):
        output = (
            row.get("output")
            or row.get("expected_output")
            or row.get("retrieval_from_file")
            or row.get("trace_dir")
            or ""
        )
        notes = row.get("error") or row.get("engagement_verdict") or "OK"
        lines.append(
            "| {stage} | {axis} | {condition} | {seed} | {success} | {score} | {output} | {notes} |".format(
                stage=row.get("stage", ""),
                axis=row.get("axis", ""),
                condition=str(row.get("condition", "")).replace("|", "/"),
                seed=row.get("seed", ""),
                success=row.get("success"),
                score=_score_text(row.get("score")),
                output=_text_preview(str(output), limit=120).replace("|", "/"),
                notes=_text_preview(str(notes), limit=220).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_staged(args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions = select_conditions(load_axis_conditions(), args.ports, args.axis_filter)
    seeds = list(getattr(args, "seed_list", [args.seed]))
    selected_stages = ["2", "2.5", "3"] if args.stage == "all" else [str(args.stage)]
    started_at = datetime.now(UTC)
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    for stage in selected_stages:
        if stage == "2":
            results.extend(run_staged_stage2(args, conditions, seeds))
        elif stage == "2.5":
            results.extend(run_staged_stage25(args, conditions, seeds))
        elif stage == "3":
            results.extend(run_staged_stage3(args, conditions, seeds))
        else:
            raise ValueError(f"Unknown staged stage: {stage}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wall_time_s = time.monotonic() - start
    args.out.write_text(
        render_staged(
            results,
            started_at=started_at,
            wall_time_s=wall_time_s,
            stages=selected_stages,
            seeds=seeds,
            n_problems=args.n_problems,
            model=args.model,
        ),
        encoding="utf-8",
    )
    write_json(
        args.out.with_suffix(".json"),
        {
            "mode": "staged",
            "stage": args.stage,
            "stages": selected_stages,
            "config": {
                "n_problems": args.n_problems,
                "seeds": seeds,
                "iters": args.iters,
                "cache": args.cache_enabled,
                "ignore_cache": args.ignore_cache,
                "max_workers": args.max_workers,
                "model": args.model,
                "started_at_utc": started_at.isoformat(timespec="seconds"),
                "wall_time_s": wall_time_s,
                "artifact_root": str(_staged_root(args)),
            },
            "results": results,
        },
    )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trace-enabled smoke sweep across all axis conditions")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["smoke", "integrated", "phase-g-lite", "staged"], default="smoke")
    parser.add_argument("--stage", choices=["2", "2.5", "3", "all"], default=None)
    parser.add_argument("--port", help="Internal single-condition port label")
    parser.add_argument("--ports", nargs="*", default=None, help="Optional subset of condition labels")
    parser.add_argument("--axes", default=None, help="Comma-separated axis labels to include")
    parser.add_argument("--conditions", default=None, help="Comma-separated alias for --ports")
    parser.add_argument("--n-problems", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list for sweep mode")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--base-config", type=Path, default=Path("configs/experiments/phase1_arc_base.yaml"))
    parser.add_argument("--label", default="smoke-2026-05-13")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cache", default="true", help="Set false to force fresh LLM calls via ignore_cache=true")
    parser.add_argument("--parallel-conditions", type=int, default=4)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--condition-timeout-s", type=int, default=900)
    parser.add_argument("--dotenv-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Directory for run traces and default summary.md")
    parser.add_argument("--output-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--staged-bank", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--retrieval-from-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("case_studies/synthesis/2026-05-13_smoke_sweep_validation.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = normalize_args(parse_args(argv))
    if args.run_one:
        if not args.port:
            raise ValueError("--run-one requires --port")
        run_one_condition(args)
        return
    if args.mode == "staged":
        results = run_staged(args)
        success_count = sum(1 for row in results if row.get("success"))
        print(
            f"Wrote {args.out} with {success_count}/{len(results)} successful staged rows."
        )
        return
    results = run_sweep(args)
    success_count = sum(1 for row in results if row.get("success"))
    explicit_yes_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("YES"))
    check_count = sum(1 for row in results if str(row.get("engagement_verdict", "")).startswith("CHECK"))
    print(
        f"Wrote {args.out} with {success_count}/{len(results)} successful runs, "
        f"{explicit_yes_count} YES verdicts, and {check_count} CHECK verdicts."
    )


if __name__ == "__main__":
    main()
