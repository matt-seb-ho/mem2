from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ConditionRun:
    condition: str
    axis: str
    seed: int
    score: float
    n_total: int
    parity_grade: str
    run_id: str
    per_problem: dict[tuple[int, str], bool]


@dataclass(frozen=True)
class PairStats:
    condition_a: str
    condition_b: str
    axis_a: str
    axis_b: str
    n: int
    a_acc: float
    b_acc: float
    gap: float
    b_only: int
    c_only: int
    mcnemar_stat: float
    mcnemar_p: float
    ci_low: float
    ci_high: float


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mcnemar_yates_pvalue(b_only: int, c_only: int) -> tuple[float, float]:
    discordant = b_only + c_only
    if discordant == 0:
        return 0.0, 1.0
    numerator = max(0, abs(b_only - c_only) - 1) ** 2
    stat = numerator / discordant
    p_value = math.erfc(math.sqrt(stat / 2.0))
    return stat, p_value


def bootstrap_ci(
    paired_diffs: list[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> tuple[float, float]:
    if not paired_diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(paired_diffs)
    estimates = []
    for _ in range(n_resamples):
        estimates.append(sum(paired_diffs[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, min(len(estimates) - 1, int((alpha / 2.0) * len(estimates))))
    hi_idx = max(0, min(len(estimates) - 1, int((1.0 - alpha / 2.0) * len(estimates)) - 1))
    return estimates[lo_idx], estimates[hi_idx]


def compare_conditions(
    condition_a: ConditionRun,
    condition_b: ConditionRun,
    *,
    n_bootstrap: int = 1000,
) -> PairStats:
    common = sorted(set(condition_a.per_problem) & set(condition_b.per_problem))
    if not common:
        return PairStats(
            condition_a=condition_a.condition,
            condition_b=condition_b.condition,
            axis_a=condition_a.axis,
            axis_b=condition_b.axis,
            n=0,
            a_acc=0.0,
            b_acc=0.0,
            gap=0.0,
            b_only=0,
            c_only=0,
            mcnemar_stat=0.0,
            mcnemar_p=1.0,
            ci_low=0.0,
            ci_high=0.0,
        )
    a_values = [condition_a.per_problem[key] for key in common]
    b_values = [condition_b.per_problem[key] for key in common]
    b_only = sum(1 for a, b in zip(a_values, b_values, strict=True) if a and not b)
    c_only = sum(1 for a, b in zip(a_values, b_values, strict=True) if (not a) and b)
    stat, p_value = mcnemar_yates_pvalue(b_only, c_only)
    diffs = [float(a) - float(b) for a, b in zip(a_values, b_values, strict=True)]
    ci_low, ci_high = bootstrap_ci(diffs, n_resamples=n_bootstrap)
    a_acc = sum(float(value) for value in a_values) / len(a_values)
    b_acc = sum(float(value) for value in b_values) / len(b_values)
    return PairStats(
        condition_a=condition_a.condition,
        condition_b=condition_b.condition,
        axis_a=condition_a.axis,
        axis_b=condition_b.axis,
        n=len(common),
        a_acc=a_acc,
        b_acc=b_acc,
        gap=a_acc - b_acc,
        b_only=b_only,
        c_only=c_only,
        mcnemar_stat=stat,
        mcnemar_p=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _resolve_summary_path(row: dict[str, Any]) -> Path:
    run_id = row.get("run_id")
    if run_id:
        portable = REPO_ROOT / "case_studies" / "runs" / str(run_id) / "summary.json"
        if portable.exists():
            return portable
    raw_path = row.get("score_summary_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        if not path.is_absolute() and (REPO_ROOT / path).exists():
            return REPO_ROOT / path
    raise FileNotFoundError(f"Could not resolve summary.json for row: {row.get('condition')} {row.get('seed')}")


def _load_condition_runs_from_aggregate(path: Path) -> list[ConditionRun]:
    payload = load_json(path)
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    runs = []
    for row in rows:
        summary = load_json(_resolve_summary_path(row))
        seed = int(summary["seed"])
        per_problem = {
            (seed, str(item["problem_uid"])): bool(item["correct"])
            for item in summary.get("per_problem", [])
            if item.get("problem_uid") is not None
        }
        runs.append(
            ConditionRun(
                condition=str(summary["condition"]),
                axis=str(summary["axis"]),
                seed=seed,
                score=float(summary.get("score") or 0.0),
                n_total=int(summary.get("n_total") or len(per_problem)),
                parity_grade=str(summary.get("parity_grade") or "unknown"),
                run_id=str(summary.get("run_id") or row.get("run_id") or ""),
                per_problem=per_problem,
            )
        )
    return runs


def merge_seed_runs(runs: list[ConditionRun]) -> dict[str, ConditionRun]:
    grouped: dict[str, list[ConditionRun]] = {}
    for run in runs:
        grouped.setdefault(run.condition, []).append(run)
    merged = {}
    for condition, condition_runs in grouped.items():
        per_problem: dict[tuple[int, str], bool] = {}
        for run in condition_runs:
            per_problem.update(run.per_problem)
        n_total = len(per_problem)
        score = sum(1 for value in per_problem.values() if value) / n_total if n_total else 0.0
        first = sorted(condition_runs, key=lambda item: item.seed)[0]
        merged[condition] = ConditionRun(
            condition=condition,
            axis=first.axis,
            seed=-1,
            score=score,
            n_total=n_total,
            parity_grade=first.parity_grade,
            run_id=",".join(run.run_id for run in sorted(condition_runs, key=lambda item: item.seed)),
            per_problem=per_problem,
        )
    return merged


def all_pairwise_stats(
    conditions: dict[str, ConditionRun],
    *,
    n_bootstrap: int = 1000,
) -> list[PairStats]:
    comparisons = []
    for condition_a, condition_b in itertools.combinations(sorted(conditions), 2):
        comparisons.append(compare_conditions(conditions[condition_a], conditions[condition_b], n_bootstrap=n_bootstrap))
    return comparisons


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fmt_p(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def render_markdown(
    conditions: dict[str, ConditionRun],
    comparisons: list[PairStats],
    *,
    title: str,
    n_bootstrap: int,
) -> str:
    alpha = 0.05 / len(comparisons) if comparisons else 0.05
    significant = [row for row in comparisons if row.mcnemar_p < alpha]
    flat = conditions.get("flat_topk")
    lines = [
        f"# {title}",
        "",
        "## Methodology",
        "- McNemar's test uses paired correct/wrong asymmetry with Yates continuity correction.",
        f"- Bootstrap 95% CI uses {n_bootstrap} resamples of paired accuracy differences.",
        f"- Bonferroni correction across {len(comparisons)} condition pairs: alpha_corrected={alpha:.6g}.",
        "- Pairing key: seed plus ARC problem UID.",
        "",
        "## Significantly different pairs",
        "",
        "| Condition A | Condition B | A acc | B acc | Gap | McNemar p | CI 95 | n |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    if significant:
        for row in sorted(significant, key=lambda item: item.mcnemar_p):
            lines.append(
                f"| {row.condition_a} | {row.condition_b} | {_fmt(row.a_acc)} | {_fmt(row.b_acc)} | "
                f"{_fmt(row.gap)} | {_fmt_p(row.mcnemar_p)} | [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] | {row.n} |"
            )
    else:
        lines.append("| none | none |  |  |  |  |  |  |")

    lines.extend(["", "## Per-axis ranking", ""])
    by_axis: dict[str, list[ConditionRun]] = {}
    for condition in conditions.values():
        by_axis.setdefault(condition.axis, []).append(condition)
    flat_pairs = {row.condition_a: row for row in comparisons if flat and row.condition_b == "flat_topk"}
    flat_pairs.update({row.condition_b: row for row in comparisons if flat and row.condition_a == "flat_topk"})
    for axis in sorted(by_axis, key=lambda value: (int(value) if str(value).isdigit() else 999, str(value))):
        lines.extend([f"### Axis {axis}", "", "| Rank | Condition | Mean acc | vs flat_topk gap CI | n |", "|---:|---|---:|---|---:|"])
        ranked = sorted(by_axis[axis], key=lambda item: item.score, reverse=True)
        for idx, condition in enumerate(ranked, start=1):
            if condition.condition == "flat_topk":
                ci_text = "[0.000, 0.000]"
            elif condition.condition in flat_pairs:
                pair = flat_pairs[condition.condition]
                if pair.condition_a == condition.condition:
                    ci_text = f"[{_fmt(pair.ci_low)}, {_fmt(pair.ci_high)}]"
                else:
                    ci_text = f"[{_fmt(-pair.ci_high)}, {_fmt(-pair.ci_low)}]"
            else:
                ci_text = "n/a"
            lines.append(f"| {idx} | {condition.condition} | {_fmt(condition.score)} | {ci_text} | {condition.n_total} |")
        lines.append("")

    lines.extend(
        [
            "## All pairwise comparisons",
            "",
            "| Condition A | Condition B | A acc | B acc | Gap | McNemar p | CI 95 | b | c | n |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(comparisons, key=lambda item: (item.axis_a, item.condition_a, item.condition_b)):
        lines.append(
            f"| {row.condition_a} | {row.condition_b} | {_fmt(row.a_acc)} | {_fmt(row.b_acc)} | "
            f"{_fmt(row.gap)} | {_fmt_p(row.mcnemar_p)} | [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] | "
            f"{row.b_only} | {row.c_only} | {row.n} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(input_path: Path, output_path: Path, *, n_bootstrap: int = 1000, title: str) -> Path:
    runs = _load_condition_runs_from_aggregate(input_path)
    conditions = merge_seed_runs(runs)
    comparisons = all_pairwise_stats(conditions, n_bootstrap=n_bootstrap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(conditions, comparisons, title=title, n_bootstrap=n_bootstrap),
        encoding="utf-8",
    )
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "conditions": {
                    key: {
                        "axis": value.axis,
                        "score": value.score,
                        "n_total": value.n_total,
                        "parity_grade": value.parity_grade,
                    }
                    for key, value in sorted(conditions.items())
                },
                "comparisons": [row.__dict__ for row in comparisons],
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired statistics for Phase G-lite case-study summaries")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("case_studies/synthesis/2026-05-13_phase_g_lite_results.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("case_studies/synthesis/2026-05-13_phase_g_lite_paired_stats.md"),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--title", default="Paired Statistical Comparison - Phase G-Lite - 2026-05-13")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = run(args.input, args.out, n_bootstrap=args.bootstrap, title=args.title)
    print(f"Wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
