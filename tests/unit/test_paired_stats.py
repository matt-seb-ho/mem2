from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.paired_stats import (
    ConditionRun,
    all_pairwise_stats,
    bootstrap_ci,
    compare_conditions,
    mcnemar_yates_pvalue,
    run,
)


def _condition(name: str, values: list[bool]) -> ConditionRun:
    return ConditionRun(
        condition=name,
        axis="1",
        seed=-1,
        score=sum(values) / len(values),
        n_total=len(values),
        parity_grade="test",
        run_id=name,
        per_problem={(42, f"p{idx:02d}"): value for idx, value in enumerate(values)},
    )


def test_mcnemar_yates_pvalue_detects_asymmetry():
    stat, p_value = mcnemar_yates_pvalue(10, 0)

    assert stat > 8.0
    assert p_value < 0.01


def test_mcnemar_yates_pvalue_equal_discordance_is_not_significant():
    stat, p_value = mcnemar_yates_pvalue(5, 5)

    assert stat == 0.0
    assert p_value == 1.0


def test_bootstrap_ci_bounds_known_difference():
    low, high = bootstrap_ci([1.0] * 10 + [0.0] * 10, n_resamples=200, seed=1)

    assert 0.25 <= low <= 0.50
    assert 0.50 <= high <= 0.75


def test_compare_conditions_aligns_by_problem_uid():
    a = _condition("a", [True] * 10 + [False] * 10)
    b = _condition("b", [False] * 10 + [False] * 10)

    stats = compare_conditions(a, b, n_bootstrap=200)

    assert stats.n == 20
    assert stats.a_acc == 0.5
    assert stats.b_acc == 0.0
    assert stats.b_only == 10
    assert stats.c_only == 0
    assert stats.mcnemar_p < 0.01


def test_all_pairwise_stats_counts_unordered_pairs():
    conditions = {
        "a": _condition("a", [True, False]),
        "b": _condition("b", [False, False]),
        "c": _condition("c", [True, True]),
    }

    assert len(all_pairwise_stats(conditions, n_bootstrap=50)) == 3


def test_run_reads_committed_summary_jsons(tmp_path, monkeypatch):
    repo = tmp_path
    run_a = repo / "case_studies" / "runs" / "run_a"
    run_b = repo / "case_studies" / "runs" / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    for run_dir, condition, values in [
        (run_a, "a", [True, True, False, False]),
        (run_b, "b", [False, True, False, False]),
    ]:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "condition": condition,
                    "axis": "1",
                    "parity_grade": "test",
                    "seed": 42,
                    "score": sum(values) / len(values),
                    "n_total": len(values),
                    "per_problem": [
                        {"problem_uid": f"p{idx}", "correct": value}
                        for idx, value in enumerate(values)
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    aggregate = repo / "case_studies" / "synthesis" / "results.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "results": [
                    {"condition": "a", "run_id": "run_a", "seed": 42, "score_summary_path": "/stale/path/a.json"},
                    {"condition": "b", "run_id": "run_b", "seed": 42, "score_summary_path": "/stale/path/b.json"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import scripts.analysis.paired_stats as paired_stats

    monkeypatch.setattr(paired_stats, "REPO_ROOT", repo)
    out = repo / "case_studies" / "synthesis" / "paired.md"

    run(aggregate, out, n_bootstrap=50, title="paired")

    assert "# paired" in out.read_text(encoding="utf-8")
    assert (out.with_suffix(".json")).exists()
