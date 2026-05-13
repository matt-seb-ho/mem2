from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from case_studies.scripts.inspect_run import inspect_run
from case_studies.scripts.render_markdown import write_summary
from case_studies.scripts.run_case_study import build_case_study_config, dry_run_summary


def _write_dummy_run(run_dir: Path) -> None:
    iter_dir = run_dir / "problems" / "abc123" / "iter_0"
    iter_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "port": "graphrag",
                "label": "unit",
                "timestamp_utc": "2026-05-13T10:30:00+00:00",
                "n_problems": 1,
                "seed": 42,
                "model": "mock",
                "llm_call_count": 1,
                "total_cost_usd": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_dir / "call_meta.json").write_text(
        json.dumps({"latency_s": 0.125}) + "\n",
        encoding="utf-8",
    )
    (iter_dir / "eval.json").write_text(
        json.dumps({"correct": True}) + "\n",
        encoding="utf-8",
    )
    (iter_dir / "retrieval_bundle.json").write_text(
        json.dumps({"metadata": {"scoring_mode": "unit", "top_k": 3}}) + "\n",
        encoding="utf-8",
    )


def test_build_case_study_config_dry_run_uses_axis_condition():
    now = datetime(2026, 5, 13, 10, 30, 0, tzinfo=UTC)
    cfg, trace_dir = build_case_study_config(
        port="graphrag",
        n_problems=5,
        seed=42,
        iters=1,
        base_config=Path("configs/experiments/smoke_arc.yaml"),
        label="Post substrate engagement",
        now=now,
    )
    summary = dry_run_summary(cfg, trace_dir)

    assert summary["run_id"] == "2026-05-13T10-30-00Z_graphrag_n5_seed42_post-substrate-engagement"
    assert summary["pipeline"]["memory_retriever"] == "graphrag"
    assert summary["benchmark_limit"] == 5
    assert cfg["components"]["provider"]["trace_dir"].endswith(summary["run_id"])
    assert cfg["case_studies"]["axis_config"] == "configs/axes/1.yaml"


def test_render_markdown_writes_summary(tmp_path):
    run_dir = tmp_path / "run1"
    _write_dummy_run(run_dir)

    out_path = write_summary(run_dir)

    rendered = out_path.read_text(encoding="utf-8")
    assert "# Case Study: graphrag - unit" in rendered
    assert "| abc123 | 0 | true | 0.125 |" in rendered
    assert "[abc123/iter_0](problems/abc123/iter_0/)" in rendered


def test_inspect_run_lists_traces(tmp_path):
    run_dir = tmp_path / "run1"
    _write_dummy_run(run_dir)

    rendered = inspect_run(run_dir)

    assert "Port: graphrag" in rendered
    assert "problems/abc123/iter_0 correct=true" in rendered
