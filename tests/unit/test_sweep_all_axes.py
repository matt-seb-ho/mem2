from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from case_studies.scripts.sweep_all_axes import (
    AxisCondition,
    configure_smoke_run,
    engagement_verdict,
    load_axis_conditions,
    render_aggregate,
    summarize_run,
)


def _write_dummy_trace(run_dir: Path) -> None:
    iter_dir = run_dir / "problems" / "task1" / "iter_0"
    call_dir = iter_dir / "llm_calls" / "call_0001"
    call_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"total_cost_usd": 0.01}) + "\n", encoding="utf-8")
    (iter_dir / "prompt.txt").write_text("Prompt with graphrag community report content", encoding="utf-8")
    (iter_dir / "retrieval_bundle.json").write_text(
        json.dumps(
            {
                "hint_text": "community summary",
                "retrieved_items": [{"id": "c1", "text": "concept"}],
                "metadata": {"scoring_mode": "graphrag_community_reports", "top_k": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (call_dir / "call_meta.json").write_text(json.dumps({"latency_s": 0.1}) + "\n", encoding="utf-8")


def test_load_axis_conditions_includes_known_ports():
    labels = {condition.label for condition in load_axis_conditions()}

    assert "flat_topk" in labels
    assert "graphrag" in labels
    assert "reorg_amem" in labels


def test_configure_smoke_run_sets_model_concurrency_and_dotenv(tmp_path):
    cfg = {"components": {"provider": {}, "inference_engine": {}}, "run": {}}
    out = configure_smoke_run(
        cfg,
        max_workers=512,
        model="deepseek/deepseek-v4-flash",
        max_tokens=2048,
        dotenv_path=tmp_path / ".env",
    )

    assert out["components"]["provider"]["default_max_concurrency"] == 512
    assert out["components"]["provider"]["dotenv_path"].endswith(".env")
    assert out["components"]["inference_engine"]["model"] == "deepseek/deepseek-v4-flash"
    assert out["components"]["inference_engine"]["gen_cfg"]["max_tokens"] == 2048
    assert out["components"]["inference_engine"]["gen_cfg"]["batch_size"] == 512


def test_summarize_run_detects_visible_label_engagement(tmp_path):
    run_dir = tmp_path / "run"
    _write_dummy_trace(run_dir)
    condition = AxisCondition(axis="1", label="graphrag", condition={"label": "graphrag"})

    summary = summarize_run(condition, run_dir, success=True, wall_time_s=1.0)

    assert summary["llm_calls"] == 1
    assert summary["retrieval_hits"] == 2
    assert summary["engagement_verdict"] == "YES - label visible"
    assert summary["cost_usd"] == 0.01


def test_engagement_verdict_flags_fallback_and_baseline():
    baseline = AxisCondition(axis="1", label="flat_topk", condition={"label": "flat_topk", "baseline": True})
    adapted = AxisCondition(axis="1", label="pathrag", condition={"label": "pathrag"})

    assert engagement_verdict(baseline, retrieval_hits=0, metadata=[], prompt="") == "N/A baseline"
    assert engagement_verdict(adapted, retrieval_hits=1, metadata=["scoring_mode=fallback_flat"], prompt="") == "NO - fallback metadata"


def test_render_aggregate_writes_expected_sections(tmp_path):
    run_dir = tmp_path / "case_studies" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    results = [
        {
            "axis": "1",
            "condition": "graphrag",
            "success": True,
            "engagement_verdict": "YES - label visible",
            "retrieval_metadata": ["scoring_mode=graphrag"],
            "sample_prompt_snippet": "prompt",
            "sample_retrieval": "retrieval",
            "trace_dir": str(run_dir),
            "llm_calls": 1,
            "cost_usd": 0.01,
        }
    ]

    rendered = render_aggregate(results, started_at=datetime(2026, 5, 13, tzinfo=UTC), wall_time_s=60.0)

    assert "# Smoke Sweep Validation - 2026-05-13" in rendered
    assert "| 1 | graphrag | YES - label visible |" in rendered
    assert "Total LLM calls: 1" in rendered
    assert "$0.0100" in rendered
