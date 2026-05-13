from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from case_studies.scripts.inspect_run import inspect_run
from case_studies.scripts.link_to_method import link_run_to_method
from case_studies.scripts.render_diff import write_diff
from case_studies.scripts.render_markdown import write_summary
from case_studies.scripts.run_case_study import build_case_study_config, dry_run_summary


def _write_dummy_run(run_dir: Path, *, prompt: str = "prompt A", response: str = "response A") -> None:
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
    (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (iter_dir / "response.txt").write_text(response, encoding="utf-8")


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


def test_build_case_study_config_preserves_base_builder_seed_for_builder_conditions():
    cfg, _ = build_case_study_config(
        port="accretive_prune",
        n_problems=3,
        seed=42,
        iters=1,
        base_config=Path("configs/experiments/phase1_arc_base.yaml"),
        label="unit",
    )

    assert cfg["pipeline"]["memory_builder"] == "accretive_prune"
    assert cfg["components"]["memory_builder"]["seed_memory_file"] == "data/arc_agi/concept_memory/compressed_v1.json"
    assert cfg["components"]["memory_builder"]["domain"] == "arc"
    assert cfg["components"]["memory_builder"]["max_concepts"] == 200


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


def test_render_diff_writes_prompt_and_response_diff(tmp_path):
    left = tmp_path / "left_run"
    right = tmp_path / "right_run"
    _write_dummy_run(left, prompt="left prompt", response="left response")
    _write_dummy_run(right, prompt="right prompt", response="right response")

    out_path = write_diff(left, right)

    rendered = out_path.read_text(encoding="utf-8")
    assert "# Case Study Diff: left_run vs right_run" in rendered
    assert "-left prompt" in rendered
    assert "+right response" in rendered


def test_link_to_method_creates_symlink_and_updates_readme(tmp_path):
    case_root = tmp_path / "case_studies"
    run_dir = case_root / "runs" / "run1"
    method_dir = case_root / "by_method" / "graphrag"
    method_dir.mkdir(parents=True)
    (method_dir / "README.md").write_text("# Graphrag Case Studies\n\n## Runs\n\nold\n", encoding="utf-8")
    _write_dummy_run(run_dir)

    link_path = link_run_to_method(run_dir, port="graphrag", case_root=case_root)

    assert link_path.is_symlink()
    rendered = (method_dir / "README.md").read_text(encoding="utf-8")
    assert "- [run1](runs/run1/)" in rendered
