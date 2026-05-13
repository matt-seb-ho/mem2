from __future__ import annotations

import json
from pathlib import Path

from analysis._shared.load_runs import load_run
from analysis.failure_taxonomy.classify import classify_run
from analysis.memory_growth.extract import extract_growth
from analysis.provenance.tracker import track_provenance
from analysis.retrieval_telemetry.extract import extract_telemetry


def _dummy_case_run(root: Path) -> Path:
    run_dir = root / "case_run"
    iter_dir = run_dir / "problems" / "task1" / "iter_0"
    iter_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": "case_run", "port": "flat_topk"}) + "\n",
        encoding="utf-8",
    )
    (iter_dir / "retrieval_bundle.json").write_text(
        json.dumps({"retrieved_items": [{"uid": "c1"}], "metadata": {"scoring_mode": "unit"}}) + "\n",
        encoding="utf-8",
    )
    (iter_dir / "eval.json").write_text(
        json.dumps({"correct": False}) + "\n",
        encoding="utf-8",
    )
    (iter_dir / "parsed.json").write_text(
        json.dumps({"attempts": []}) + "\n",
        encoding="utf-8",
    )
    (iter_dir / "call_meta.json").write_text(
        json.dumps({"latency_s": 0.1}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_load_run_collects_case_study_traces(tmp_path):
    run_dir = _dummy_case_run(tmp_path)

    run = load_run(run_dir)

    assert run["run_id"] == "case_run"
    assert len(run["traces"]) == 1
    assert run["traces"][0]["task_id"] == "task1"


def test_analysis_stubs_write_placeholder_json(tmp_path):
    run_dir = _dummy_case_run(tmp_path)
    outputs = {
        "failure": classify_run(run_dir, tmp_path / "failure.json"),
        "growth": extract_growth(run_dir, tmp_path / "growth.json"),
        "telemetry": extract_telemetry(run_dir, tmp_path / "telemetry.json"),
        "provenance": track_provenance(run_dir, tmp_path / "provenance.json"),
    }

    assert _read(outputs["failure"])["status"] == "pending_no_llm_classifier"
    assert _read(outputs["growth"])["memory_size_over_time"] == []
    assert _read(outputs["telemetry"])["retrieval_hit_counts"] == {}
    assert _read(outputs["provenance"])["lineage"] == []
