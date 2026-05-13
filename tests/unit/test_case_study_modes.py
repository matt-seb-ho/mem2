from __future__ import annotations

import json
from pathlib import Path

from case_studies.scripts.modes.comparative import write_comparative
from case_studies.scripts.modes.counterfactual import write_counterfactual
from case_studies.scripts.modes.error_analysis import write_error_analysis
from case_studies.scripts.modes.phase_shift_envelope import write_phase_shift_envelope
from case_studies.scripts.modes._shared.trace_loader import load_case_run


def _write_trace(
    run_dir: Path,
    *,
    run_id: str,
    task_id: str = "abc123",
    correct: bool = False,
    prompt: str = "solve the grid",
    response: str = "prediction text",
) -> None:
    iter_dir = run_dir / "problems" / task_id / "iter_0"
    iter_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "port": "graphrag",
                "label": "unit",
                "timestamp_utc": "2026-05-13T10:30:00+00:00",
                "n_problems": 1,
                "seed": 42,
                "model": "mock",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (iter_dir / "response.txt").write_text(response, encoding="utf-8")
    (iter_dir / "call_meta.json").write_text(json.dumps({"latency_s": 0.25}) + "\n", encoding="utf-8")
    (iter_dir / "eval.json").write_text(json.dumps({"correct": correct, "expected": "A", "predicted": "B"}) + "\n", encoding="utf-8")
    (iter_dir / "parsed.json").write_text(json.dumps({"code": "pass"}) + "\n", encoding="utf-8")
    (iter_dir / "retrieval_bundle.json").write_text(
        json.dumps(
            {
                "metadata": {"scoring_mode": "unit", "top_k": 2},
                "items": [
                    {"id": "c1", "text": "first concept"},
                    {"id": "c2", "text": "second concept"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")


def test_trace_loader_reads_trace_records(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_trace(run_dir, run_id="run_a", correct=False)

    run = load_case_run(run_dir)

    assert run.run_id == "run_a"
    assert run.traces[0].key == "abc123/iter_0"
    assert run.traces[0].retrieval_bundle["metadata"]["top_k"] == 2


def test_error_analysis_writes_failed_trace_report(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_trace(run_dir, run_id="run_a", correct=False)

    out_path = write_error_analysis(run_dir)

    rendered = out_path.read_text(encoding="utf-8")
    assert "# Error Analysis: run_a" in rendered
    assert "abc123/iter_0" in rendered
    assert "Failure label: TODO" in rendered
    assert "analyses/error_analysis.md" in (run_dir / "summary.md").read_text(encoding="utf-8")


def test_counterfactual_writes_bundle_edit_plan(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_trace(run_dir, run_id="run_a", correct=False)

    out_path = write_counterfactual(run_dir, drop_top_k=1, inject_text="missing concept")

    rendered = out_path.read_text(encoding="utf-8")
    assert "# Counterfactual Bundle Plan: run_a" in rendered
    assert "missing concept" in rendered
    assert "id=c1" in rendered


def test_comparative_writes_side_by_side_report(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left, run_id="left", correct=False, prompt="left prompt", response="left response")
    _write_trace(right, run_id="right", correct=True, prompt="right prompt", response="right response")

    out_path = write_comparative([left, right])

    rendered = out_path.read_text(encoding="utf-8")
    assert "# Comparative Case Study" in rendered
    assert "| abc123/iter_0 | wrong | correct |" in rendered
    assert "-left prompt" in rendered


def test_phase_shift_envelope_writes_stability_matrix(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left, run_id="left", correct=False)
    _write_trace(right, run_id="right", correct=True)

    out_path = write_phase_shift_envelope([left, right])

    rendered = out_path.read_text(encoding="utf-8")
    assert "# Phase-Shift Envelope" in rendered
    assert "| abc123/iter_0 | 2 | 1 | 1 |" in rendered
