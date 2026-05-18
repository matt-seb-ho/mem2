from __future__ import annotations

import asyncio
import json
from pathlib import Path

from case_studies._tracer import (
    TraceCollectingProviderClient,
    count_llm_calls,
    reset_trace_context,
    set_trace_context,
    write_attempt_eval_trace,
)
from mem2.core.entities import AttemptRecord, EvalRecord
from mem2.cli.run import deep_merge, load_yaml
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components
from mem2.providers.mock_client import MockProviderClient


def _load_smoke_config() -> dict:
    cfg = load_yaml(Path("configs/experiments/smoke_arc.yaml"))
    if "_base_" in cfg:
        base = load_yaml(Path("configs/base.yaml"))
        cfg = deep_merge(base, {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


def test_trace_collecting_provider_writes_prompt_response_and_meta(tmp_path):
    trace_dir = tmp_path / "case_run"
    provider = TraceCollectingProviderClient(MockProviderClient(), trace_dir=trace_dir)

    async def _run() -> None:
        tokens = set_trace_context(trace_dir, "task/1", 0)
        try:
            await provider.async_generate("hello", "mock", {"n": 1})
        finally:
            reset_trace_context(tokens)

    asyncio.run(_run())

    iter_dir = trace_dir / "problems" / "task_1" / "iter_0"
    assert (iter_dir / "prompt.txt").read_text(encoding="utf-8") == "hello"
    assert "transform" in (iter_dir / "response.txt").read_text(encoding="utf-8")
    call_meta = json.loads((iter_dir / "call_meta.json").read_text(encoding="utf-8"))
    assert call_meta["provider"] == "mock"
    assert call_meta["request_kind"] == "generate"
    assert count_llm_calls(trace_dir) == 1


def test_runner_opt_in_trace_dir_writes_problem_artifacts(tmp_path):
    trace_dir = tmp_path / "case_run"
    cfg = _load_smoke_config()
    cfg.setdefault("run", {})["output_root"] = str(tmp_path / "runs")
    cfg.setdefault("case_studies", {})["trace_dir"] = str(trace_dir)
    cfg["case_studies"]["port"] = "none"
    cfg["case_studies"]["label"] = "unit"
    cfg.setdefault("components", {}).setdefault("benchmark", {})["limit"] = 1

    components = resolve_components(cfg)
    bundle = run_sync(cfg, components)

    assert bundle.summary["total_attempts"] > 0
    meta = json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == bundle.summary.get("run_id", meta["run_id"])
    assert meta["port"] == "none"
    assert meta["n_problems"] == 1
    assert meta["llm_call_count"] >= 1

    iter_dirs = list((trace_dir / "problems").glob("*/iter_0"))
    assert len(iter_dirs) == 1
    iter_dir = iter_dirs[0]
    for name in [
        "prompt.txt",
        "response.txt",
        "call_meta.json",
        "retrieval_bundle.json",
        "parsed.json",
        "eval.json",
    ]:
        assert (iter_dir / name).exists(), name


def test_attempt_eval_trace_promotes_final_attempt_response(tmp_path):
    trace_dir = tmp_path / "case_run"
    iter_dir = trace_dir / "problems" / "task1" / "iter_0"
    iter_dir.mkdir(parents=True)
    (iter_dir / "response.txt").write_text("Rule: first-stage hypothesis", encoding="utf-8")

    attempt = AttemptRecord(
        problem_uid="task1",
        pass_idx=0,
        branch_id="gepa_hsea",
        prompt="debug prompt",
        completion="```python\ndef transform(input_grid):\n    return input_grid\n```",
        metadata={"gepa_hsea_stage_order": ["hypothesize", "synthesize", "execute", "answer"]},
    )
    eval_record = EvalRecord(
        problem_uid="task1",
        attempt_idx=0,
        is_correct=True,
        train_details=[],
        test_details=[],
        metadata={},
    )

    write_attempt_eval_trace([attempt], [eval_record], trace_dir=trace_dir, task_id="task1", iter_id=0)

    assert "def transform" in (iter_dir / "response.txt").read_text(encoding="utf-8")
    parsed = json.loads((iter_dir / "parsed.json").read_text(encoding="utf-8"))
    assert parsed["attempts"][0]["branch_id"] == "gepa_hsea"
    assert json.loads((iter_dir / "eval.json").read_text(encoding="utf-8"))["correct"] is True
