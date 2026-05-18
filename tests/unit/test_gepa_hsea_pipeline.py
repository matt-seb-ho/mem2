from __future__ import annotations

import asyncio

from mem2.branches.inference_engine.gepa_hsea import (
    GepaHseaInferenceEngine,
    make_debug_prompt,
    make_hypothesize_prompt,
    make_synthesize_prompt,
)
from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext, TrajectoryPlan


class FakeProvider:
    name = "fake"
    version = "test"
    supports_multi_completion = True

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def async_generate(self, prompt: str, model: str, gen_cfg: dict) -> list[str]:
        self.calls.append({"prompt": prompt, "model": model, "gen_cfg": dict(gen_cfg)})
        if not self.responses:
            raise AssertionError("FakeProvider response queue exhausted")
        return [self.responses.pop(0)]


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="p1",
        train_pairs=[
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            {"input": [[5]], "output": [[5]]},
        ],
        test_pairs=[{"input": [[7]], "output": [[7]]}],
    )


def _ctx() -> RunContext:
    return RunContext(run_id="test", seed=0, config={}, output_dir="/tmp/mem2-test")


def _retrieval() -> RetrievalBundle:
    return RetrievalBundle(
        problem_uid="p1",
        hint_text="Identity transformations often preserve every colored cell.",
        retrieved_items=[],
    )


def _plan() -> TrajectoryPlan:
    return TrajectoryPlan(num_paths=1, strategy="single_path")


def _run(coro):
    return asyncio.run(coro)


def test_prompt_builders_match_hsea_contract():
    problem = _problem()
    h_prompt = make_hypothesize_prompt(problem, "short hint")
    assert "First, reason step-by-step" in h_prompt
    assert 'prefixed exactly with "Rule:"' in h_prompt
    assert "short hint" in h_prompt

    s_prompt = make_synthesize_prompt(problem, "copy the input grid")
    assert "transform_matrix(grid)" in s_prompt
    assert "Return only one ```python fenced code block." in s_prompt

    d_prompt = make_debug_prompt(problem, "def transform_matrix(grid): return grid", "Pair 1: FAIL")
    assert "Previous code:" in d_prompt
    assert "Execution trace per training pair:" in d_prompt
    assert "Diagnose the failure mode in 1-2 sentences." in d_prompt


def test_hsea_stages_fire_and_debug_can_reach_answer():
    provider = FakeProvider(
        [
            "The grid is unchanged.\nRule: return the input grid unchanged.",
            "```python\ndef transform_matrix(grid):\n    return [[0 for _ in row] for row in grid]\n```",
            "The code filled zeros instead of copying cells.\n```python\ndef transform_matrix(grid):\n    return grid\n```",
        ]
    )
    engine = GepaHseaInferenceEngine(model="fake", max_retries=3)

    attempts = _run(
        engine.initial_attempt(
            _ctx(),
            provider,
            _problem(),
            _retrieval(),
            _plan(),
        )
    )

    assert len(attempts) == 1
    metadata = attempts[0].metadata
    assert metadata["gepa_hsea_stage_order"] == [
        "hypothesize",
        "synthesize",
        "execute",
        "debug",
        "execute",
        "answer",
    ]
    assert metadata["gepa_hsea_all_train_pass_gate_hit"] is True
    assert metadata["gepa_hsea_debug_iterations"] == 1
    assert len(provider.calls) == 3
    assert "def transform(input_grid)" in attempts[0].completion


def test_all_train_pass_gate_skips_debug_when_synthesis_succeeds():
    provider = FakeProvider(
        [
            "The output copies every input cell.\nRule: copy the input grid exactly.",
            "```python\ndef transform_matrix(grid):\n    return grid\n```",
        ]
    )
    engine = GepaHseaInferenceEngine(model="fake", max_retries=3)

    attempts = _run(
        engine.initial_attempt(
            _ctx(),
            provider,
            _problem(),
            _retrieval(),
            _plan(),
        )
    )

    metadata = attempts[0].metadata
    assert metadata["gepa_hsea_all_train_pass_gate_hit"] is True
    assert metadata["gepa_hsea_debug_iterations"] == 0
    assert "debug" not in metadata["gepa_hsea_stage_order"]
    assert len(provider.calls) == 2


def test_retry_loop_terminates_at_max_retries():
    provider = FakeProvider(
        [
            "The output should change.\nRule: replace the input with value 9.",
            "```python\ndef transform_matrix(grid):\n    return grid\n```",
            "Still wrong.\n```python\ndef transform_matrix(grid):\n    return grid\n```",
            "Still wrong again.\n```python\ndef transform_matrix(grid):\n    return grid\n```",
        ]
    )
    problem = ProblemSpec(
        uid="p2",
        train_pairs=[{"input": [[1]], "output": [[9]]}],
        test_pairs=[{"input": [[2]], "output": [[9]]}],
    )
    engine = GepaHseaInferenceEngine(model="fake", max_retries=2)

    attempts = _run(
        engine.initial_attempt(
            _ctx(),
            provider,
            problem,
            _retrieval(),
            _plan(),
        )
    )

    metadata = attempts[0].metadata
    assert metadata["gepa_hsea_all_train_pass_gate_hit"] is False
    assert metadata["gepa_hsea_debug_iterations"] == 2
    assert metadata["gepa_hsea_stage_order"].count("debug") == 2
    assert len(provider.calls) == 4
