"""Tests for pipeline-level Router implementations."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mem2.branches.router._items import extract_item_texts, split_concepts_from_hint
from mem2.branches.router.llm import LlmRouter, _parse_selection
from mem2.branches.router.nli import NliRouter
from mem2.branches.router.none import NoneRouter
from mem2.branches.router.threshold import ThresholdRouter
from mem2.core.entities import ProblemSpec, RetrievalBundle, RunContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def ctx():
    return RunContext(run_id="test-run", seed=42, config={}, output_dir="/tmp/test")


@pytest.fixture()
def problem():
    return ProblemSpec(
        uid="p1",
        train_pairs=[{"input": [[1]], "output": [[2]]}],
        test_pairs=[{"input": [[3]], "output": [[4]]}],
        metadata={"problem_text": "Find the sum of 1+2", "question_content": "Write code"},
    )


@pytest.fixture()
def retrieval():
    return RetrievalBundle(
        problem_uid="p1",
        hint_text="Use dynamic programming",
        retrieved_items=[{"name": "dp"}],
        metadata={"selected_names": ["dp", "greedy"], "pre_filter_count": 5},
    )


@pytest.fixture()
def empty_retrieval():
    return RetrievalBundle(
        problem_uid="p1",
        hint_text=None,
        retrieved_items=[],
        metadata={},
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# TestNoneRouter
# ---------------------------------------------------------------------------
class TestNoneRouter:
    def test_passthrough(self, ctx, problem, retrieval):
        router = NoneRouter()
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=retrieval))
        assert result is retrieval
        assert result.hint_text == "Use dynamic programming"

    def test_metadata_unchanged(self, ctx, problem, retrieval):
        router = NoneRouter()
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=retrieval))
        assert result.metadata == retrieval.metadata

    def test_name(self):
        assert NoneRouter.name == "none"


# ---------------------------------------------------------------------------
# TestThresholdRouter
# ---------------------------------------------------------------------------
class TestThresholdRouter:
    def test_passthrough_no_thresholds(self, ctx, problem, retrieval):
        router = ThresholdRouter(strategy="none")
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=retrieval))
        assert result.hint_text == "Use dynamic programming"
        assert result.metadata["routing_included"] is True

    def test_concept_count_skip(self, ctx, problem, retrieval):
        router = ThresholdRouter(max_concept_count=1)
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=retrieval))
        assert result.hint_text is None
        assert result.metadata["routing_included"] is False
        assert any("concept_count" in r for r in result.metadata["routing_reasons"])

    def test_hint_length_skip(self, ctx, problem):
        long_hint = "x" * 500
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text=long_hint,
            retrieved_items=[],
            metadata={"selected_names": ["a"]},
        )
        router = ThresholdRouter(max_hint_chars=100)
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        assert result.hint_text is None
        assert result.metadata["routing_included"] is False

    def test_metadata_propagation(self, ctx, problem, retrieval):
        router = ThresholdRouter(strategy="none")
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=retrieval))
        # Original metadata keys preserved
        assert "selected_names" in result.metadata
        assert "routing_included" in result.metadata

    def test_name(self):
        assert ThresholdRouter.name == "threshold"


# ---------------------------------------------------------------------------
# TestLlmRouter — per-item filtering, single LLM call
# ---------------------------------------------------------------------------
class TestLlmRouter:
    def _make_ps_bundle(self):
        return RetrievalBundle(
            problem_uid="p1",
            hint_text="- concept: dp\n  description: dynamic prog\n- concept: greedy\n  description: greedy alg",
            retrieved_items=[{"concept": "dp"}, {"concept": "greedy"}],
            metadata={"selected_names": ["dp", "greedy"]},
        )

    def _make_oe_bundle(self):
        return RetrievalBundle(
            problem_uid="p1",
            hint_text="hint A\nhint B\nhint C",
            retrieved_items=[
                {"source_uid": "a", "hint": "hint A"},
                {"source_uid": "b", "hint": "hint B"},
                {"source_uid": "c", "hint": "hint C"},
            ],
            metadata={},
        )

    def test_ps_select_subset(self, ctx, problem):
        """LLM selects item 1 of 2 → only dp kept."""
        provider = AsyncMock()
        provider.async_generate = AsyncMock(return_value=["1"])
        router = LlmRouter(model="test-model", domain="math")
        bundle = self._make_ps_bundle()
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert result.hint_text is not None
        assert "dp" in result.hint_text
        assert "greedy" not in result.hint_text
        assert len(result.retrieved_items) == 1
        assert result.metadata["routing_included"] is True
        assert result.metadata["routing_included_items"] == ["dp"]
        assert result.metadata["routing_excluded_items"] == ["greedy"]
        assert result.metadata["selected_names"] == ["dp"]

    def test_ps_select_all(self, ctx, problem):
        """LLM selects both items → all kept."""
        provider = AsyncMock()
        provider.async_generate = AsyncMock(return_value=["1, 2"])
        router = LlmRouter(model="test-model", domain="math")
        bundle = self._make_ps_bundle()
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert "dp" in result.hint_text
        assert "greedy" in result.hint_text
        assert len(result.retrieved_items) == 2

    def test_oe_select_subset(self, ctx, problem):
        """LLM selects items 1 and 3 of 3 → hints A and C kept."""
        provider = AsyncMock()
        provider.async_generate = AsyncMock(return_value=["1, 3"])
        router = LlmRouter(model="test-model", domain="math")
        bundle = self._make_oe_bundle()
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert "hint A" in result.hint_text
        assert "hint B" not in result.hint_text
        assert "hint C" in result.hint_text
        assert len(result.retrieved_items) == 2

    def test_none_response_drops_all(self, ctx, problem):
        """LLM says NONE → all hints dropped."""
        provider = AsyncMock()
        provider.async_generate = AsyncMock(return_value=["NONE"])
        router = LlmRouter(model="test-model", domain="math")
        bundle = self._make_ps_bundle()
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert result.hint_text is None
        assert result.retrieved_items == []
        assert result.metadata["routing_included"] is False

    def test_parse_failure_keeps_all(self, ctx, problem):
        """Unparseable response → fail-open, keep everything."""
        provider = AsyncMock()
        provider.async_generate = AsyncMock(return_value=["maybe relevant?"])
        router = LlmRouter(model="test-model", domain="arc")
        bundle = self._make_ps_bundle()
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert result.hint_text is not None
        assert result.metadata["routing_included"] is True
        assert result.metadata.get("routing_parse_failure") is True

    def test_empty_hint_passthrough(self, ctx, problem, empty_retrieval):
        provider = AsyncMock()
        router = LlmRouter(model="test-model")
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=empty_retrieval))
        assert result is empty_retrieval
        provider.async_generate.assert_not_called()

    def test_no_items_passthrough(self, ctx, problem):
        """Bundle with hint_text but no retrieved_items → unchanged."""
        bundle = RetrievalBundle(
            problem_uid="p1", hint_text="some hint",
            retrieved_items=[], metadata={},
        )
        provider = AsyncMock()
        router = LlmRouter(model="test-model")
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert result is bundle
        provider.async_generate.assert_not_called()

    def test_metadata_fields(self, ctx, problem):
        provider = AsyncMock()
        provider.async_generate = AsyncMock(return_value=["1, 2"])
        router = LlmRouter(model="m1", domain="code")
        bundle = self._make_ps_bundle()
        result = _run(router.route(ctx=ctx, provider=provider, problem=problem, retrieval=bundle))
        assert result.metadata["routing_model"] == "m1"
        assert "routing_completion" in result.metadata
        assert "routing_prompt" in result.metadata
        assert "routing_included" in result.metadata

    def test_name(self):
        assert LlmRouter.name == "llm"


# ---------------------------------------------------------------------------
# TestNliRouter — per-item filtering
# ---------------------------------------------------------------------------
class TestNliRouter:
    def _mock_cross_encoder(self, logits_per_pair):
        """Create a mock CrossEncoder returning logits for each pair."""
        mock_ce = MagicMock()
        mock_ce.predict.return_value = logits_per_pair
        return mock_ce

    # -- ps_selector-style items (concept blocks) --------------------------

    def test_ps_all_kept(self, ctx, problem):
        """All concepts above threshold → hint_text rebuilt from all blocks."""
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="- concept: dp\n  description: dynamic prog\n- concept: greedy\n  description: greedy alg",
            retrieved_items=[{"concept": "dp"}, {"concept": "greedy"}],
            metadata={"selected_names": ["dp", "greedy"]},
        )
        # Both items score high entailment
        router = NliRouter(entailment_threshold=0.5, domain="math")
        router._cross_encoder = self._mock_cross_encoder([
            [0.0, 0.0, 10.0],  # dp → high entailment
            [0.0, 0.0, 10.0],  # greedy → high entailment
        ])
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        assert result.hint_text is not None
        assert "dp" in result.hint_text
        assert "greedy" in result.hint_text
        assert result.metadata["routing_included"] is True
        assert len(result.retrieved_items) == 2

    def test_ps_partial_filter(self, ctx, problem):
        """One concept above, one below → only survivor kept."""
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="- concept: dp\n  description: dynamic prog\n- concept: greedy\n  description: greedy alg",
            retrieved_items=[{"concept": "dp"}, {"concept": "greedy"}],
            metadata={"selected_names": ["dp", "greedy"]},
        )
        router = NliRouter(entailment_threshold=0.5, domain="math")
        router._cross_encoder = self._mock_cross_encoder([
            [0.0, 0.0, 10.0],  # dp → high entailment
            [10.0, 0.0, 0.0],  # greedy → contradiction
        ])
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        assert result.hint_text is not None
        assert "dp" in result.hint_text
        assert "greedy" not in result.hint_text
        assert len(result.retrieved_items) == 1
        assert result.retrieved_items[0]["concept"] == "dp"
        assert result.metadata["routing_included"] is True
        assert "dp" in result.metadata["routing_included_items"]
        assert "greedy" in result.metadata["routing_excluded_items"]
        # selected_names updated
        assert result.metadata["selected_names"] == ["dp"]

    def test_ps_all_excluded(self, ctx, problem):
        """All concepts below threshold → hint_text nulled."""
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="- concept: dp\n  description: dynamic prog",
            retrieved_items=[{"concept": "dp"}],
            metadata={"selected_names": ["dp"]},
        )
        router = NliRouter(entailment_threshold=0.5, domain="math")
        router._cross_encoder = self._mock_cross_encoder([
            [10.0, 0.0, 0.0],  # dp → contradiction
        ])
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        assert result.hint_text is None
        assert result.retrieved_items == []
        assert result.metadata["routing_included"] is False

    def test_ps_scores_in_metadata(self, ctx, problem):
        """Per-item scores are recorded in routing_nli_scores."""
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="- concept: dp\n  description: x\n- concept: greedy\n  description: y",
            retrieved_items=[{"concept": "dp"}, {"concept": "greedy"}],
            metadata={"selected_names": ["dp", "greedy"]},
        )
        router = NliRouter(entailment_threshold=0.5, domain="math")
        router._cross_encoder = self._mock_cross_encoder([
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
        ])
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        scores = result.metadata["routing_nli_scores"]
        assert "dp" in scores
        assert "greedy" in scores
        assert scores["dp"] > 0.5
        assert scores["greedy"] < 0.5

    # -- oe_selector-style items (with hint field) -------------------------

    def test_oe_partial_filter(self, ctx, problem):
        """OE items with hint fields: filter individually."""
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="hint A\nhint B",
            retrieved_items=[
                {"source_uid": "a", "hint": "hint A", "situation": "s1"},
                {"source_uid": "b", "hint": "hint B", "situation": "s2"},
            ],
            metadata={},
        )
        router = NliRouter(entailment_threshold=0.5, domain="math")
        router._cross_encoder = self._mock_cross_encoder([
            [0.0, 0.0, 10.0],  # hint A → keep
            [10.0, 0.0, 0.0],  # hint B → drop
        ])
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        assert "hint A" in result.hint_text
        assert "hint B" not in result.hint_text
        assert len(result.retrieved_items) == 1
        assert result.retrieved_items[0]["source_uid"] == "a"

    # -- edge cases --------------------------------------------------------

    def test_empty_hint_passthrough(self, ctx, problem, empty_retrieval):
        router = NliRouter()
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=empty_retrieval))
        assert result is empty_retrieval

    def test_no_items_passthrough(self, ctx, problem):
        """Bundle with hint_text but no retrieved_items → unchanged."""
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="some hint",
            retrieved_items=[],
            metadata={},
        )
        router = NliRouter()
        result = _run(router.route(ctx=ctx, provider=None, problem=problem, retrieval=bundle))
        assert result is bundle

    def test_lazy_loading(self):
        router = NliRouter()
        assert router._cross_encoder is None

    def test_name(self):
        assert NliRouter.name == "nli"


# ---------------------------------------------------------------------------
# TestParseSelection (LLM response parsing)
# ---------------------------------------------------------------------------
class TestParseSelection:
    def test_comma_separated(self):
        assert _parse_selection("1, 3, 5", 5) == [1, 3, 5]

    def test_single_number(self):
        assert _parse_selection("2", 3) == [2]

    def test_none_response(self):
        assert _parse_selection("NONE", 3) == []

    def test_none_in_sentence(self):
        assert _parse_selection("None of these are relevant", 3) == []

    def test_out_of_range_ignored(self):
        assert _parse_selection("1, 99", 3) == [1]

    def test_all_out_of_range_is_parse_failure(self):
        assert _parse_selection("99, 100", 3) is None

    def test_no_numbers_is_parse_failure(self):
        assert _parse_selection("maybe relevant?", 3) is None

    def test_empty_is_parse_failure(self):
        assert _parse_selection("", 3) is None


# ---------------------------------------------------------------------------
# TestSplitConceptsFromHint
# ---------------------------------------------------------------------------
class TestSplitConceptsFromHint:
    def test_two_concepts(self):
        hint = "- concept: dp\n  description: dynamic prog\n- concept: greedy\n  description: greedy alg"
        blocks = split_concepts_from_hint(hint, ["dp", "greedy"])
        assert "dp" in blocks
        assert "greedy" in blocks
        assert "dynamic prog" in blocks["dp"]
        assert "greedy alg" in blocks["greedy"]
        assert "greedy" not in blocks["dp"]

    def test_missing_concept_skipped(self):
        hint = "- concept: dp\n  description: x"
        blocks = split_concepts_from_hint(hint, ["dp", "nonexistent"])
        assert "dp" in blocks
        assert "nonexistent" not in blocks

    def test_section_header_boundary(self):
        hint = "## structure\n- concept: dp\n  description: x\n## routines\n- concept: greedy\n  description: y"
        blocks = split_concepts_from_hint(hint, ["dp", "greedy"])
        assert "dp" in blocks
        assert "## routines" not in blocks["dp"]


# ---------------------------------------------------------------------------
# TestExtractItemTexts
# ---------------------------------------------------------------------------
class TestExtractItemTexts:
    def test_oe_items(self):
        bundle = RetrievalBundle(
            problem_uid="p1", hint_text="a\nb",
            retrieved_items=[
                {"source_uid": "x", "hint": "hint A"},
                {"source_uid": "y", "hint": "hint B"},
            ],
            metadata={},
        )
        result = extract_item_texts(bundle.retrieved_items, bundle)
        assert len(result) == 2
        assert result[0] == (0, "x", "hint A")
        assert result[1] == (1, "y", "hint B")

    def test_ps_items(self):
        bundle = RetrievalBundle(
            problem_uid="p1",
            hint_text="- concept: dp\n  description: x\n- concept: greedy\n  description: y",
            retrieved_items=[{"concept": "dp"}, {"concept": "greedy"}],
            metadata={},
        )
        result = extract_item_texts(bundle.retrieved_items, bundle)
        assert len(result) == 2
        assert result[0][1] == "dp"
        assert result[1][1] == "greedy"

    def test_empty_items(self):
        bundle = RetrievalBundle(
            problem_uid="p1", hint_text="x",
            retrieved_items=[], metadata={},
        )
        assert extract_item_texts(bundle.retrieved_items, bundle) == []


# ---------------------------------------------------------------------------
# TestRouterWiring
# ---------------------------------------------------------------------------
class TestRouterWiring:
    def test_router_resolved_from_config(self):
        from mem2.orchestrator.wiring import _build_component
        from mem2.registry.router import ROUTERS

        router = _build_component(ROUTERS, "none", {})
        assert router.name == "none"

    def test_all_routers_in_registry(self):
        from mem2.registry.router import ROUTERS

        assert "none" in ROUTERS
        assert "threshold" in ROUTERS
        assert "llm" in ROUTERS
        assert "nli" in ROUTERS

    def test_defaults_to_none_when_key_absent(self):
        """resolve_components works when config has no router key."""
        from mem2.orchestrator.wiring import resolve_components

        config = {
            "pipeline": {
                "task_adapter": "arc_grid",
                "benchmark": "arc_agi",
                "memory_builder": "none",
                "memory_retriever": "none",
                "trajectory_policy": "single_path",
                "provider": "mock",
                "inference_engine": "python_transform_retry",
                "feedback_engine": "gt_check",
                "evaluator": "arc_exec",
                "artifact_sink": "json_local",
                # no "router" key
            },
            "components": {
                "task_adapter": {"task_name": "arc_grid"},
                "benchmark": {"data_root": "/tmp/fake", "limit": 1},
                "inference_engine": {"model": "mock"},
                "provider": {"profile_name": "mock"},
            },
        }
        components = resolve_components(config)
        assert components.router.name == "none"

    def test_explicit_router_config(self):
        """resolve_components works with explicit router config."""
        from mem2.orchestrator.wiring import resolve_components

        config = {
            "pipeline": {
                "task_adapter": "arc_grid",
                "benchmark": "arc_agi",
                "memory_builder": "none",
                "memory_retriever": "none",
                "router": "threshold",
                "trajectory_policy": "single_path",
                "provider": "mock",
                "inference_engine": "python_transform_retry",
                "feedback_engine": "gt_check",
                "evaluator": "arc_exec",
                "artifact_sink": "json_local",
            },
            "components": {
                "task_adapter": {"task_name": "arc_grid"},
                "benchmark": {"data_root": "/tmp/fake", "limit": 1},
                "inference_engine": {"model": "mock"},
                "provider": {"profile_name": "mock"},
                "router": {"max_concept_count": 3},
            },
        }
        components = resolve_components(config)
        assert components.router.name == "threshold"
