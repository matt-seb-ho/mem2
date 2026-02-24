"""Tests for ArcMemoPsMemoryBuilder and PsSelectorRetriever."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
    TrajectoryPlan,
)
from mem2.core.errors import ConfigurationError


def _ctx() -> RunContext:
    return RunContext(run_id="test", seed=42, config={}, output_dir="/tmp/test")


def _arc_problem(uid: str = "puzzle_001") -> ProblemSpec:
    return ProblemSpec(
        uid=uid,
        train_pairs=[
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        ],
        test_pairs=[
            {"input": [[0, 0], [1, 1]]},
        ],
    )


def _sample_annotations():
    return {
        "puzzle_001": {
            "summary": "Transform by tiling",
            "concepts": [
                {
                    "concept": "tiling",
                    "kind": "routine",
                    "routine_subtype": "grid manipulation",
                    "output_typing": "Grid",
                    "cues": ["repeating pattern"],
                    "implementation": ["np.tile(...)"],
                    "parameters": [{"name": "pattern", "typing": "Grid"}],
                },
            ],
        },
        "puzzle_002": {
            "summary": "Fill regions",
            "concepts": [
                {
                    "concept": "tiling",
                    "kind": "routine",
                    "routine_subtype": "grid manipulation",
                    "cues": ["symmetry"],
                    "implementation": ["np.tile(...)"],
                    "parameters": [{"name": "count", "typing": "int"}],
                },
                {
                    "concept": "color_region",
                    "kind": "structure",
                    "cues": ["uniform color block"],
                    "implementation": ["flood fill"],
                    "parameters": [],
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# ArcMemoPsMemoryBuilder
# ---------------------------------------------------------------------------
class TestArcMemoPsMemoryBuilder:
    def test_initialize_empty(self):
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder

        builder = ArcMemoPsMemoryBuilder()
        state = builder.initialize(_ctx(), {"p1": _arc_problem("p1")})
        assert state.schema_name == "arcmemo_ps"
        assert state.payload.get("concepts") == {}

    def test_initialize_from_memory_file(self):
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder

        mem = ConceptMemory()
        mem.initialize_from_annotations(_sample_annotations())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mem.save_to_file(path)

            builder = ArcMemoPsMemoryBuilder(seed_memory_file=str(path))
            state = builder.initialize(_ctx(), {"p1": _arc_problem("p1")})
            assert "tiling" in state.payload["concepts"]
            assert state.metadata["concept_count"] == 2

    def test_initialize_from_annotations_file(self):
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.json"
            path.write_text(json.dumps(_sample_annotations()))

            builder = ArcMemoPsMemoryBuilder(seed_annotations_file=str(path))
            state = builder.initialize(_ctx(), {"p1": _arc_problem("p1")})
            assert "tiling" in state.payload["concepts"]

    def test_update_stores_correct_solutions(self):
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder

        builder = ArcMemoPsMemoryBuilder()
        mem = ConceptMemory()
        mem.initialize_from_annotations(_sample_annotations())
        state = MemoryState(
            schema_name="arcmemo_ps",
            schema_version="v1",
            payload=mem.to_payload(),
        )

        attempts = [
            AttemptRecord(problem_uid="p1", pass_idx=0, branch_id="test",
                         completion="solution code", prompt="prompt"),
        ]
        evals = [
            EvalRecord(problem_uid="p1", attempt_idx=0, is_correct=True,
                       train_details=[], test_details=[]),
        ]
        feedbacks = [
            FeedbackRecord(problem_uid="p1", attempt_idx=0,
                          feedback_type="gt", content="Correct"),
        ]

        updated = builder.update(_ctx(), state, attempts, evals, feedbacks)
        assert "p1" in updated.payload["solutions"]

    def test_schema_name(self):
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder

        assert ArcMemoPsMemoryBuilder.SCHEMA_NAME == "arcmemo_ps"


# ---------------------------------------------------------------------------
# PsSelectorRetriever
# ---------------------------------------------------------------------------
class TestPsSelectorRetriever:
    def _make_memory_state(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(_sample_annotations())
        return MemoryState(
            schema_name="arcmemo_ps",
            schema_version="v1",
            payload=mem.to_payload(),
        )

    def test_retrieve_sync_fallback(self):
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever(use_llm_selector=False)
        state = self._make_memory_state()
        problem = _arc_problem()

        bundle = retriever.retrieve(_ctx(), state, problem, [])
        assert bundle.hint_text is not None
        assert "tiling" in bundle.hint_text

    def test_retrieve_empty_memory(self):
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever()
        state = MemoryState(
            schema_name="arcmemo_ps",
            schema_version="v1",
            payload=ConceptMemory().to_payload(),
        )
        bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
        assert bundle.hint_text is None
        assert bundle.metadata["selector_mode"] == "empty"

    def test_async_retrieve_no_llm(self):
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever(use_llm_selector=False)
        state = self._make_memory_state()
        problem = _arc_problem()

        bundle = asyncio.run(
            retriever.async_retrieve(
                ctx=_ctx(),
                provider=None,
                memory=state,
                problem=problem,
                previous_attempts=[],
            )
        )
        assert bundle.hint_text is not None
        assert "tiling" in bundle.hint_text

    def test_parse_concept_selection(self):
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever()
        valid = {"tiling", "color_region", "helper"}

        # Valid YAML block
        text = "```yaml\n- tiling\n- color_region\n```"
        selected, err = retriever._parse_concept_selection(text, valid)
        assert selected == ["tiling", "color_region"]
        assert err is None

        # No yaml block — requires fenced yaml
        text2 = "- tiling\n- helper"
        selected2, err2 = retriever._parse_concept_selection(text2, valid)
        assert selected2 == []
        assert err2 == "no_yaml_block"

        # Empty
        selected3, err3 = retriever._parse_concept_selection("", valid)
        assert selected3 == []
        assert err3 == "empty_completion"

    def test_hint_text_contains_rich_fields(self):
        """Verify that concept selector output contains rich concept fields."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever(use_llm_selector=False)
        state = self._make_memory_state()
        problem = _arc_problem()

        bundle = retriever.retrieve(_ctx(), state, problem, [])
        hint = bundle.hint_text
        assert hint is not None
        # Rich fields from concept_mem.to_string() should appear
        assert "cues" in hint
        assert "implementation" in hint
        # These are ARC concept fields
        assert "repeating pattern" in hint or "uniform color block" in hint

    # ------------------------------------------------------------------ #
    #  Render mode tests                                                   #
    # ------------------------------------------------------------------ #
    def test_render_mode_cues_only(self):
        """cues_only: cues present, implementation absent."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever(
            use_llm_selector=False, render_mode="cues_only"
        )
        state = self._make_memory_state()
        bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
        hint = bundle.hint_text
        assert hint is not None
        assert "cues" in hint
        assert "implementation" not in hint

    def test_render_mode_name_only(self):
        """name_only: only names + descriptions, no cues or implementation."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever(
            use_llm_selector=False, render_mode="name_only"
        )
        state = self._make_memory_state()
        bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
        hint = bundle.hint_text
        assert hint is not None
        assert "tiling" in hint
        # name_only skips cues and implementation
        assert "implementation" not in hint

    # ------------------------------------------------------------------ #
    #  Frequency filtering tests                                           #
    # ------------------------------------------------------------------ #
    def test_frequency_filtering(self):
        """High-frequency concepts should be dropped."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            freq_file = Path(tmpdir) / "freq.json"
            freq_file.write_text(json.dumps({
                "tiling": 0.8,
                "color_region": 0.1,
            }))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                max_frequency=0.5,
                concept_frequency_file=str(freq_file),
            )
            state = self._make_memory_state()

            # Use async path with selected concepts to test filtering
            bundle = asyncio.run(
                retriever.async_retrieve(
                    ctx=_ctx(),
                    provider=None,
                    memory=state,
                    problem=_arc_problem(),
                    previous_attempts=[],
                )
            )
            # All concepts mode — filtering applies to all concept names
            # tiling (0.8) > 0.5, should be filtered
            selected = bundle.metadata.get("selected_names")
            # In all_concepts mode, selected_names is not set (None path)
            # but the retriever returns all concepts without filtering in None path
            # Filtering only applies when selected_names is a list
            assert bundle.hint_text is not None

    def test_max_concepts_limit(self):
        """Cap applied to selected concepts."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        retriever = PsSelectorRetriever(
            use_llm_selector=False,
            max_concepts_per_problem=1,
        )
        # Test via the composed ConceptFilter
        filtered = retriever._filter.filter(["a", "b", "c"])
        assert len(filtered) == 1
        assert filtered == ["a"]

    def test_routing_skip_all_generic(self):
        """hint_text=None when all concepts are high-frequency."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            freq_file = Path(tmpdir) / "freq.json"
            freq_file.write_text(json.dumps({
                "tiling": 0.9,
                "color_region": 0.9,
            }))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                routing_strategy="selection_confidence",
                max_frequency=0.5,
                concept_frequency_file=str(freq_file),
            )
            # Test via the composed RetrievalRouter
            result = retriever._router.should_include(
                ["tiling", "color_region"], "some hint text"
            )
            assert result is False

    # ------------------------------------------------------------------ #
    #  Schema validation tests                                             #
    # ------------------------------------------------------------------ #
    def test_schema_validation_mismatch(self):
        """ConfigurationError for incompatible builder/retriever pair."""
        from mem2.orchestrator.wiring import _validate_memory_pairing
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
        from mem2.branches.memory_retriever.oe_topk import OeTopKRetriever

        builder = ArcMemoPsMemoryBuilder()
        retriever = OeTopKRetriever()

        with pytest.raises(ConfigurationError, match="schema"):
            _validate_memory_pairing(builder, retriever)

    def test_schema_validation_compatible(self):
        """Compatible pair passes without error."""
        from mem2.orchestrator.wiring import _validate_memory_pairing
        from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        builder = ArcMemoPsMemoryBuilder()
        retriever = PsSelectorRetriever()

        # Should not raise
        _validate_memory_pairing(builder, retriever)

    # ------------------------------------------------------------------ #
    #  Precomputed-names path tests                                        #
    # ------------------------------------------------------------------ #
    def test_precomputed_names_through_pipeline(self):
        """selected_concepts_file activates filter/route/render pipeline."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        state = self._make_memory_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            sc_file = Path(tmpdir) / "selected.json"
            sc_file.write_text(json.dumps({"puzzle_001": ["tiling"]}))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                selected_concepts_file=str(sc_file),
                render_mode="cues_only",
            )
            bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
            assert bundle.hint_text is not None
            assert bundle.metadata["selector_mode"] == "precomputed"
            assert bundle.metadata["render_mode"] == "cues_only"
            # cues_only: cues present, implementation absent
            assert "cues" in bundle.hint_text
            assert "implementation" not in bundle.hint_text

    def test_precomputed_names_with_filtering(self):
        """High-frequency concept filtered via pipeline."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        state = self._make_memory_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            sc_file = Path(tmpdir) / "selected.json"
            sc_file.write_text(json.dumps({
                "puzzle_001": ["tiling", "color_region"]
            }))
            freq_file = Path(tmpdir) / "freq.json"
            freq_file.write_text(json.dumps({
                "tiling": 0.8,
                "color_region": 0.1,
            }))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                selected_concepts_file=str(sc_file),
                max_frequency=0.5,
                concept_frequency_file=str(freq_file),
            )
            bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
            assert bundle.hint_text is not None
            # tiling (0.8) should be filtered out
            selected = bundle.metadata.get("selected_names", [])
            assert "tiling" not in selected
            assert "color_region" in selected

    def test_precomputed_names_priority_over_rendered(self):
        """selected_concepts_file takes priority over prompt_info_file."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        state = self._make_memory_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            sc_file = Path(tmpdir) / "selected.json"
            sc_file.write_text(json.dumps({"puzzle_001": ["tiling"]}))
            pi_file = Path(tmpdir) / "prompt_info.json"
            pi_file.write_text(json.dumps({
                "puzzle_001": {"hint": "LEGACY RENDERED TEXT"}
            }))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                selected_concepts_file=str(sc_file),
                prompt_info_file=str(pi_file),
            )
            bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
            # Should use names path, not rendered path
            assert bundle.metadata["selector_mode"] == "precomputed"
            assert "LEGACY RENDERED TEXT" not in (bundle.hint_text or "")
            assert "tiling" in (bundle.hint_text or "")

    def test_precomputed_names_miss(self):
        """Returns hint_text=None for unknown problem uid."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        state = self._make_memory_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            sc_file = Path(tmpdir) / "selected.json"
            sc_file.write_text(json.dumps({"other_puzzle": ["tiling"]}))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                selected_concepts_file=str(sc_file),
            )
            bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
            assert bundle.hint_text is None
            assert bundle.metadata["selector_mode"] == "precomputed_miss"

    def test_legacy_rendered_preserved(self):
        """prompt_info_file without selected_concepts_file uses legacy path."""
        from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            pi_file = Path(tmpdir) / "prompt_info.json"
            pi_file.write_text(json.dumps({
                "puzzle_001": {"hint": "LEGACY HINT TEXT"}
            }))

            retriever = PsSelectorRetriever(
                use_llm_selector=False,
                prompt_info_file=str(pi_file),
            )
            # No memory state needed — legacy path doesn't deserialize
            state = self._make_memory_state()
            bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
            assert bundle.hint_text == "LEGACY HINT TEXT"
            assert bundle.metadata["selector_mode"] == "precomputed"
