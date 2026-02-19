"""Tests for ConceptPsMemoryBuilder and ConceptSelectorRetriever."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

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
# ConceptPsMemoryBuilder
# ---------------------------------------------------------------------------
class TestConceptPsMemoryBuilder:
    def test_initialize_empty(self):
        from mem2.branches.memory_builder.concept_ps import ConceptPsMemoryBuilder

        builder = ConceptPsMemoryBuilder()
        state = builder.initialize(_ctx(), {"p1": _arc_problem("p1")})
        assert state.schema_name == "concept_ps"
        assert state.payload.get("concepts") == {}

    def test_initialize_from_memory_file(self):
        from mem2.branches.memory_builder.concept_ps import ConceptPsMemoryBuilder

        mem = ConceptMemory()
        mem.initialize_from_annotations(_sample_annotations())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mem.save_to_file(path)

            builder = ConceptPsMemoryBuilder(seed_memory_file=str(path))
            state = builder.initialize(_ctx(), {"p1": _arc_problem("p1")})
            assert "tiling" in state.payload["concepts"]
            assert state.metadata["concept_count"] == 2

    def test_initialize_from_annotations_file(self):
        from mem2.branches.memory_builder.concept_ps import ConceptPsMemoryBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.json"
            path.write_text(json.dumps(_sample_annotations()))

            builder = ConceptPsMemoryBuilder(seed_annotations_file=str(path))
            state = builder.initialize(_ctx(), {"p1": _arc_problem("p1")})
            assert "tiling" in state.payload["concepts"]

    def test_update_stores_correct_solutions(self):
        from mem2.branches.memory_builder.concept_ps import ConceptPsMemoryBuilder

        builder = ConceptPsMemoryBuilder()
        mem = ConceptMemory()
        mem.initialize_from_annotations(_sample_annotations())
        state = MemoryState(
            schema_name="concept_ps",
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

    def test_reflect(self):
        from mem2.branches.memory_builder.concept_ps import ConceptPsMemoryBuilder

        builder = ConceptPsMemoryBuilder()
        problem = _arc_problem("p1")
        attempts = [
            AttemptRecord(problem_uid="p1", pass_idx=0, branch_id="test",
                         completion="code here", prompt="prompt"),
        ]
        feedbacks = [
            FeedbackRecord(problem_uid="p1", attempt_idx=0,
                          feedback_type="gt", content="Wrong"),
        ]
        items = builder.reflect(_ctx(), problem, attempts, feedbacks)
        assert len(items) == 1
        assert items[0]["problem_uid"] == "p1"


# ---------------------------------------------------------------------------
# ConceptSelectorRetriever
# ---------------------------------------------------------------------------
class TestConceptSelectorRetriever:
    def _make_memory_state(self):
        mem = ConceptMemory()
        mem.initialize_from_annotations(_sample_annotations())
        return MemoryState(
            schema_name="concept_ps",
            schema_version="v1",
            payload=mem.to_payload(),
        )

    def test_retrieve_sync_fallback(self):
        from mem2.branches.memory_retriever.concept_selector import ConceptSelectorRetriever

        retriever = ConceptSelectorRetriever(use_llm_selector=False)
        state = self._make_memory_state()
        problem = _arc_problem()

        bundle = retriever.retrieve(_ctx(), state, problem, [])
        assert bundle.hint_text is not None
        assert "tiling" in bundle.hint_text
        assert "Concepts from Previously Solved" in bundle.hint_text

    def test_retrieve_empty_memory(self):
        from mem2.branches.memory_retriever.concept_selector import ConceptSelectorRetriever

        retriever = ConceptSelectorRetriever()
        state = MemoryState(
            schema_name="concept_ps",
            schema_version="v1",
            payload=ConceptMemory().to_payload(),
        )
        bundle = retriever.retrieve(_ctx(), state, _arc_problem(), [])
        assert bundle.hint_text is None
        assert bundle.metadata["selector_mode"] == "empty"

    def test_async_retrieve_no_llm(self):
        from mem2.branches.memory_retriever.concept_selector import ConceptSelectorRetriever

        retriever = ConceptSelectorRetriever(use_llm_selector=False)
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
        from mem2.branches.memory_retriever.concept_selector import ConceptSelectorRetriever

        retriever = ConceptSelectorRetriever()
        valid = {"tiling", "color_region", "helper"}

        # Valid YAML block
        text = "```yaml\n- tiling\n- color_region\n```"
        selected, err = retriever._parse_concept_selection(text, valid)
        assert selected == ["tiling", "color_region"]
        assert err is None

        # No yaml block but parseable
        text2 = "- tiling\n- helper"
        selected2, err2 = retriever._parse_concept_selection(text2, valid)
        assert "tiling" in selected2

        # Empty
        selected3, err3 = retriever._parse_concept_selection("", valid)
        assert selected3 == []
        assert err3 == "empty_completion"

    def test_hint_text_contains_rich_fields(self):
        """Verify that concept selector output contains rich concept fields."""
        from mem2.branches.memory_retriever.concept_selector import ConceptSelectorRetriever

        retriever = ConceptSelectorRetriever(use_llm_selector=False)
        state = self._make_memory_state()
        problem = _arc_problem()

        bundle = retriever.retrieve(_ctx(), state, problem, [])
        hint = bundle.hint_text
        # Rich fields from concept_mem.to_string() should appear
        assert "cues" in hint
        assert "implementation" in hint
        # These are ARC concept fields
        assert "repeating pattern" in hint or "uniform color block" in hint
