"""Tests for feedback engine — parity-critical path.

Verifies that GroundTruthFeedbackEngine produces correct feedback records
with properly structured metadata for downstream retry prompt construction.
"""
import asyncio

from mem2.branches.feedback_engine.gt_check import GroundTruthFeedbackEngine
from mem2.core.entities import AttemptRecord, EvalRecord, ProblemSpec, RunContext


def _make_ctx() -> RunContext:
    return RunContext(run_id="test", seed=42, config={}, output_dir="/tmp/test")


def _make_problem() -> ProblemSpec:
    return ProblemSpec(
        uid="p1",
        train_pairs=[
            {"input": [[0]], "output": [[1]]},
            {"input": [[2]], "output": [[3]]},
        ],
        test_pairs=[{"input": [[4]], "output": [[5]]}],
    )


def _make_attempt(uid: str = "p1") -> AttemptRecord:
    return AttemptRecord(
        problem_uid=uid,
        pass_idx=0,
        branch_id="test",
        completion="```python\ndef transform(x): return x\n```",
        prompt="test prompt",
    )


def _make_eval_correct(uid: str = "p1") -> EvalRecord:
    return EvalRecord(
        problem_uid=uid,
        attempt_idx=0,
        is_correct=True,
        train_details=[
            {"is_train": True, "pair_idx": 0, "correct": True},
            {"is_train": True, "pair_idx": 1, "correct": True},
        ],
        test_details=[{"is_train": False, "pair_idx": 0, "correct": True}],
        metadata={"parsing_error": None},
    )


def _make_eval_with_train_failures(uid: str = "p1") -> EvalRecord:
    return EvalRecord(
        problem_uid=uid,
        attempt_idx=0,
        is_correct=False,
        train_details=[
            {"is_train": True, "pair_idx": 0, "correct": True, "output": [[1]], "expected": [[1]]},
            {"is_train": True, "pair_idx": 1, "correct": False, "output": [[0]], "expected": [[3]]},
        ],
        test_details=[{"is_train": False, "pair_idx": 0, "correct": False}],
        metadata={"parsing_error": None},
    )


def _make_eval_with_parsing_error(uid: str = "p1") -> EvalRecord:
    return EvalRecord(
        problem_uid=uid,
        attempt_idx=0,
        is_correct=False,
        train_details=[],
        test_details=[],
        metadata={"parsing_error": "no python code block found."},
    )


class TestGroundTruthFeedback:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_correct_attempt_gives_positive(self):
        engine = GroundTruthFeedbackEngine()
        records = self._run(engine.generate(
            ctx=_make_ctx(),
            provider=None,
            problem=_make_problem(),
            attempts=[_make_attempt()],
            eval_records=[_make_eval_correct()],
        ))
        assert len(records) == 1
        assert records[0].content == "Correct"
        assert records[0].metadata["is_correct"] is True

    def test_train_failure_produces_mismatches(self):
        engine = GroundTruthFeedbackEngine()
        records = self._run(engine.generate(
            ctx=_make_ctx(),
            provider=None,
            problem=_make_problem(),
            attempts=[_make_attempt()],
            eval_records=[_make_eval_with_train_failures()],
        ))
        assert len(records) == 1
        fb = records[0]
        assert fb.metadata["is_correct"] is False
        assert len(fb.metadata["mismatches"]) == 1
        assert fb.metadata["mismatches"][0]["example_idx"] == 2  # pair_idx=1 → example_idx=2

    def test_parsing_error_produces_error(self):
        engine = GroundTruthFeedbackEngine()
        records = self._run(engine.generate(
            ctx=_make_ctx(),
            provider=None,
            problem=_make_problem(),
            attempts=[_make_attempt()],
            eval_records=[_make_eval_with_parsing_error()],
        ))
        assert len(records) == 1
        fb = records[0]
        assert fb.metadata["is_correct"] is False
        assert len(fb.metadata["errors"]) == 1
        assert "no python code block" in fb.metadata["errors"][0]

    def test_metadata_structure_for_retry_prompt(self):
        """The retry prompt builder reads fb.metadata['errors'] and fb.metadata['mismatches'].
        This test ensures those keys always exist."""
        engine = GroundTruthFeedbackEngine()
        records = self._run(engine.generate(
            ctx=_make_ctx(),
            provider=None,
            problem=_make_problem(),
            attempts=[_make_attempt()],
            eval_records=[_make_eval_with_train_failures()],
        ))
        fb = records[0]
        assert "errors" in fb.metadata
        assert "mismatches" in fb.metadata
        assert isinstance(fb.metadata["errors"], list)
        assert isinstance(fb.metadata["mismatches"], list)
