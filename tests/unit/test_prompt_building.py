"""Tests for prompt construction — parity-critical path.

Verifies that initial and retry prompts have the correct structure
and content, since any deviation cascades into different model behavior.
"""
import numpy as np

from mem2.core.entities import AttemptRecord, FeedbackRecord, ProblemSpec, RetrievalBundle
from mem2.prompting.options import ArcMemoPromptOptions
from mem2.prompting.render import (
    format_grid_numpy,
    make_initial_prompt,
    make_retry_prompt,
    prompt_fingerprint,
)


def _make_problem(uid: str = "test_001") -> ProblemSpec:
    return ProblemSpec(
        uid=uid,
        train_pairs=[
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
        ],
        test_pairs=[
            {"input": [[4, 5], [5, 4]], "output": [[5, 4], [4, 5]]},
        ],
    )


def _make_attempt(uid: str, completion: str, prompt: str = "prompt") -> AttemptRecord:
    return AttemptRecord(
        problem_uid=uid,
        pass_idx=0,
        branch_id="test",
        completion=completion,
        prompt=prompt,
        metadata={"initial_prompt": prompt},
    )


def _make_feedback(uid: str, errors: list[str], mismatches: list[dict]) -> FeedbackRecord:
    return FeedbackRecord(
        problem_uid=uid,
        attempt_idx=0,
        feedback_type="gt",
        content="Incorrect",
        metadata={"errors": errors, "mismatches": mismatches},
    )


class TestFormatGrid:
    def test_basic_grid(self):
        grid = [[0, 1], [2, 3]]
        result = format_grid_numpy(grid)
        assert "0" in result
        assert "3" in result

    def test_numpy_array(self):
        arr = np.array([[0, 1], [2, 3]])
        result = format_grid_numpy(arr)
        assert "0" in result


class TestInitialPrompt:
    def test_contains_required_sections(self):
        problem = _make_problem()
        prompt = make_initial_prompt(problem, retrieval=None)
        assert "### Introduction" in prompt
        assert "### Your Puzzle Grids" in prompt
        assert "### Instructions" in prompt
        assert "def transform" in prompt

    def test_no_hint_by_default(self):
        problem = _make_problem()
        opts = ArcMemoPromptOptions(include_hint=False)
        prompt = make_initial_prompt(problem, retrieval=None, options=opts)
        assert "### Hints" not in prompt

    def test_with_hint(self):
        problem = _make_problem()
        retrieval = RetrievalBundle(
            problem_uid="test_001",
            hint_text="Try flipping the grid diagonally",
            retrieved_items=[],
        )
        opts = ArcMemoPromptOptions(include_hint=True)
        prompt = make_initial_prompt(problem, retrieval=retrieval, options=opts)
        assert "### Hints" in prompt
        assert "flipping the grid" in prompt

    def test_fingerprint_deterministic(self):
        problem = _make_problem()
        prompt = make_initial_prompt(problem, retrieval=None)
        fp1 = prompt_fingerprint(prompt)
        fp2 = prompt_fingerprint(prompt)
        assert fp1 == fp2
        assert len(fp1) == 64  # sha256 hex

    def test_train_pairs_in_prompt(self):
        problem = _make_problem()
        prompt = make_initial_prompt(problem, retrieval=None)
        assert "Example 1" in prompt
        assert "Example 2" in prompt

    def test_test_input_in_prompt(self):
        problem = _make_problem()
        prompt = make_initial_prompt(problem, retrieval=None)
        assert "test example" in prompt.lower()


class TestRetryPrompt:
    def test_contains_previous_attempts_section(self):
        initial_prompt = "Initial prompt content"
        attempt = _make_attempt("p1", "```python\ndef transform(x): return x\n```", initial_prompt)
        feedback = _make_feedback("p1", errors=[], mismatches=[])
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
        )
        assert "### Your Previous Response(s) and Outcomes" in prompt
        assert "### New Instructions" in prompt
        assert initial_prompt in prompt

    def test_includes_error_feedback(self):
        initial_prompt = "Initial prompt"
        attempt = _make_attempt("p1", "bad code")
        feedback = _make_feedback("p1", errors=["SyntaxError: invalid syntax"], mismatches=[])
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
        )
        assert "SyntaxError" in prompt

    def test_includes_mismatch_feedback(self):
        initial_prompt = "Initial prompt"
        attempt = _make_attempt("p1", "```python\ndef transform(x): return x\n```")
        feedback = _make_feedback(
            "p1",
            errors=[],
            mismatches=[{"example_idx": 1, "output": [[0, 0], [0, 0]]}],
        )
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
        )
        assert "Output Mismatches" in prompt
        assert "Example 1" in prompt

    def test_new_concepts_appended(self):
        initial_prompt = "Initial prompt"
        attempt = _make_attempt("p1", "code")
        feedback = _make_feedback("p1", errors=[], mismatches=[])
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
            new_concepts="Try a different approach",
        )
        assert "### Reselected Lessons" in prompt
        assert "different approach" in prompt

    def test_no_concepts_without_flag(self):
        initial_prompt = "Initial prompt"
        attempt = _make_attempt("p1", "code")
        feedback = _make_feedback("p1", errors=[], mismatches=[])
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
            new_concepts=None,
        )
        assert "### Reselected Lessons" not in prompt

    def test_num_feedback_passes_limits_history(self):
        initial_prompt = "Initial prompt"
        attempts = [
            _make_attempt("p1", f"attempt_{i}", initial_prompt)
            for i in range(3)
        ]
        feedbacks = [
            _make_feedback("p1", errors=[f"error_{i}"], mismatches=[])
            for i in range(3)
        ]
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=attempts,
            feedback=feedbacks,
            num_feedback_passes=1,
        )
        # Only the last attempt should be included
        assert "#### Attempt 3" in prompt
        assert "#### Attempt 1" not in prompt


class TestRetryPolicy:
    def test_error_feedback_first_limits(self):
        initial_prompt = "Initial prompt"
        attempt = _make_attempt("p1", "code")
        feedback = _make_feedback(
            "p1",
            errors=["error_1", "error_2", "error_3"],
            mismatches=[],
        )
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
            error_feedback="first",
        )
        assert "error_1" in prompt
        assert "error_2" not in prompt

    def test_error_feedback_all_includes_all(self):
        initial_prompt = "Initial prompt"
        attempt = _make_attempt("p1", "code")
        feedback = _make_feedback(
            "p1",
            errors=["error_1", "error_2"],
            mismatches=[],
        )
        prompt = make_retry_prompt(
            initial_prompt=initial_prompt,
            attempts=[attempt],
            feedback=[feedback],
            error_feedback="all",
        )
        assert "error_1" in prompt
        assert "error_2" in prompt
