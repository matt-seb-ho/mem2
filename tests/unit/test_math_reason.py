"""Tests for math reasoning pipeline components.

Covers: evaluator (boxed answer parsing), feedback engine,
inference engine prompt construction.
"""
import asyncio

from mem2.core.entities import (
    AttemptRecord,
    EvalRecord,
    FeedbackRecord,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
    TrajectoryPlan,
)


def _ctx() -> RunContext:
    return RunContext(run_id="test", seed=42, config={}, output_dir="/tmp/test")


def _math_problem(uid: str = "cmath_0", text: str = "What is 2+3?", answer: int = 5) -> ProblemSpec:
    return ProblemSpec(
        uid=uid,
        train_pairs=[],
        test_pairs=[],
        metadata={
            "problem_text": text,
            "answer_str": str(answer),
            "answer_int": answer,
            "math_type": "Number Theory",
            "level": "Level 1",
        },
    )


def _attempt(uid: str, completion: str, prompt: str = "p") -> AttemptRecord:
    return AttemptRecord(
        problem_uid=uid,
        pass_idx=0,
        branch_id="math_reason",
        completion=completion,
        prompt=prompt,
        metadata={"initial_prompt": prompt},
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class TestMathReasonEvaluator:
    def test_correct_boxed_answer(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=5)
        attempt = _attempt(
            "cmath_0",
            "We have 2+3=5. Therefore the answer is \\boxed{5}."
        )
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is True

    def test_wrong_boxed_answer(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=5)
        attempt = _attempt(
            "cmath_0",
            "I think the answer is \\boxed{99}."
        )
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert records[0].test_details[0]["output"] == 99
        assert records[0].test_details[0]["expected"] == 5

    def test_no_boxed_answer(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "The answer is 5.")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert "no \\boxed{}" in records[0].metadata.get("parsing_error", "")

    def test_non_integer_boxed_answer(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "The answer is \\boxed{five}.")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert "non-integer" in records[0].metadata.get("parsing_error", "")

    def test_multiple_boxed_takes_last(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=42)
        attempt = _attempt(
            "cmath_0",
            "First we get \\boxed{7}, then \\boxed{6}, so the final answer is \\boxed{42}."
        )
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert records[0].is_correct is True

    def test_negative_answer(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=-3)
        attempt = _attempt("cmath_0", "The answer is \\boxed{-3}.")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert records[0].is_correct is True

    def test_comma_separated_number(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        problem = _math_problem(answer=1000)
        attempt = _attempt("cmath_0", "The answer is \\boxed{1,000}.")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert records[0].is_correct is True

    def test_aggregate(self):
        from mem2.branches.evaluator.math_reason_eval import MathReasonEvaluator

        ev = MathReasonEvaluator()
        records = [
            EvalRecord(problem_uid="p1", attempt_idx=0, is_correct=True,
                       train_details=[], test_details=[]),
            EvalRecord(problem_uid="p2", attempt_idx=0, is_correct=False,
                       train_details=[], test_details=[]),
            EvalRecord(problem_uid="p2", attempt_idx=1, is_correct=True,
                       train_details=[], test_details=[]),
        ]
        agg = ev.aggregate(_ctx(), records)
        assert agg["total_puzzles"] == 2
        assert agg["strict_solved_puzzles"] == 2
        assert agg["solve_rate"] == 1.0


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------
class TestMathReasonFeedback:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_correct_feedback(self):
        from mem2.branches.feedback_engine.math_reason_gt import MathReasonGroundTruthFeedbackEngine

        engine = MathReasonGroundTruthFeedbackEngine()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "reasoning")
        eval_rec = EvalRecord(
            problem_uid="cmath_0", attempt_idx=0, is_correct=True,
            train_details=[], test_details=[],
            metadata={"parsing_error": None},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        assert len(records) == 1
        assert records[0].content == "Correct"
        assert records[0].metadata["is_correct"] is True

    def test_wrong_answer_feedback_no_leak(self):
        from mem2.branches.feedback_engine.math_reason_gt import MathReasonGroundTruthFeedbackEngine

        engine = MathReasonGroundTruthFeedbackEngine()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "reasoning")
        eval_rec = EvalRecord(
            problem_uid="cmath_0", attempt_idx=0, is_correct=False,
            train_details=[], test_details=[],
            metadata={"parsing_error": None},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        fb = records[0]
        assert fb.metadata["is_correct"] is False
        assert "Incorrect" in fb.content
        # Must NOT leak the expected answer
        assert "5" not in fb.content

    def test_no_boxed_feedback(self):
        from mem2.branches.feedback_engine.math_reason_gt import MathReasonGroundTruthFeedbackEngine

        engine = MathReasonGroundTruthFeedbackEngine()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "no boxed answer")
        eval_rec = EvalRecord(
            problem_uid="cmath_0", attempt_idx=0, is_correct=False,
            train_details=[], test_details=[],
            metadata={"parsing_error": "no \\boxed{} answer found"},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        fb = records[0]
        assert "Parsing Issue" in fb.content
        assert "\\boxed{}" in fb.content


# ---------------------------------------------------------------------------
# Inference engine prompt construction
# ---------------------------------------------------------------------------
class TestMathReasonInferencePrompt:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_initial_prompt_contains_problem(self):
        from mem2.branches.inference_engine.math_reason import MathReasonInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathReasonInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _math_problem(text="What is the remainder when 2003 is divided by 11?")
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=None, trajectory_plan=plan,
        ))
        assert len(attempts) == 1
        prompt = attempts[0].prompt
        assert "2003" in prompt
        assert "divided by 11" in prompt
        assert "\\boxed{" in prompt
        # Should NOT mention solve() or Python code
        assert "def solve()" not in prompt
        assert "```python" not in prompt.lower()

    def test_initial_prompt_with_hints(self):
        from mem2.branches.inference_engine.math_reason import MathReasonInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathReasonInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _math_problem(text="Find 7 mod 3")
        retrieval = RetrievalBundle(
            problem_uid="cmath_0",
            hint_text="Use modular arithmetic properties",
            retrieved_items=[],
        )
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=retrieval, trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Hints" in prompt
        assert "modular arithmetic" in prompt

    def test_retry_prompt_includes_feedback(self):
        from mem2.branches.inference_engine.math_reason import MathReasonInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathReasonInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _math_problem(text="What is 2+3?", answer=5)
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        prev_attempt = _attempt(
            "cmath_0",
            "I think 2+3=7, so \\boxed{7}.",
            "initial prompt"
        )
        prev_feedback = FeedbackRecord(
            problem_uid="cmath_0", attempt_idx=0, feedback_type="gt",
            content="Incorrect",
            metadata={"errors": []},
        )

        attempts = self._run(engine.retry_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=None, attempt_history=[prev_attempt],
            feedback_history=[prev_feedback], trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Previous Response" in prompt
        assert "\\boxed{" in prompt

    def test_no_hints_when_disabled(self):
        from mem2.branches.inference_engine.math_reason import MathReasonInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathReasonInferenceEngine(model="mock", include_initial_hints=False)
        provider = MockProviderClient()
        problem = _math_problem(text="Find 7 mod 3")
        retrieval = RetrievalBundle(
            problem_uid="cmath_0",
            hint_text="Use modular arithmetic properties",
            retrieved_items=[],
        )
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=retrieval, trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Hints" not in prompt


# ---------------------------------------------------------------------------
# Answer parsing utilities
# ---------------------------------------------------------------------------
class TestAnswerParsing:
    def test_extract_boxed_simple(self):
        from mem2.branches.evaluator.math_reason_eval import extract_boxed_answer
        assert extract_boxed_answer("The answer is \\boxed{42}.") == "42"

    def test_extract_boxed_nested_braces(self):
        from mem2.branches.evaluator.math_reason_eval import extract_boxed_answer
        assert extract_boxed_answer("\\boxed{2^{10}}") == "2^{10}"

    def test_extract_boxed_none(self):
        from mem2.branches.evaluator.math_reason_eval import extract_boxed_answer
        assert extract_boxed_answer("No boxed answer here") is None

    def test_extract_boxed_last(self):
        from mem2.branches.evaluator.math_reason_eval import extract_boxed_answer
        assert extract_boxed_answer("\\boxed{1} and \\boxed{2}") == "2"

    def test_parse_integer_simple(self):
        from mem2.branches.evaluator.math_reason_eval import parse_integer_answer
        assert parse_integer_answer("42") == 42

    def test_parse_integer_negative(self):
        from mem2.branches.evaluator.math_reason_eval import parse_integer_answer
        assert parse_integer_answer("-7") == -7

    def test_parse_integer_comma(self):
        from mem2.branches.evaluator.math_reason_eval import parse_integer_answer
        assert parse_integer_answer("1,000") == 1000

    def test_parse_integer_non_numeric(self):
        from mem2.branches.evaluator.math_reason_eval import parse_integer_answer
        assert parse_integer_answer("five") is None

    def test_parse_integer_fraction(self):
        from mem2.branches.evaluator.math_reason_eval import parse_integer_answer
        assert parse_integer_answer("3/4") is None
