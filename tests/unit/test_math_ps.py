"""Tests for math-PS vertical slice components.

Covers: benchmark loading, evaluator (code execution + answer matching),
feedback engine, inference engine prompt construction.
"""
import asyncio

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
        branch_id="math_ps_solve",
        completion=completion,
        prompt=prompt,
        metadata={"initial_prompt": prompt},
    )


# ---------------------------------------------------------------------------
# Benchmark adapter
# ---------------------------------------------------------------------------
class TestCompetitionMathPsBenchmark:
    def test_loads_problems(self):
        from mem2.branches.benchmark.competition_math_ps import CompetitionMathPsBenchmarkAdapter

        adapter = CompetitionMathPsBenchmarkAdapter(
            data_root="/root/workspace/data/hf/qwedsacf__competition_math",
            types=["Number Theory"],
            limit=10,
        )
        problems = adapter.load(_ctx())
        assert len(problems) == 10
        for uid, p in problems.items():
            assert p.metadata["answer_int"] is not None
            assert isinstance(p.metadata["answer_int"], int)
            assert p.metadata["problem_text"]

    def test_validates_nonempty(self):
        from mem2.branches.benchmark.competition_math_ps import CompetitionMathPsBenchmarkAdapter
        from mem2.core.errors import DataValidationError

        adapter = CompetitionMathPsBenchmarkAdapter(
            data_root="/root/workspace/data/hf/qwedsacf__competition_math",
            types=["Number Theory"],
            limit=3,
        )
        problems = adapter.load(_ctx())
        adapter.validate(problems)  # should not raise

        import pytest
        with pytest.raises(DataValidationError):
            adapter.validate({})

    def test_filters_by_type(self):
        from mem2.branches.benchmark.competition_math_ps import CompetitionMathPsBenchmarkAdapter

        adapter = CompetitionMathPsBenchmarkAdapter(
            data_root="/root/workspace/data/hf/qwedsacf__competition_math",
            types=["Counting & Probability"],
            limit=5,
        )
        problems = adapter.load(_ctx())
        for p in problems.values():
            assert p.metadata["math_type"] == "Counting & Probability"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class TestMathPsEvaluator:
    def test_correct_answer(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator(timeout_s=5.0)
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "```python\ndef solve():\n    return 2 + 3\n```")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is True

    def test_wrong_answer(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator(timeout_s=5.0)
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "```python\ndef solve():\n    return 99\n```")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert records[0].test_details[0]["output"] == 99
        assert records[0].test_details[0]["expected"] == 5

    def test_no_code_block(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator(timeout_s=5.0)
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "The answer is 5.")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert records[0].metadata.get("parsing_error") is not None

    def test_runtime_error(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator(timeout_s=5.0)
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "```python\ndef solve():\n    return 1/0\n```")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert records[0].test_details[0].get("error") is not None

    def test_no_solve_function(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator(timeout_s=5.0)
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "```python\ndef compute():\n    return 5\n```")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False

    def test_timeout(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator(timeout_s=1.0)
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "```python\nimport time\ndef solve():\n    time.sleep(10)\n    return 5\n```")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert "timeout" in records[0].test_details[0].get("error", "").lower()

    def test_aggregate(self):
        from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

        ev = MathPsExecutionEvaluator()
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
        assert agg["strict_solved_puzzles"] == 2  # both solved (p2 on retry)
        assert agg["solve_rate"] == 1.0


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------
class TestMathPsFeedback:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_correct_feedback(self):
        from mem2.branches.feedback_engine.math_ps_gt import MathPsGroundTruthFeedbackEngine

        engine = MathPsGroundTruthFeedbackEngine()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "code")
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

    def test_wrong_answer_feedback(self):
        from mem2.branches.feedback_engine.math_ps_gt import MathPsGroundTruthFeedbackEngine

        engine = MathPsGroundTruthFeedbackEngine()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "code")
        eval_rec = EvalRecord(
            problem_uid="cmath_0", attempt_idx=0, is_correct=False,
            train_details=[],
            test_details=[{"correct": False, "output": 99, "expected": 5}],
            metadata={"parsing_error": None},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        fb = records[0]
        assert fb.metadata["is_correct"] is False
        assert len(fb.metadata["mismatches"]) == 1
        assert "99" in fb.content
        assert "5" in fb.content

    def test_parsing_error_feedback(self):
        from mem2.branches.feedback_engine.math_ps_gt import MathPsGroundTruthFeedbackEngine

        engine = MathPsGroundTruthFeedbackEngine()
        problem = _math_problem(answer=5)
        attempt = _attempt("cmath_0", "no code")
        eval_rec = EvalRecord(
            problem_uid="cmath_0", attempt_idx=0, is_correct=False,
            train_details=[], test_details=[],
            metadata={"parsing_error": "no python code block found."},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        fb = records[0]
        assert "no python code block" in fb.metadata["errors"][0]


# ---------------------------------------------------------------------------
# Inference engine prompt construction
# ---------------------------------------------------------------------------
class TestMathPsInferencePrompt:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_initial_prompt_contains_problem(self):
        from mem2.branches.inference_engine.math_ps_solve import MathPsSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathPsSolveInferenceEngine(model="mock")
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
        assert "def solve()" in prompt
        assert "```python" in prompt.lower() or "python" in prompt.lower()

    def test_initial_prompt_with_hints(self):
        from mem2.branches.inference_engine.math_ps_solve import MathPsSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathPsSolveInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _math_problem(text="Find 7 mod 3")
        retrieval = RetrievalBundle(
            problem_uid="cmath_0",
            hint_text="Use the modulo operator for remainder problems",
            retrieved_items=[],
        )
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=retrieval, trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Hints" in prompt
        assert "modulo operator" in prompt

    def test_retry_prompt_includes_feedback(self):
        from mem2.branches.inference_engine.math_ps_solve import MathPsSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = MathPsSolveInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _math_problem(text="What is 2+3?", answer=5)
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        prev_attempt = _attempt("cmath_0", "```python\ndef solve():\n    return 99\n```", "initial prompt")
        prev_feedback = FeedbackRecord(
            problem_uid="cmath_0", attempt_idx=0, feedback_type="gt",
            content="Wrong answer",
            metadata={"errors": [], "mismatches": [{"output": 99, "expected": 5}]},
        )

        attempts = self._run(engine.retry_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=None, attempt_history=[prev_attempt],
            feedback_history=[prev_feedback], trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Previous Response" in prompt
        assert "99" in prompt
        assert "5" in prompt
