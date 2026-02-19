"""Tests for LiveCodeBench pipeline components.

Covers: evaluator (code execution with stdin/stdout),
feedback engine (per-test-case), inference engine prompt construction,
task adapter.
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


def _lcb_problem(
    uid: str = "lcb_0",
    question: str = "Read two integers and print their sum.",
    public_tests: list[dict] | None = None,
    private_tests: list[dict] | None = None,
) -> ProblemSpec:
    if public_tests is None:
        public_tests = [
            {"input": "2 3\n", "expected_output": "5\n"},
        ]
    if private_tests is None:
        private_tests = [
            {"input": "10 20\n", "expected_output": "30\n"},
            {"input": "0 0\n", "expected_output": "0\n"},
        ]
    return ProblemSpec(
        uid=uid,
        train_pairs=[],
        test_pairs=[],
        metadata={
            "question_content": question,
            "difficulty": "easy",
            "public_test_cases": public_tests,
            "private_test_cases": private_tests,
            "starter_code": "",
        },
    )


def _attempt(uid: str, completion: str, prompt: str = "p") -> AttemptRecord:
    return AttemptRecord(
        problem_uid=uid,
        pass_idx=0,
        branch_id="lcb_solve",
        completion=completion,
        prompt=prompt,
        metadata={"initial_prompt": prompt},
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class TestLcbEvaluator:
    def test_correct_solution(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator(timeout_s=5.0)
        problem = _lcb_problem()
        code = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        attempt = _attempt("lcb_0", code)
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is True
        # All 3 tests should pass
        assert all(d["correct"] for d in records[0].test_details)

    def test_wrong_answer(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator(timeout_s=5.0)
        problem = _lcb_problem()
        code = "```python\nprint(999)\n```"
        attempt = _attempt("lcb_0", code)
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False

    def test_no_code_block(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator(timeout_s=5.0)
        problem = _lcb_problem()
        attempt = _attempt("lcb_0", "The answer is to add them.")
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert records[0].metadata.get("parsing_error") is not None

    def test_runtime_error(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator(timeout_s=5.0)
        problem = _lcb_problem()
        code = "```python\nraise ValueError('boom')\n```"
        attempt = _attempt("lcb_0", code)
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert any(d.get("error") for d in records[0].test_details)

    def test_timeout(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator(timeout_s=1.0)
        problem = _lcb_problem()
        code = "```python\nimport time\ntime.sleep(10)\nprint(5)\n```"
        attempt = _attempt("lcb_0", code)
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False
        assert any("timeout" in str(d.get("error", "")).lower() for d in records[0].test_details)

    def test_partial_pass(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator(timeout_s=5.0)
        # Create problem where only one test case matches
        problem = _lcb_problem(
            public_tests=[{"input": "2 3\n", "expected_output": "5\n"}],
            private_tests=[{"input": "10 20\n", "expected_output": "99\n"}],
        )
        code = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        attempt = _attempt("lcb_0", code)
        records = ev.evaluate(_ctx(), problem, [attempt])
        assert len(records) == 1
        assert records[0].is_correct is False  # Not all tests pass
        # But one should pass
        passed = [d for d in records[0].test_details if d.get("correct")]
        failed = [d for d in records[0].test_details if not d.get("correct")]
        assert len(passed) == 1
        assert len(failed) == 1

    def test_aggregate(self):
        from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator

        ev = LcbExecutionEvaluator()
        records = [
            EvalRecord(
                problem_uid="p1", attempt_idx=0, is_correct=True,
                train_details=[], test_details=[
                    {"correct": True, "pair_idx": 0},
                    {"correct": True, "pair_idx": 1},
                ],
            ),
            EvalRecord(
                problem_uid="p2", attempt_idx=0, is_correct=False,
                train_details=[], test_details=[
                    {"correct": True, "pair_idx": 0},
                    {"correct": False, "pair_idx": 1},
                ],
            ),
        ]
        agg = ev.aggregate(_ctx(), records)
        assert agg["total_puzzles"] == 2
        assert agg["strict_solved_puzzles"] == 1
        assert agg["solve_rate"] == 0.5
        assert agg["test_pass_rate"] == 0.75  # 3/4 tests pass


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------
class TestLcbFeedback:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_correct_feedback(self):
        from mem2.branches.feedback_engine.lcb_gt import LcbGroundTruthFeedbackEngine

        engine = LcbGroundTruthFeedbackEngine()
        problem = _lcb_problem()
        attempt = _attempt("lcb_0", "code")
        eval_rec = EvalRecord(
            problem_uid="lcb_0", attempt_idx=0, is_correct=True,
            train_details=[], test_details=[
                {"correct": True, "pair_idx": 0},
            ],
            metadata={"parsing_error": None},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        assert len(records) == 1
        assert records[0].content == "All test cases passed"
        assert records[0].metadata["is_correct"] is True

    def test_failed_tests_feedback(self):
        from mem2.branches.feedback_engine.lcb_gt import LcbGroundTruthFeedbackEngine

        engine = LcbGroundTruthFeedbackEngine()
        problem = _lcb_problem()
        attempt = _attempt("lcb_0", "code")
        eval_rec = EvalRecord(
            problem_uid="lcb_0", attempt_idx=0, is_correct=False,
            train_details=[],
            test_details=[
                {"correct": True, "pair_idx": 0},
                {"correct": False, "pair_idx": 1, "output": "999", "expected": "30"},
            ],
            metadata={"parsing_error": None},
        )
        records = self._run(engine.generate(
            ctx=_ctx(), provider=None, problem=problem,
            attempts=[attempt], eval_records=[eval_rec],
        ))
        fb = records[0]
        assert fb.metadata["is_correct"] is False
        assert len(fb.metadata["test_failures"]) == 1
        assert "999" in fb.content
        assert "30" in fb.content

    def test_parsing_error_feedback(self):
        from mem2.branches.feedback_engine.lcb_gt import LcbGroundTruthFeedbackEngine

        engine = LcbGroundTruthFeedbackEngine()
        problem = _lcb_problem()
        attempt = _attempt("lcb_0", "no code")
        eval_rec = EvalRecord(
            problem_uid="lcb_0", attempt_idx=0, is_correct=False,
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
class TestLcbInferencePrompt:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_initial_prompt_contains_problem(self):
        from mem2.branches.inference_engine.lcb_solve import LcbSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = LcbSolveInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _lcb_problem(question="Read two integers and print their sum.")
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=None, trajectory_plan=plan,
        ))
        assert len(attempts) == 1
        prompt = attempts[0].prompt
        assert "two integers" in prompt
        assert "sum" in prompt
        assert "stdin" in prompt or "input()" in prompt

    def test_initial_prompt_with_hints(self):
        from mem2.branches.inference_engine.lcb_solve import LcbSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = LcbSolveInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _lcb_problem()
        retrieval = RetrievalBundle(
            problem_uid="lcb_0",
            hint_text="Use split() to parse space-separated integers",
            retrieved_items=[],
        )
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=retrieval, trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Hints" in prompt
        assert "split()" in prompt

    def test_initial_prompt_with_test_cases(self):
        from mem2.branches.inference_engine.lcb_solve import LcbSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = LcbSolveInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _lcb_problem()
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        attempts = self._run(engine.initial_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=None, trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Example Test Cases" in prompt
        assert "2 3" in prompt

    def test_retry_prompt_includes_feedback(self):
        from mem2.branches.inference_engine.lcb_solve import LcbSolveInferenceEngine
        from mem2.providers.mock_client import MockProviderClient

        engine = LcbSolveInferenceEngine(model="mock")
        provider = MockProviderClient()
        problem = _lcb_problem()
        plan = TrajectoryPlan(num_paths=1, strategy="single")

        prev_attempt = _attempt("lcb_0", "```python\nprint(999)\n```", "initial prompt")
        prev_feedback = FeedbackRecord(
            problem_uid="lcb_0", attempt_idx=0, feedback_type="gt",
            content="Test failed",
            metadata={
                "errors": [],
                "test_failures": [{"test_idx": 0, "expected": "5", "actual": "999"}],
            },
        )

        attempts = self._run(engine.retry_attempt(
            ctx=_ctx(), provider=provider, problem=problem,
            retrieval=None, attempt_history=[prev_attempt],
            feedback_history=[prev_feedback], trajectory_plan=plan,
        ))
        prompt = attempts[0].prompt
        assert "Previous Response" in prompt


# ---------------------------------------------------------------------------
# Task adapter
# ---------------------------------------------------------------------------
class TestLcbTaskAdapter:
    def test_task_spec(self):
        from mem2.branches.task_adapter.livecodebench import LiveCodeBenchTaskAdapter

        adapter = LiveCodeBenchTaskAdapter()
        spec = adapter.get_task_spec(_ctx())
        assert spec.task_name == "livecodebench"
        assert "stdin" in spec.task_description or "code" in spec.task_description.lower()

    def test_format_problem(self):
        from mem2.branches.task_adapter.livecodebench import LiveCodeBenchTaskAdapter

        adapter = LiveCodeBenchTaskAdapter()
        problem = _lcb_problem()
        formatted = adapter.format_problem_sample(problem)
        assert formatted["uid"] == "lcb_0"
        assert formatted["difficulty"] == "easy"


# ---------------------------------------------------------------------------
# Benchmark adapter (test_cases parsing)
# ---------------------------------------------------------------------------
class TestLcbBenchmark:
    def test_parse_test_cases_dict_list(self):
        from mem2.branches.benchmark.livecodebench import _parse_test_cases

        raw = [
            {"input": "1 2\n", "expected_output": "3\n"},
            {"input": "3 4\n", "expected_output": "7\n"},
        ]
        cases = _parse_test_cases(raw)
        assert len(cases) == 2
        assert cases[0]["input"] == "1 2\n"
        assert cases[0]["expected_output"] == "3\n"

    def test_parse_test_cases_json_string(self):
        import json
        from mem2.branches.benchmark.livecodebench import _parse_test_cases

        raw = json.dumps([{"input": "5\n", "expected_output": "25\n"}])
        cases = _parse_test_cases(raw)
        assert len(cases) == 1

    def test_parse_test_cases_none(self):
        from mem2.branches.benchmark.livecodebench import _parse_test_cases

        assert _parse_test_cases(None) == []

    def test_parse_test_cases_empty(self):
        from mem2.branches.benchmark.livecodebench import _parse_test_cases

        assert _parse_test_cases("") == []
