"""Tests for retry policy — parity-critical path.

Verifies that ArcMemoRetryPolicy correctly determines whether a puzzle
needs retry based on train/test/all criterion.
"""
from mem2.core.entities import EvalRecord
from mem2.core.retry_policy import ArcMemoRetryPolicy


def _make_eval(
    train_correct: list[bool],
    test_correct: list[bool],
) -> EvalRecord:
    train = [{"correct": c, "pair_idx": i} for i, c in enumerate(train_correct)]
    test = [{"correct": c, "pair_idx": i} for i, c in enumerate(test_correct)]
    return EvalRecord(
        problem_uid="p1",
        attempt_idx=0,
        is_correct=all(test_correct),
        train_details=train,
        test_details=test,
    )


class TestRetryPolicyCriterion:
    def test_train_criterion_all_train_pass(self):
        policy = ArcMemoRetryPolicy(criterion="train")
        rec = _make_eval([True, True], [False])
        assert not policy.needs_retry(rec)

    def test_train_criterion_train_fail(self):
        policy = ArcMemoRetryPolicy(criterion="train")
        rec = _make_eval([True, False], [True])
        assert policy.needs_retry(rec)

    def test_test_criterion_all_test_pass(self):
        policy = ArcMemoRetryPolicy(criterion="test")
        rec = _make_eval([False], [True, True])
        assert not policy.needs_retry(rec)

    def test_test_criterion_test_fail(self):
        policy = ArcMemoRetryPolicy(criterion="test")
        rec = _make_eval([True], [True, False])
        assert policy.needs_retry(rec)

    def test_all_criterion_everything_pass(self):
        policy = ArcMemoRetryPolicy(criterion="all")
        rec = _make_eval([True], [True])
        assert not policy.needs_retry(rec)

    def test_all_criterion_train_fail(self):
        policy = ArcMemoRetryPolicy(criterion="all")
        rec = _make_eval([False], [True])
        assert policy.needs_retry(rec)

    def test_empty_details_needs_retry(self):
        policy = ArcMemoRetryPolicy(criterion="train")
        rec = _make_eval([], [])
        assert policy.needs_retry(rec)


class TestRetryPolicyValidation:
    def test_valid_defaults(self):
        policy = ArcMemoRetryPolicy()
        ok, err = policy.is_valid()
        assert ok
        assert err is None

    def test_invalid_max_passes(self):
        policy = ArcMemoRetryPolicy(max_passes=0)
        ok, err = policy.is_valid()
        assert not ok

    def test_invalid_criterion(self):
        policy = ArcMemoRetryPolicy(criterion="invalid")
        ok, err = policy.is_valid()
        assert not ok

    def test_invalid_error_feedback(self):
        policy = ArcMemoRetryPolicy(error_feedback="invalid")
        ok, err = policy.is_valid()
        assert not ok


class TestRetryPolicyFromConfig:
    def test_from_config_defaults(self):
        policy = ArcMemoRetryPolicy.from_config({})
        assert policy.max_passes == 3
        assert policy.criterion == "train"

    def test_from_config_override(self):
        cfg = {
            "run": {
                "retry_policy": {
                    "max_passes": 5,
                    "criterion": "test",
                    "error_feedback": "first",
                }
            }
        }
        policy = ArcMemoRetryPolicy.from_config(cfg)
        assert policy.max_passes == 5
        assert policy.criterion == "test"
        assert policy.error_feedback == "first"
