"""Tests for scoring logic — parity-critical path.

Verifies official and strict score calculations match expected semantics.
"""
from mem2.analysis.score_parity import (
    flatten_eval_records,
    official_score_sum,
    strict_score_sum,
)
from mem2.core.entities import EvalRecord


def _make_eval(
    uid: str,
    attempt_idx: int,
    train_correct: list[bool],
    test_correct: list[bool],
    step_idx: int | None = None,
) -> EvalRecord:
    train = [
        {"is_train": True, "pair_idx": i, "correct": c}
        for i, c in enumerate(train_correct)
    ]
    test = [
        {"is_train": False, "pair_idx": i, "correct": c}
        for i, c in enumerate(test_correct)
    ]
    is_correct = all(test_correct) if test_correct else False
    md = {}
    if step_idx is not None:
        md["global_step_idx"] = step_idx
    return EvalRecord(
        problem_uid=uid,
        attempt_idx=attempt_idx,
        is_correct=is_correct,
        train_details=train,
        test_details=test,
        metadata=md,
    )


class TestOfficialScore:
    def test_single_puzzle_single_attempt_correct(self):
        records = [_make_eval("p1", 0, [True, True], [True])]
        rows = flatten_eval_records(records)
        assert official_score_sum(rows) == 1.0

    def test_single_puzzle_single_attempt_wrong(self):
        records = [_make_eval("p1", 0, [True], [False])]
        rows = flatten_eval_records(records)
        assert official_score_sum(rows) == 0.0

    def test_any_attempt_solves_test_pair(self):
        """Official score: per-puzzle per-test-pair, success if ANY attempt solves it."""
        records = [
            _make_eval("p1", 0, [True], [False], step_idx=0),
            _make_eval("p1", 1, [True], [True], step_idx=1),
        ]
        rows = flatten_eval_records(records)
        assert official_score_sum(rows) == 1.0

    def test_multi_test_pairs_partial(self):
        """2 test pairs, only 1 solved → official = 0.5."""
        records = [
            _make_eval("p1", 0, [True], [True, False]),
        ]
        rows = flatten_eval_records(records)
        assert official_score_sum(rows) == 0.5

    def test_multiple_puzzles(self):
        records = [
            _make_eval("p1", 0, [True], [True]),
            _make_eval("p2", 0, [True], [False]),
        ]
        rows = flatten_eval_records(records)
        assert official_score_sum(rows) == 1.0  # p1=1.0, p2=0.0

    def test_empty(self):
        assert official_score_sum([]) == 0.0


class TestStrictScore:
    def test_single_puzzle_all_correct(self):
        records = [_make_eval("p1", 0, [True, True], [True])]
        rows = flatten_eval_records(records)
        # strict with include_train=True, last step, any
        score = strict_score_sum(rows, include_train=True, step_selection="last", aggregate_step_method="any")
        assert score == 1.0

    def test_single_puzzle_train_fail(self):
        """Strict score counts train failures when include_train=True."""
        records = [_make_eval("p1", 0, [False, True], [True])]
        rows = flatten_eval_records(records)
        score = strict_score_sum(rows, include_train=True, step_selection="last", aggregate_step_method="any")
        assert score == 0.0  # train failure means step is not "all correct"

    def test_last_step_selection(self):
        """Only the last step per puzzle counts."""
        records = [
            _make_eval("p1", 0, [False], [False], step_idx=0),
            _make_eval("p1", 1, [True], [True], step_idx=1),
        ]
        rows = flatten_eval_records(records)
        score = strict_score_sum(rows, include_train=True, step_selection="last", aggregate_step_method="any")
        assert score == 1.0

    def test_empty(self):
        assert strict_score_sum([]) == 0.0
