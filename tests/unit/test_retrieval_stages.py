"""Tests for format-independent retrieval stages (ConceptFilter, RetrievalRouter)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mem2.retrieval.filters import ConceptFilter
from mem2.retrieval.routers import RetrievalRouter, RoutingDecision


# ---------------------------------------------------------------------------
# ConceptFilter
# ---------------------------------------------------------------------------
class TestConceptFilter:
    def test_noop_when_disabled(self):
        """No filtering when max_frequency=0 and max_concepts=0."""
        f = ConceptFilter()
        names = ["a", "b", "c"]
        assert f.filter(names) == ["a", "b", "c"]

    def test_frequency_filter(self):
        """Drops concepts above max_frequency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            freq_file = Path(tmpdir) / "freq.json"
            freq_file.write_text(json.dumps({"a": 0.8, "b": 0.2, "c": 0.5}))

            f = ConceptFilter(max_frequency=0.5, frequency_file=str(freq_file))
            result = f.filter(["a", "b", "c"])
            assert result == ["b", "c"]

    def test_max_concepts_cap(self):
        """Caps to max_concepts."""
        f = ConceptFilter(max_concepts=2)
        assert f.filter(["a", "b", "c", "d"]) == ["a", "b"]

    def test_frequency_then_cap(self):
        """Frequency filter applies before cap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            freq_file = Path(tmpdir) / "freq.json"
            freq_file.write_text(json.dumps({"a": 0.9, "b": 0.1, "c": 0.2, "d": 0.1}))

            f = ConceptFilter(
                max_frequency=0.5, max_concepts=2, frequency_file=str(freq_file)
            )
            # a filtered out (0.9 > 0.5), then cap to 2
            result = f.filter(["a", "b", "c", "d"])
            assert result == ["b", "c"]

    def test_empty_input(self):
        """Empty list returns empty."""
        f = ConceptFilter(max_frequency=0.5, max_concepts=3)
        assert f.filter([]) == []

    def test_frequencies_property(self):
        """Loaded frequencies accessible via property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            freq_file = Path(tmpdir) / "freq.json"
            freq_file.write_text(json.dumps({"x": 0.5}))

            f = ConceptFilter(frequency_file=str(freq_file))
            assert f.frequencies == {"x": 0.5}

    def test_missing_frequency_file(self):
        """Missing file doesn't crash, just empty frequencies."""
        f = ConceptFilter(frequency_file="/nonexistent/path.json")
        assert f.frequencies == {}
        assert f.filter(["a", "b"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# RetrievalRouter
# ---------------------------------------------------------------------------
class TestRetrievalRouter:
    def test_none_always_includes(self):
        """'none' strategy always returns True."""
        r = RetrievalRouter(strategy="none")
        assert r.should_include(["a"], "hint").include is True
        assert r.should_include(None, None).include is True

    def test_confidence_all_generic_skips(self):
        """Skips when all concepts are above frequency threshold."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9, "b": 0.8},
        )
        assert r.should_include(["a", "b"], "hint").include is False

    def test_confidence_mixed_includes(self):
        """Includes when at least one concept is below threshold."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9, "b": 0.2},
        )
        assert r.should_include(["a", "b"], "hint").include is True

    def test_confidence_no_frequencies_includes(self):
        """Includes when no frequency data available."""
        r = RetrievalRouter(strategy="selection_confidence")
        assert r.should_include(["a"], "hint").include is True

    def test_confidence_empty_names_includes(self):
        """Includes when names list is empty/None."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequencies={"a": 0.9},
        )
        assert r.should_include(None, "hint").include is True
        assert r.should_include([], "hint").include is True

    def test_hint_length_within_limit(self):
        """Includes when hint is within char limit."""
        r = RetrievalRouter(strategy="hint_length", max_hint_chars=100)
        assert r.should_include(["a"], "short hint").include is True

    def test_hint_length_exceeds_limit(self):
        """Skips when hint exceeds char limit."""
        r = RetrievalRouter(strategy="hint_length", max_hint_chars=10)
        assert r.should_include(["a"], "this is a very long hint text").include is False

    def test_hint_length_no_limit(self):
        """max_hint_chars=0 means disabled."""
        r = RetrievalRouter(strategy="hint_length", max_hint_chars=0)
        assert r.should_include(["a"], "x" * 10000).include is True

    def test_unknown_strategy_includes(self):
        """Unknown strategy defaults to include."""
        r = RetrievalRouter(strategy="unknown_future_strategy")
        assert r.should_include(["a"], "hint").include is True


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------
class TestRoutingDecision:
    def test_bool_true(self):
        """RoutingDecision with include=True is truthy."""
        d = RoutingDecision(include=True)
        assert bool(d) is True
        assert d  # works in if-statement

    def test_bool_false(self):
        """RoutingDecision with include=False is falsy."""
        d = RoutingDecision(include=False, reasons=["test"])
        assert bool(d) is False
        assert not d  # works in `if not` pattern

    def test_reasons_default_empty(self):
        """Default reasons is empty list."""
        d = RoutingDecision(include=True)
        assert d.reasons == []

    def test_backward_compat_not_pattern(self):
        """``if not router.should_include(...)`` still works."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9},
        )
        result = r.should_include(["a"], "hint")
        # This is the pattern used in ps_selector.py
        if not result:
            skipped = True
        else:
            skipped = False
        assert skipped is True


# ---------------------------------------------------------------------------
# Composite routing (AND logic)
# ---------------------------------------------------------------------------
class TestCompositeRouting:
    def test_concept_count_gating_skip(self):
        """Skips when concept count exceeds max_concept_count."""
        r = RetrievalRouter(max_concept_count=3)
        result = r.should_include(["a", "b", "c", "d"], "hint")
        assert result.include is False
        assert any("concept_count:4>3" in reason for reason in result.reasons)

    def test_concept_count_gating_pass(self):
        """Includes when concept count is within limit."""
        r = RetrievalRouter(max_concept_count=4)
        result = r.should_include(["a", "b", "c", "d"], "hint")
        assert result.include is True

    def test_concept_count_disabled(self):
        """max_concept_count=0 means disabled."""
        r = RetrievalRouter(max_concept_count=0)
        result = r.should_include(["a", "b", "c", "d", "e", "f"], "hint")
        assert result.include is True

    def test_pre_filter_count_gating_skip(self):
        """Skips when pre_filter_count exceeds max_pre_filter_count."""
        r = RetrievalRouter(max_pre_filter_count=5)
        result = r.should_include(["a", "b"], "hint", pre_filter_count=7)
        assert result.include is False
        assert any("pre_filter_count:7>5" in reason for reason in result.reasons)

    def test_pre_filter_count_gating_pass(self):
        """Includes when pre_filter_count is within limit."""
        r = RetrievalRouter(max_pre_filter_count=5)
        result = r.should_include(["a", "b"], "hint", pre_filter_count=4)
        assert result.include is True

    def test_pre_filter_count_disabled(self):
        """max_pre_filter_count=0 means disabled."""
        r = RetrievalRouter(max_pre_filter_count=0)
        result = r.should_include(["a"], "hint", pre_filter_count=100)
        assert result.include is True

    def test_hint_chars_composite(self):
        """max_hint_chars works as composite threshold without strategy."""
        r = RetrievalRouter(max_hint_chars=10)
        result = r.should_include(["a"], "this is a long hint")
        assert result.include is False
        assert any("hint_chars:" in reason for reason in result.reasons)

    def test_composite_and_logic_all_fail(self):
        """Multiple simultaneous triggers — all reasons recorded."""
        r = RetrievalRouter(
            max_concept_count=2,
            max_hint_chars=10,
            max_pre_filter_count=3,
        )
        result = r.should_include(
            ["a", "b", "c", "d"],
            "this is a very long hint text",
            pre_filter_count=5,
        )
        assert result.include is False
        assert len(result.reasons) == 3
        assert any("concept_count:" in r for r in result.reasons)
        assert any("hint_chars:" in r for r in result.reasons)
        assert any("pre_filter_count:" in r for r in result.reasons)

    def test_composite_and_logic_one_fail(self):
        """Skip when only one composite condition fails."""
        r = RetrievalRouter(
            max_concept_count=10,
            max_hint_chars=5,  # This one fails
        )
        result = r.should_include(["a", "b"], "long enough hint")
        assert result.include is False
        assert len(result.reasons) == 1
        assert "hint_chars:" in result.reasons[0]

    def test_composite_with_strategy(self):
        """Composite thresholds AND strategy both evaluated."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9, "b": 0.8},
            max_concept_count=10,  # passes
        )
        result = r.should_include(["a", "b"], "hint")
        assert result.include is False
        assert "all_generic" in result.reasons

    def test_composite_pass_strategy_pass(self):
        """All conditions pass → include."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.1},
            max_concept_count=5,
            max_hint_chars=1000,
        )
        result = r.should_include(["a"], "short hint")
        assert result.include is True
        assert result.reasons == []

    def test_reasons_format(self):
        """Reasons follow the 'condition:actual>threshold' format."""
        r = RetrievalRouter(max_concept_count=3)
        result = r.should_include(["a", "b", "c", "d", "e"], "hint")
        assert result.reasons == ["concept_count:5>3"]

    def test_legacy_none_no_thresholds(self):
        """Legacy: strategy='none' with no thresholds behaves as before."""
        r = RetrievalRouter(strategy="none")
        result = r.should_include(["a", "b", "c"], "any hint text")
        assert result.include is True
        assert result.reasons == []

    def test_legacy_confidence_no_thresholds(self):
        """Legacy: strategy='selection_confidence' alone behaves as before."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9, "b": 0.8},
        )
        result = r.should_include(["a", "b"], "hint")
        assert result.include is False
        assert "all_generic" in result.reasons

    def test_concept_count_none_names(self):
        """Concept count check skipped when names is None."""
        r = RetrievalRouter(max_concept_count=2)
        result = r.should_include(None, "hint")
        assert result.include is True

    def test_concept_count_empty_names(self):
        """Concept count check skipped when names is empty list."""
        r = RetrievalRouter(max_concept_count=2)
        result = r.should_include([], "hint")
        assert result.include is True

    def test_hint_chars_none_text(self):
        """Hint chars check skipped when hint_text is None."""
        r = RetrievalRouter(max_hint_chars=10)
        result = r.should_include(["a"], None)
        assert result.include is True
