"""Tests for format-independent retrieval stages (ConceptFilter, RetrievalRouter)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mem2.retrieval.filters import ConceptFilter
from mem2.retrieval.routers import RetrievalRouter


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
        assert r.should_include(["a"], "hint") is True
        assert r.should_include(None, None) is True

    def test_confidence_all_generic_skips(self):
        """Skips when all concepts are above frequency threshold."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9, "b": 0.8},
        )
        assert r.should_include(["a", "b"], "hint") is False

    def test_confidence_mixed_includes(self):
        """Includes when at least one concept is below threshold."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequency_threshold=0.5,
            frequencies={"a": 0.9, "b": 0.2},
        )
        assert r.should_include(["a", "b"], "hint") is True

    def test_confidence_no_frequencies_includes(self):
        """Includes when no frequency data available."""
        r = RetrievalRouter(strategy="selection_confidence")
        assert r.should_include(["a"], "hint") is True

    def test_confidence_empty_names_includes(self):
        """Includes when names list is empty/None."""
        r = RetrievalRouter(
            strategy="selection_confidence",
            frequencies={"a": 0.9},
        )
        assert r.should_include(None, "hint") is True
        assert r.should_include([], "hint") is True

    def test_hint_length_within_limit(self):
        """Includes when hint is within char limit."""
        r = RetrievalRouter(strategy="hint_length", max_hint_chars=100)
        assert r.should_include(["a"], "short hint") is True

    def test_hint_length_exceeds_limit(self):
        """Skips when hint exceeds char limit."""
        r = RetrievalRouter(strategy="hint_length", max_hint_chars=10)
        assert r.should_include(["a"], "this is a very long hint text") is False

    def test_hint_length_no_limit(self):
        """max_hint_chars=0 means disabled."""
        r = RetrievalRouter(strategy="hint_length", max_hint_chars=0)
        assert r.should_include(["a"], "x" * 10000) is True

    def test_unknown_strategy_includes(self):
        """Unknown strategy defaults to include."""
        r = RetrievalRouter(strategy="unknown_future_strategy")
        assert r.should_include(["a"], "hint") is True
