"""Tests for async_retrieve on all memory retrievers.

Verifies that every retriever implements async_retrieve per the protocol,
and that simple retrievers (none, oe_topk) delegate to sync retrieve.
"""
from __future__ import annotations

import asyncio

from mem2.branches.memory_retriever.none import NoneMemoryRetriever
from mem2.branches.memory_retriever.oe_topk import OeTopKRetriever
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="test", seed=42, config={}, output_dir="/tmp/test")


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="p1",
        train_pairs=[{"input": [[0]], "output": [[1]]}],
        test_pairs=[{"input": [[2]], "output": [[3]]}],
    )


def _memory_empty() -> MemoryState:
    return MemoryState(schema_name="none", schema_version="v1", payload={})


def _memory_oe() -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_oe",
        schema_version="v1",
        payload={
            "entries": [
                {"problem_uid": "p1", "hint": "hint A"},
                {"problem_uid": "p1", "hint": "hint B"},
                {"problem_uid": "p2", "hint": "hint C"},
            ]
        },
    )


class TestNoneRetrieverAsync:
    def test_async_retrieve_returns_none_hint(self):
        """NoneMemoryRetriever.async_retrieve returns hint_text=None."""
        r = NoneMemoryRetriever()
        bundle = asyncio.run(r.async_retrieve(
            ctx=_ctx(), provider=None, memory=_memory_empty(),
            problem=_problem(), previous_attempts=[],
        ))
        assert bundle.hint_text is None
        assert bundle.problem_uid == "p1"

    def test_async_matches_sync(self):
        """async_retrieve returns same result as sync retrieve."""
        r = NoneMemoryRetriever()
        sync_bundle = r.retrieve(_ctx(), _memory_empty(), _problem(), [])
        async_bundle = asyncio.run(r.async_retrieve(
            ctx=_ctx(), provider=None, memory=_memory_empty(),
            problem=_problem(), previous_attempts=[],
        ))
        assert sync_bundle.hint_text == async_bundle.hint_text
        assert sync_bundle.problem_uid == async_bundle.problem_uid


class TestOeTopKRetrieverAsync:
    def test_async_retrieve_returns_hints(self):
        """OeTopKRetriever.async_retrieve returns scoped hints."""
        r = OeTopKRetriever(top_k=2)
        bundle = asyncio.run(r.async_retrieve(
            ctx=_ctx(), provider=None, memory=_memory_oe(),
            problem=_problem(), previous_attempts=[],
        ))
        assert bundle.hint_text is not None
        assert "hint A" in bundle.hint_text
        assert "hint B" in bundle.hint_text

    def test_async_matches_sync(self):
        """async_retrieve returns same result as sync retrieve."""
        r = OeTopKRetriever(top_k=2)
        sync_bundle = r.retrieve(_ctx(), _memory_oe(), _problem(), [])
        async_bundle = asyncio.run(r.async_retrieve(
            ctx=_ctx(), provider=None, memory=_memory_oe(),
            problem=_problem(), previous_attempts=[],
        ))
        assert sync_bundle.hint_text == async_bundle.hint_text
        assert sync_bundle.retrieved_items == async_bundle.retrieved_items

    def test_async_ignores_extra_params(self):
        """selector_model and provider are accepted but ignored."""
        r = OeTopKRetriever(top_k=1)
        bundle = asyncio.run(r.async_retrieve(
            ctx=_ctx(), provider="not_used", memory=_memory_oe(),
            problem=_problem(), previous_attempts=[],
            selector_model="not_used_model",
        ))
        assert bundle.hint_text is not None
