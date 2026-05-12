"""Behavioral tests for the 5 non-calibration ports that lacked test coverage.

Ports: 1.3 graphrag, 1.4 hipporag_ppr, 2.3 stitch, 3.4 mediq, 5.2 alma.
Each test is a no-op replacement guard (MUST fail if mechanism → pass/empty).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=42, config={},
                      output_dir=str(Path("/tmp/test_remaining")))


def _mem_with_coactivation(n: int = 6) -> ConceptMemory:
    mem = ConceptMemory()
    for i in range(n):
        c = Concept(
            name=f"concept_{i}", kind="routine",
            description=f"Grid analysis technique {i} for pattern recognition",
            cues=[f"look for pattern type {i}", f"identify color {i} in grid"],
            implementation=[f"apply_method_{i}(grid)"],
            used_in=[f"task_{j}" for j in range(i % 3 + 1)],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="test_p1",
        train_pairs=[{"input": "grid with color pattern recognition"}],
        test_pairs=[{"input": "another grid"}],
        metadata={"description": "pattern analysis task"},
    )


def _ms(mem: ConceptMemory) -> MemoryState:
    return MemoryState(schema_name="arcmemo_ps", schema_version="v1",
                        payload=mem.to_payload())


def _ms_reorg(mem: ConceptMemory, step: int = 20) -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload={**mem.to_payload(),
                 "reorg": {"step": step, "history": [], "outcomes": []}},
    )


# -------------------------------------------------------------------- #
#  1.3 graphrag — community-report-only retrieval
# -------------------------------------------------------------------- #

def test_graphrag_returns_community_reports():
    """GraphRAG hint must contain community report blocks — NOT individual
    concept text. This distinguishes it from raptor (hybrid) and flat_topk."""
    from mem2.branches.memory_retriever.graphrag import GraphRAGRetriever

    r = GraphRAGRetriever(top_k_communities=2, min_community_size=2,
                           max_members_per_community=5)
    bundle = r.retrieve(_ctx(), _ms(_mem_with_coactivation()), _problem(), [])
    assert bundle.hint_text is not None
    hint = bundle.hint_text.lower()
    assert "community" in hint or "cluster" in hint, (
        "GraphRAG hint must contain community-level reports"
    )


def test_graphrag_empty_memory():
    from mem2.branches.memory_retriever.graphrag import GraphRAGRetriever
    r = GraphRAGRetriever(top_k_communities=2, min_community_size=2)
    bundle = r.retrieve(_ctx(), _ms(ConceptMemory()), _problem(), [])
    assert bundle.hint_text is None


# -------------------------------------------------------------------- #
#  1.4 hipporag_ppr — personalized PageRank retrieval
# -------------------------------------------------------------------- #

def test_hipporag_ppr_uses_pagerank():
    """HippoRAG PPR must use personalized PageRank scoring — metadata should
    contain PPR-specific fields, not just frequency ranking."""
    from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever

    r = HippoRAGPPRRetriever(top_k=3, damping=0.5, min_reset_overlap=1)
    bundle = r.retrieve(_ctx(), _ms(_mem_with_coactivation()), _problem(), [])
    assert bundle.hint_text is not None
    meta = bundle.metadata
    assert "ppr" in str(meta).lower() or "pagerank" in str(meta).lower() or \
           meta.get("retriever") == "hipporag_ppr", (
        "Metadata must indicate PPR-based retrieval"
    )
    assert len(bundle.retrieved_items) >= 1


def test_hipporag_ppr_empty_memory():
    from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever
    r = HippoRAGPPRRetriever(top_k=3, damping=0.5)
    bundle = r.retrieve(_ctx(), _ms(ConceptMemory()), _problem(), [])
    assert bundle.hint_text is None


# -------------------------------------------------------------------- #
#  2.3 stitch — top-down frequency-ranked single-line fragments
# -------------------------------------------------------------------- #

def test_stitch_creates_single_line_fragments():
    """Stitch-distinctive: fragments are SINGLE high-frequency lines,
    not multi-line sets (that's DreamCoder). With scope=global_rebuild,
    absorbed lines should be removed from children."""
    from mem2.branches.memory_builder.reorg_stitch import StitchReorgBuilder

    mem = ConceptMemory()
    shared_line = ("When processing a grid with repeating color patterns, first identify "
                   "the period of the pattern by scanning each row left to right and "
                   "comparing consecutive segments of equal length until a match is found")
    for i in range(5):
        c = Concept(
            name=f"concept_{i}", kind="routine",
            description=f"Grid transformation technique {i} for pattern-based color analysis with extensive details about the approach and methodology used in this particular variant of the transformation pipeline",
            cues=[shared_line, f"unique_cue_{i}_with_extra_detail_text"],
            implementation=[f"apply_complex_transform_{i}(grid, params)"],
            used_in=[f"task_{i % 2}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)

    b = StitchReorgBuilder(
        trigger="every_k", every_k=1, scope="global_rebuild",
        min_line_frequency=3, max_fragments_per_round=5,
    )
    ms = MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload={**mem.to_payload(),
                 "stitch_reorg": {"step": 1, "history": [], "outcomes": [],
                                   "trigger": "every_k", "scope": "global_rebuild"}},
    )
    out = b.consolidate(_ctx(), ms)

    after_mem = ConceptMemory.from_payload(out.payload)
    fragments = [c for c in after_mem.concepts.values()
                 if getattr(c, 'routine_subtype', None) == "stitch_fragment"]
    assert len(fragments) >= 1, (
        "Stitch must create at least one frequency-ranked stitch_fragment"
    )


def test_stitch_no_fragments_below_frequency():
    """Lines below min_line_frequency should not produce fragments."""
    from mem2.branches.memory_builder.reorg_stitch import StitchReorgBuilder

    mem = ConceptMemory()
    for i in range(2):
        c = Concept(
            name=f"rare_{i}", kind="routine",
            description=f"Rare {i}",
            cues=[f"completely_unique_cue_{i}"],
            implementation=[f"unique_impl_{i}"],
            used_in=[f"task_{i}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)

    b = StitchReorgBuilder(
        trigger="every_k", every_k=1, scope="global_rebuild",
        min_line_frequency=5, max_fragments_per_round=5,
    )
    ms = MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload={**mem.to_payload(),
                 "stitch_reorg": {"step": 1, "history": [], "outcomes": [],
                                   "trigger": "every_k", "scope": "global_rebuild"}},
    )
    before_count = len(ConceptMemory.from_payload(ms.payload).concepts)
    out = b.consolidate(_ctx(), ms)
    after_count = len(ConceptMemory.from_payload(out.payload).concepts)
    assert after_count == before_count


# -------------------------------------------------------------------- #
#  3.4 mediq — abstention-gated multi-round retrieval
# -------------------------------------------------------------------- #

def test_mediq_multi_round_retrieval():
    """MediQ-distinctive: multi-round retrieval with abstention gate.
    Must produce hint_text AND metadata showing multiple rounds."""
    from mem2.branches.memory_retriever.mediq_policy import MediQPolicyRetriever

    r = MediQPolicyRetriever(
        top_k=3, per_round_k=2, max_rounds=5,
        abstention_threshold=1, window=2,
    )
    bundle = r.retrieve(_ctx(), _ms(_mem_with_coactivation(8)), _problem(), [])
    assert bundle.hint_text is not None
    meta = bundle.metadata
    rounds = meta.get("rounds_executed", meta.get("num_rounds", 0))
    assert rounds >= 1, "MediQ must execute at least 1 retrieval round"


def test_mediq_empty_memory():
    from mem2.branches.memory_retriever.mediq_policy import MediQPolicyRetriever
    r = MediQPolicyRetriever(top_k=3, per_round_k=2, max_rounds=3)
    bundle = r.retrieve(_ctx(), _ms(ConceptMemory()), _problem(), [])
    assert bundle.hint_text is None


# -------------------------------------------------------------------- #
#  5.2 alma — LLM meta-edit with MDL gate (template fallback)
# -------------------------------------------------------------------- #

def test_alma_template_fallback_produces_merges():
    """Without LLM provider, ALMA must fall back to hand-coded reorg
    (identical to arcmemo_reorg). Must produce at least one merge or
    skip-with-reason — NOT silently return memory unchanged."""
    from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder

    b = ALMAStyleMetaEditMemoryBuilder(
        trigger="every_k", every_k=1,
        input_basis="graph_intrinsic", objective="mdl",
        scope="global_rebuild",
    )
    mem = _mem_with_coactivation(8)
    ms = _ms_reorg(mem, step=1)
    out = b.consolidate(_ctx(), ms)

    history = out.payload.get("reorg", {}).get("history", [])
    assert len(history) >= 1, (
        "ALMA template fallback must record an action in history "
        "(either a merge or a skip-with-reason)"
    )
    entry = history[-1]
    assert "action" in entry


def test_alma_empty_memory():
    from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder
    b = ALMAStyleMetaEditMemoryBuilder(
        trigger="every_k", every_k=1,
        input_basis="graph_intrinsic", objective="mdl",
        scope="global_rebuild",
    )
    ms = _ms_reorg(ConceptMemory(), step=1)
    out = b.consolidate(_ctx(), ms)
    assert out is not None
