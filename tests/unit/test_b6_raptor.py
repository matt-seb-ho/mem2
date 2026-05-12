"""Behavioral tests for B.6 raptor (RAPTOR hierarchical tree retriever).

These tests are the no-op replacement guard from Path-C rebuild (doc 75 §2.3).
They MUST FAIL if the cluster-summary rendering is removed from _render_bundle.

The paper's core mechanism: parent-summary nodes appear in the retrieved output
alongside leaf concepts. The original mem2 implementation (doc 74 §0 finding 2)
computed summaries and used them for ranking but never rendered them — the "2-layer
hierarchical retrieval" claim was misrepresentation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=42, config={},
                      output_dir=str(Path("/tmp/test_b6")))


def _raptor_mem() -> ConceptMemory:
    """Build a ConceptMemory with concepts that form a Louvain community."""
    mem = ConceptMemory()
    for i in range(3):
        c = Concept(
            name=f"pattern_concept_{i}",
            kind="routine",
            description=f"Pattern recognition technique {i} for color counting and grid analysis",
            cues=[f"if grid has repeating colors apply technique {i}"],
            implementation=[f"count_colors(grid, method={i})"],
            used_in=["task_1", "task_2"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    c = Concept(
        name="isolated_concept",
        kind="routine",
        description="Completely unrelated isolated technique",
        cues=["isolated cue only"],
        implementation=["isolated impl only"],
        used_in=["task_99"],
    )
    mem.concepts[c.name] = c
    mem.categories[c.kind].append(c.name)
    return mem


def _raptor_problem() -> ProblemSpec:
    return ProblemSpec(
        uid="test_raptor_problem",
        train_pairs=[{"input": "grid with repeating colors and patterns"}],
        test_pairs=[{"input": "another grid color counting task"}],
        metadata={"description": "color counting grid analysis"},
    )


def _make_ms(mem: ConceptMemory) -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload=mem.to_payload(),
    )


# -------------------------------------------------------------------- #
#  Test 1: NO-OP REPLACEMENT GUARD — cluster summary text must appear
#  in hint_text. This MUST fail if _render_bundle only renders leaves.
# -------------------------------------------------------------------- #

def test_raptor_is_not_a_noop():
    """The retrieved hint must contain cluster-summary text, not just leaf
    concept text. Removing the summary-rendering from _render_bundle MUST
    break this test."""
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    r = RAPTORRetriever(top_k=3, parent_ratio=0.5, min_community_size=2)
    mem = _raptor_mem()
    ms = _make_ms(mem)
    bundle = r.retrieve(_ctx(), ms, _raptor_problem(), [])

    assert bundle.hint_text is not None
    hint_lower = bundle.hint_text.lower()
    assert "cluster" in hint_lower or "summary" in hint_lower or "hierarchical" in hint_lower, (
        "hint_text must contain cluster-summary text — "
        "if missing, the 2-layer hierarchy claim is misrepresentation"
    )


# -------------------------------------------------------------------- #
#  Test 2: MECHANISM-FIRES — cluster summary includes member descriptions.
# -------------------------------------------------------------------- #

def test_raptor_renders_cluster_member_descriptions():
    """When a community is selected, its member descriptions must appear
    in hint_text. This is the paper-faithful behavior: parent-summary
    text (aggregation of child descriptions) appears in the output."""
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    r = RAPTORRetriever(top_k=3, parent_ratio=0.5, min_community_size=2)
    mem = _raptor_mem()
    ms = _make_ms(mem)
    bundle = r.retrieve(_ctx(), ms, _raptor_problem(), [])

    hint = bundle.hint_text or ""
    member_desc_fragments = [
        "pattern recognition technique",
        "color counting",
    ]
    found = sum(1 for frag in member_desc_fragments if frag in hint.lower())
    assert found >= 1, (
        f"Cluster-summary text must include member descriptions. "
        f"Searched for {member_desc_fragments} in hint, found {found}."
    )


# -------------------------------------------------------------------- #
#  Test 3: CONTROL — no communities → leaf-only retrieval.
# -------------------------------------------------------------------- #

def test_raptor_no_clusters_yields_leaf_only():
    """When all concepts are isolated (no co-activation edges → no
    communities), the retriever should still work, returning leaf-only."""
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    r = RAPTORRetriever(top_k=2, parent_ratio=0.5, min_community_size=2)
    mem = ConceptMemory()
    for i in range(3):
        c = Concept(
            name=f"solo_{i}", kind="routine",
            description=f"Solo concept {i}",
            cues=[f"cue_{i}"], implementation=[f"impl_{i}"],
            used_in=[f"unique_task_{i}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    ms = _make_ms(mem)
    problem = ProblemSpec(uid="p1", train_pairs=[{"input": "cue"}],
                          test_pairs=[], metadata={})
    bundle = r.retrieve(_ctx(), ms, problem, [])
    assert bundle.hint_text is not None
    assert bundle.metadata.get("n_clusters") == 0


# -------------------------------------------------------------------- #
#  Test 4: METADATA — retrieved_items tracks cluster entries.
# -------------------------------------------------------------------- #

def test_raptor_retrieved_items_include_cluster_info():
    """When clusters are selected, retrieved_items should include entries
    that mark them as cluster-type, so downstream analysis can distinguish
    leaf vs parent selections."""
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    r = RAPTORRetriever(top_k=3, parent_ratio=0.5, min_community_size=2)
    mem = _raptor_mem()
    ms = _make_ms(mem)
    bundle = r.retrieve(_ctx(), ms, _raptor_problem(), [])

    cluster_items = [it for it in bundle.retrieved_items
                     if it.get("type") == "cluster_summary"]
    assert len(cluster_items) >= 1, (
        "retrieved_items must include at least one cluster_summary entry "
        "when communities were selected"
    )
    assert "members" in cluster_items[0]
