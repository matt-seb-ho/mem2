from __future__ import annotations

import json
from pathlib import Path

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _mem() -> ConceptMemory:
    mem = ConceptMemory()
    mem.concepts["alpha"] = Concept(
        name="alpha",
        kind="routine",
        routine_subtype="intermediate",
        description="flat alpha description",
        used_in=["p1", "p2"],
    )
    mem.concepts["beta"] = Concept(
        name="beta",
        kind="routine",
        routine_subtype="intermediate",
        description="flat beta description",
        used_in=["p1", "p2"],
    )
    mem.categories["routine"] = ["alpha", "beta"]
    return mem


def _state(mem: ConceptMemory) -> MemoryState:
    return MemoryState(schema_name="arcmemo_ps", schema_version="v1", payload=mem.to_payload())


def _ctx() -> RunContext:
    return RunContext(run_id="test", config={}, seed=0, output_dir="/tmp")


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="raptor-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "need bridge tree traversal summary"},
    )


def _ranking_mem() -> ConceptMemory:
    mem = ConceptMemory()
    mem.concepts["alpha"] = Concept(
        name="alpha",
        kind="routine",
        routine_subtype="intermediate",
        description="ordinary alpha description",
        used_in=["p1"],
    )
    mem.concepts["beta"] = Concept(
        name="beta",
        kind="routine",
        routine_subtype="intermediate",
        description="fallback beacon beta description",
        used_in=["p2"],
    )
    mem.categories["routine"] = ["alpha", "beta"]
    return mem


def _ranking_problem() -> ProblemSpec:
    return ProblemSpec(
        uid="raptor-ranking-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "fallback beta bridge traversal summary retrieval query"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "source_tree": "data/arc_agi/concept_memory/shared/raptor_tree_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "raptor",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "leaf_node_id": "rt_L0_N000",
                "tree_membership_rationale": "Alpha anchors bridge tree traversal.",
                "leaf_text": "Alpha leaf text describes bridge tree traversal and summary retrieval for the query.",
                "path_to_root": [
                    {
                        "level": 0,
                        "node_id": "rt_L0_N000",
                        "summary_role": "leaf bridge operation",
                        "retrieval_text": "bridge tree traversal leaf",
                    },
                    {
                        "level": 1,
                        "node_id": "rt_L1_N000",
                        "summary_role": "parent summary",
                        "retrieval_text": "recursive parent summary",
                    },
                ],
                "collapsed_tree_keywords": ["bridge tree", "summary retrieval"],
                "tree_traversal_cues": ["bridge query"],
            },
            {
                "concept_id": "beta",
                "leaf_node_id": "rt_L0_N001",
                "tree_membership_rationale": "Beta is a supporting leaf.",
                "leaf_text": "Beta leaf text describes a supporting recursive tree branch.",
                "path_to_root": [
                    {
                        "level": 0,
                        "node_id": "rt_L0_N001",
                        "summary_role": "supporting leaf",
                        "retrieval_text": "supporting branch leaf",
                    }
                ],
                "collapsed_tree_keywords": ["supporting branch"],
                "tree_traversal_cues": ["support query"],
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def _tree_artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "fixture",
        "model": "fixture",
        "levels": [
            {
                "level": 0,
                "nodes": [
                    {
                        "node_id": "rt_L0_ALPHA",
                        "summary": "Plain alpha leaf.",
                        "member_communities": ["alpha_comm"],
                        "member_concepts": ["alpha"],
                    },
                    {
                        "node_id": "rt_L0_BETA",
                        "summary": "Fallback beacon beta leaf.",
                        "member_communities": ["beta_comm"],
                        "member_concepts": ["beta"],
                    },
                ],
            },
            {
                "level": 1,
                "nodes": [
                    {
                        "node_id": "rt_L1_ROOT",
                        "summary": "Root summary.",
                        "member_communities": ["alpha_comm", "beta_comm"],
                        "member_concepts": ["alpha", "beta"],
                        "member_node_ids": ["rt_L0_ALPHA", "rt_L0_BETA"],
                        "child_node_ids": ["rt_L0_ALPHA", "rt_L0_BETA"],
                    }
                ],
            },
        ],
        "stats": {"num_levels": 2, "nodes_per_level": [2, 1]},
    }
    path.write_text(json.dumps(data))
    return path


def test_raptor_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    retriever = RAPTORRetriever(
        top_k=1,
        adapted_memory_path=_artifact(tmp_path / "raptor_memory_v1.json"),
        raptor_tree_path=tmp_path / "missing_tree.json",
        community_summaries_path=tmp_path / "missing_communities.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "raptor_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_leaf_records_rendered"] >= 1
    assert "RAPTOR adapted leaf records" in (bundle.hint_text or "")
    assert "bridge tree traversal" in (bundle.hint_text or "")


def test_raptor_tree_descent_uses_adapted_text_for_leaf_selection(tmp_path: Path):
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    tree_path = _tree_artifact(tmp_path / "raptor_tree_v1.json")
    mem = _ranking_mem()
    adapted = RAPTORRetriever(
        top_k=1,
        adapted_memory_path=_artifact(tmp_path / "raptor_memory_v1.json"),
        raptor_tree_path=tree_path,
        community_summaries_path=tmp_path / "missing_communities.json",
    ).retrieve(_ctx(), _state(mem), _ranking_problem(), [])
    fallback = RAPTORRetriever(
        top_k=1,
        adapted_memory_path=tmp_path / "missing_adapted.json",
        raptor_tree_path=tree_path,
        community_summaries_path=tmp_path / "missing_communities.json",
    ).retrieve(_ctx(), _state(mem), _ranking_problem(), [])

    assert adapted.metadata["scoring_mode"] == "raptor_tree_descent"
    assert fallback.metadata["scoring_mode"] == "raptor_tree_descent"
    assert adapted.metadata["tree_depth_used"] >= 2
    assert adapted.metadata["tree_path"][0] == "rt_L1_ROOT"
    assert adapted.retrieved_items[0]["name"] == "alpha"
    assert fallback.retrieved_items[0]["name"] == "beta"
    assert adapted.retrieved_items[0]["name"] != fallback.retrieved_items[0]["name"], (
        "adapted tree scoring produced the same leaf as the no-adapted path"
    )


def test_raptor_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    retriever = RAPTORRetriever(
        top_k=1,
        adapted_memory_path=tmp_path / "missing.json",
        raptor_tree_path=tmp_path / "missing_tree.json",
        community_summaries_path=tmp_path / "missing_communities.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "RAPTOR adapted leaf records" not in (bundle.hint_text or "")
