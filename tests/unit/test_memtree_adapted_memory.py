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
        uid="memtree-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "need bridge target collapsed tree retrieval"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "source_hierarchy": "data/arc_agi/concept_memory/shared/hierarchical_reports_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "memtree",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "tree_position": {
                    "leaf_node_id": "memtree::alpha",
                    "parent_node_id": "L0_C000",
                    "depth": 2,
                    "insertion_decision": "traverse_deeper",
                    "depth_threshold_rationale": "Alpha belongs under the bridge parent because tree queries mention bridge target retrieval.",
                },
                "node_content": {
                    "leaf_content": "Alpha leaf handles bridge target retrieval through a collapsed tree memory node.",
                    "embedding_text": "bridge target collapsed tree retrieval alpha leaf placement",
                    "aggregate_contribution": "Alpha updates the parent with bridge target placement evidence.",
                },
                "path_to_root": [
                    {
                        "node_id": "memtree::alpha",
                        "depth": 2,
                        "content_summary": "bridge target leaf",
                        "update_role": "leaf evidence",
                    },
                    {
                        "node_id": "L0_C000",
                        "depth": 1,
                        "content_summary": "bridge target parent",
                        "update_role": "parent aggregate",
                    },
                ],
                "collapsed_retrieval_card": "Alpha collapsed tree card for bridge target retrieval from the MemTree index.",
                "retrieval_keywords": ["bridge target", "collapsed tree"],
                "sibling_group": {
                    "sibling_role": "bridge sibling",
                    "near_sibling_concepts": ["beta"],
                },
            },
            {
                "concept_id": "beta",
                "tree_position": {
                    "leaf_node_id": "memtree::beta",
                    "parent_node_id": "L0_C001",
                    "depth": 2,
                    "insertion_decision": "create_new_leaf",
                    "depth_threshold_rationale": "Beta forms a separate supporting branch for generic tree retrieval.",
                },
                "node_content": {
                    "leaf_content": "Beta leaf stores supporting tree retrieval information away from the bridge target.",
                    "embedding_text": "supporting branch generic tree retrieval beta leaf",
                    "aggregate_contribution": "Beta updates another parent with supporting branch evidence.",
                },
                "path_to_root": [
                    {
                        "node_id": "memtree::beta",
                        "depth": 2,
                        "content_summary": "supporting tree leaf",
                        "update_role": "leaf evidence",
                    }
                ],
                "collapsed_retrieval_card": "Beta collapsed tree card for a supporting retrieval branch.",
                "retrieval_keywords": ["supporting branch"],
                "sibling_group": {
                    "sibling_role": "support sibling",
                    "near_sibling_concepts": ["alpha"],
                },
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_memtree_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.memtree import MemTreeAdaptedRetriever

    retriever = MemTreeAdaptedRetriever(
        top_k=1,
        adapted_memory_path=_artifact(tmp_path / "memtree_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "memtree_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_nodes_rendered"] == 1
    assert bundle.metadata["tree_paths_rendered"] >= 2
    assert bundle.metadata["retrieval_mode"] == "collapsed_tree"
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "memtree_leaf: memtree::alpha" in (bundle.hint_text or "")
    assert "path_to_root" in (bundle.hint_text or "")


def test_memtree_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.memtree import MemTreeAdaptedRetriever

    retriever = MemTreeAdaptedRetriever(top_k=1, adapted_memory_path=tmp_path / "missing.json")
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert bundle.metadata["adapted_nodes_rendered"] == 0
    assert "memtree_leaf:" not in (bundle.hint_text or "")
