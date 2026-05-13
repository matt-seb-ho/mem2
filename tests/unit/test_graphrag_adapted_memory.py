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
        uid="graph-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "need global report bridge cluster"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "graphrag",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "primary_community_id": "community_0",
                "community_role": "bridge cluster operation",
                "contribution_to_cluster": "Alpha contributes a bridge operation to the community report. It links local object handling to global report structure so the map step can preserve cluster context.",
                "map_reduce_card": "Alpha is the bridge cluster operation used by the community report.",
                "summary_path": [
                    {"level": 0, "community_id": "community_0", "role_at_level": "leaf", "report_connection": "bridge operation"}
                ],
                "entity_relationship_claims": [
                    {"claim": "Alpha links bridge operations to report clusters.", "importance": "high"}
                ],
                "query_focus_keywords": ["global report", "bridge cluster"],
            },
            {
                "concept_id": "beta",
                "primary_community_id": "community_0",
                "community_role": "supporting operation",
                "contribution_to_cluster": "Beta contributes a supporting operation to the same community report. It is less directly related to bridge queries but remains part of the local cluster.",
                "map_reduce_card": "Beta is a supporting operation in the community report.",
                "summary_path": [
                    {"level": 0, "community_id": "community_0", "role_at_level": "leaf", "report_connection": "supporting operation"}
                ],
                "entity_relationship_claims": [
                    {"claim": "Beta supports the local operation cluster.", "importance": "medium"}
                ],
                "query_focus_keywords": ["supporting operation"],
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_graphrag_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.graphrag import GraphRAGRetriever

    retriever = GraphRAGRetriever(
        top_k_communities=1,
        min_community_size=2,
        adapted_memory_path=_artifact(tmp_path / "graphrag_memory_v1.json"),
        community_summaries_path=tmp_path / "missing_communities.json",
        hierarchical_reports_path=tmp_path / "missing_hierarchy.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "graphrag_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_cards_rendered"] >= 1
    assert "Adapted GraphRAG map cards" in (bundle.hint_text or "")
    assert "bridge cluster operation" in (bundle.hint_text or "")


def test_graphrag_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.graphrag import GraphRAGRetriever

    retriever = GraphRAGRetriever(
        top_k_communities=1,
        min_community_size=2,
        adapted_memory_path=tmp_path / "missing.json",
        community_summaries_path=tmp_path / "missing_communities.json",
        hierarchical_reports_path=tmp_path / "missing_hierarchy.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "Adapted GraphRAG map cards" not in (bundle.hint_text or "")
