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
        cues=["bridge entity cue"],
        used_in=["p1", "p2"],
    )
    mem.concepts["beta"] = Concept(
        name="beta",
        kind="object",
        description="flat beta description",
        cues=["global relation cue"],
        used_in=["p1", "p2"],
    )
    mem.categories["routine"] = ["alpha"]
    mem.categories["object"] = ["beta"]
    return mem


def _state(mem: ConceptMemory) -> MemoryState:
    return MemoryState(schema_name="arcmemo_ps", schema_version="v1", payload=mem.to_payload())


def _ctx() -> RunContext:
    return RunContext(run_id="test", config={}, seed=0, output_dir="/tmp")


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="lightrag-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "retrieve bridge entity with global relation context"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "lightrag",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "local_entities": [
                    {
                        "mention": "bridge entity",
                        "entity_type": "routine",
                        "entity_summary": "Alpha is the bridge entity used for local retrieval.",
                    }
                ],
                "global_relationships": [
                    {
                        "relation": "connects",
                        "target_concept": "beta",
                        "relation_summary": "Alpha connects to beta as global relation context.",
                        "strength": 0.95,
                    }
                ],
                "low_level_keywords": ["bridge entity", "local retrieval"],
                "high_level_keywords": ["global relation", "connected context"],
                "entity_value_summary": "Alpha local entity value for bridge entity retrieval.",
                "relation_value_summary": "Alpha global relationship value for beta context.",
                "one_hop_neighbors": ["beta"],
                "chunk_reference": "alpha concept chunk",
                "retrieval_notes": "Alpha should appear in local and global LightRAG blocks.",
            },
            {
                "concept_id": "beta",
                "local_entities": [
                    {
                        "mention": "unrelated entity",
                        "entity_type": "object",
                        "entity_summary": "Beta is unrelated for local retrieval.",
                    }
                ],
                "global_relationships": [
                    {
                        "relation": "unrelated",
                        "target_concept": "alpha",
                        "relation_summary": "Beta is weaker context.",
                        "strength": 0.1,
                    }
                ],
                "low_level_keywords": ["unrelated entity", "object"],
                "high_level_keywords": ["unrelated relation", "weak context"],
                "entity_value_summary": "Beta unrelated entity value.",
                "relation_value_summary": "Beta unrelated relationship value.",
                "one_hop_neighbors": ["alpha"],
                "chunk_reference": "beta concept chunk",
                "retrieval_notes": "Beta is secondary.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_lightrag_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    retriever = LightRAGRetriever(
        top_k_entities=1,
        top_m_relationships=1,
        embedding_npz_path=tmp_path / "missing.npz",
        embedding_meta_path=tmp_path / "missing.json",
        adapted_memory_path=_artifact(tmp_path / "lightrag_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "lightrag_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_local_items_rendered"] == 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "lightrag_local_entities" in (bundle.hint_text or "")
    assert "lightrag_global_relationships" in (bundle.hint_text or "")


def test_lightrag_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    retriever = LightRAGRetriever(
        top_k_entities=1,
        top_m_relationships=1,
        embedding_npz_path=tmp_path / "missing.npz",
        embedding_meta_path=tmp_path / "missing.json",
        adapted_memory_path=tmp_path / "missing_adapter.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "lightrag_local_entities" not in (bundle.hint_text or "")
