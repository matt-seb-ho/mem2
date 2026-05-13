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
        uid="hmem-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "route through bridge category and target trace"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "hmem_hierarchical",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "domain": "ARC-AGI",
                "category": "bridge category",
                "category_position_index": "L1:bridge",
                "subcategory": "target trace",
                "subcategory_position_index": "L2:bridge:target",
                "memory_trace": {
                    "title": "bridge trace",
                    "keywords": ["bridge", "target"],
                    "trace_summary": "Alpha stores the bridge trace used for target routing.",
                },
                "episode": {
                    "summary": "Alpha is the fine-grained episode selected by the bridge trace.",
                    "grounded_operations": ["bridge"],
                    "when_to_route_here": "route here when bridge category and target trace are requested",
                },
                "routing_keywords": ["bridge category", "target trace"],
                "confidence_weight": 0.95,
                "retrieval_notes": "Use the bridge category before descending to target trace.",
            },
            {
                "concept_id": "beta",
                "domain": "ARC-AGI",
                "category": "other category",
                "category_position_index": "L1:other",
                "subcategory": "other trace",
                "subcategory_position_index": "L2:other:trace",
                "memory_trace": {
                    "title": "other trace",
                    "keywords": ["other", "trace"],
                    "trace_summary": "Beta is unrelated.",
                },
                "episode": {
                    "summary": "Beta is an unrelated fine-grained episode.",
                    "grounded_operations": ["other"],
                    "when_to_route_here": "route here for unrelated queries",
                },
                "routing_keywords": ["other"],
                "confidence_weight": 0.7,
                "retrieval_notes": "Unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_hmem_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.hmem_hierarchical import HMEMHierarchicalRetriever

    retriever = HMEMHierarchicalRetriever(
        top_k=1,
        per_layer_top_k=1,
        adapted_memory_path=_artifact(tmp_path / "hmem_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "hmem_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["hierarchy_source"] == "adapted_memory_v1"
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "hmem_route" in (bundle.hint_text or "")
    assert "bridge category" in (bundle.hint_text or "")


def test_hmem_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.hmem_hierarchical import HMEMHierarchicalRetriever

    retriever = HMEMHierarchicalRetriever(
        top_k=1,
        per_layer_top_k=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata.get("adapted_memory_source") in {None, "flat"}
    assert bundle.metadata.get("adapted_records_loaded", 0) == 0
    assert "hmem_route" not in (bundle.hint_text or "")
