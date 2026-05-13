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
        uid="bridge-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "use bridge node and target object"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "hipporag_ppr",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "passage_text": "Alpha passage stores a bridge node for target object retrieval.",
                "entity_mentions": [
                    {"text": "bridge node", "type": "concept", "role": "query node", "supporting_text": "bridge node"},
                    {"text": "target object", "type": "object", "role": "retrieved passage cue", "supporting_text": "target object"},
                ],
                "triples": [
                    {"subject": "bridge node", "predicate": "activates", "object": "target object", "confidence": 0.9, "supporting_text": "bridge node for target object"}
                ],
                "query_node_terms": ["bridge node", "target object"],
                "node_specificity_hints": [{"node": "bridge node", "specificity": "high", "reason": "rare cue"}],
                "retrieval_notes": "Alpha should receive reset mass from bridge queries.",
            },
            {
                "concept_id": "beta",
                "passage_text": "Beta passage stores an unrelated memory node.",
                "entity_mentions": [
                    {"text": "unrelated memory", "type": "concept", "role": "query node", "supporting_text": "unrelated"},
                    {"text": "other object", "type": "object", "role": "cue", "supporting_text": "other"},
                ],
                "triples": [
                    {"subject": "unrelated memory", "predicate": "mentions", "object": "other object", "confidence": 0.8, "supporting_text": "other"}
                ],
                "query_node_terms": ["unrelated memory"],
                "node_specificity_hints": [],
                "retrieval_notes": "Beta is unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_hipporag_ppr_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever

    retriever = HippoRAGPPRRetriever(
        top_k=1,
        min_reset_overlap=1,
        adapted_memory_path=_artifact(tmp_path / "hipporag_ppr_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "hipporag_ppr_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["num_adapted_reset_matches"] >= 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "hipporag_passage" in (bundle.hint_text or "")
    assert "bridge node" in (bundle.hint_text or "")


def test_hipporag_ppr_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever

    retriever = HippoRAGPPRRetriever(
        top_k=1,
        min_reset_overlap=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "hipporag_passage" not in (bundle.hint_text or "")
