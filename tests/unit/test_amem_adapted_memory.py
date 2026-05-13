from __future__ import annotations

import json
from pathlib import Path

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _mem() -> ConceptMemory:
    mem = ConceptMemory()
    mem.concepts["alpha"] = Concept(name="alpha", kind="routine", description="flat alpha", used_in=["p1"])
    mem.concepts["beta"] = Concept(name="beta", kind="routine", description="flat beta", used_in=["p1"])
    mem.categories["routine"] = ["alpha", "beta"]
    return mem


def _state(mem: ConceptMemory) -> MemoryState:
    return MemoryState(schema_name="arcmemo_ps", schema_version="v1", payload=mem.to_payload())


def _ctx() -> RunContext:
    return RunContext(run_id="test", config={}, seed=0, output_dir="/tmp")


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "port": "amem",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "note": {
                    "content": "Alpha is an atomic bridge note for target object reasoning.",
                    "timestamp": "stable concept memory",
                    "keywords": ["bridge note", "target object"],
                    "tags": ["zettel", "bridge"],
                    "contextual_description": "Alpha links bridge note reasoning to target object retrieval in an A-Mem note network.",
                },
                "zettel_links": [
                    {
                        "target_concept": "beta",
                        "link_type": "applied_with",
                        "rationale": "Beta can be retrieved as a related note when bridge note context needs a target object support concept.",
                        "confidence": 0.9,
                    }
                ],
                "memory_evolution": {
                    "context_update": "Strengthen bridge note context after linking.",
                    "tag_updates": ["bridge"],
                    "neighbor_update_suggestions": [
                        {"target_concept": "beta", "suggested_update": "Add target object bridge support."}
                    ],
                },
                "retrieval_text": "bridge note target object A-Mem retrieval",
            },
            {
                "concept_id": "beta",
                "note": {
                    "content": "Beta is a supporting note for ordinary retrieval.",
                    "timestamp": "stable concept memory",
                    "keywords": ["ordinary"],
                    "tags": ["support"],
                    "contextual_description": "Beta supports ordinary note retrieval without bridge emphasis.",
                },
                "zettel_links": [
                    {
                        "target_concept": "alpha",
                        "link_type": "similar_to",
                        "rationale": "Alpha and beta are both small test notes.",
                        "confidence": 0.5,
                    }
                ],
                "memory_evolution": {
                    "context_update": "Keep support tags.",
                    "tag_updates": ["support"],
                    "neighbor_update_suggestions": [],
                },
                "retrieval_text": "ordinary support retrieval",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_amem_adapted_memory_renders_zettelkasten_notes(tmp_path: Path):
    from mem2.branches.memory_retriever.amem import AMEMAdaptedRetriever

    retriever = AMEMAdaptedRetriever(top_k=1, adapted_memory_path=_artifact(tmp_path / "amem_memory_v1.json"))
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge note target object"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "amem_memory_v1"
    assert bundle.metadata["substrate"] == "zettelkasten_note_v1"
    assert bundle.metadata["zettel_links_rendered"] == 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "amem_note" in (bundle.hint_text or "")
    assert "zettel_links" in (bundle.hint_text or "")


def test_amem_adapted_memory_falls_back_when_missing(tmp_path: Path):
    from mem2.branches.memory_retriever.amem import AMEMAdaptedRetriever

    retriever = AMEMAdaptedRetriever(top_k=1, adapted_memory_path=tmp_path / "missing.json")
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge note target object"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "amem_note" not in (bundle.hint_text or "")
