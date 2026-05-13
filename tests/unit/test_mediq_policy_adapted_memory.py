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
        cues=["alpha cue"],
        used_in=["p1", "p2"],
    )
    mem.concepts["beta"] = Concept(
        name="beta",
        kind="routine",
        routine_subtype="intermediate",
        description="flat beta description",
        cues=["beta cue"],
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
        uid="mediq-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "ask about bridge target information before committing"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "mediq_policy",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "initial_assessment": "Alpha can identify when bridge target evidence is missing.",
                "question_type": "object_property",
                "missing_information_targets": ["bridge target information"],
                "atomic_question_templates": ["Which object supplies the bridge target information?"],
                "expected_info_gain": 0.95,
                "abstention_policy": {
                    "ask_when": "ask when bridge target information is absent",
                    "commit_when": "commit when the bridge target object is identified",
                    "confidence_threshold_hint": 0.8,
                },
                "evidence_integration": "Add the bridge target response to the concept state.",
                "routing_keywords": ["bridge target", "ask information"],
                "retrieval_notes": "Alpha should guide the first follow-up question.",
            },
            {
                "concept_id": "beta",
                "initial_assessment": "Beta handles unrelated evidence.",
                "question_type": "other",
                "missing_information_targets": ["unrelated feature"],
                "atomic_question_templates": ["Which unrelated feature is present?"],
                "expected_info_gain": 0.1,
                "abstention_policy": {
                    "ask_when": "ask for unrelated features",
                    "commit_when": "commit for unrelated features",
                    "confidence_threshold_hint": 0.2,
                },
                "evidence_integration": "Add unrelated evidence.",
                "routing_keywords": ["unrelated"],
                "retrieval_notes": "Beta is unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_mediq_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.mediq_policy import MediQPolicyRetriever

    retriever = MediQPolicyRetriever(
        top_k=1,
        per_round_k=1,
        adapted_memory_path=_artifact(tmp_path / "mediq_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "mediq_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_policy_items_rendered"] == 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "mediq_initial_assessment" in (bundle.hint_text or "")
    assert "Which object supplies the bridge target information?" in (bundle.hint_text or "")


def test_mediq_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.mediq_policy import MediQPolicyRetriever

    retriever = MediQPolicyRetriever(
        top_k=1,
        per_round_k=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "mediq_initial_assessment" not in (bundle.hint_text or "")
