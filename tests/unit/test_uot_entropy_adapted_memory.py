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
        cues=["balanced split cue"],
        used_in=["p1", "p2"],
    )
    mem.concepts["beta"] = Concept(
        name="beta",
        kind="routine",
        routine_subtype="intermediate",
        description="flat beta description",
        cues=["unrelated cue"],
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
        uid="uot-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "choose a balanced split question to reduce uncertainty"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "uot_entropy",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "uncertainty_state": "Whether the task requires a balanced split.",
                "candidate_question": "Does the evidence support a balanced split question?",
                "yes_partition_hint": ["balanced split evidence appears"],
                "no_partition_hint": ["balanced split evidence is absent"],
                "expected_yes_ratio": 0.5,
                "entropy_reward": 1.0,
                "information_gain_target": "Reduce uncertainty about split selection.",
                "simulation_tree_role": "root_candidate",
                "reward_propagation_notes": "High reward should propagate to the next question.",
                "routing_keywords": ["balanced split", "reduce uncertainty"],
                "retrieval_notes": "Alpha is the relevant UoT candidate.",
            },
            {
                "concept_id": "beta",
                "uncertainty_state": "Unrelated ambiguity.",
                "candidate_question": "Does unrelated evidence appear?",
                "yes_partition_hint": ["unrelated evidence appears"],
                "no_partition_hint": ["unrelated evidence is absent"],
                "expected_yes_ratio": 0.95,
                "entropy_reward": 0.1,
                "information_gain_target": "Unrelated target.",
                "simulation_tree_role": "other",
                "reward_propagation_notes": "Low reward.",
                "routing_keywords": ["unrelated"],
                "retrieval_notes": "Beta is unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_uot_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.uot_entropy import UoTEntropyRetriever

    retriever = UoTEntropyRetriever(
        top_k=1,
        per_round_k=1,
        adapted_memory_path=_artifact(tmp_path / "uot_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "uot_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_entropy_items_rendered"] == 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "uot_question" in (bundle.hint_text or "")
    assert "balanced split question" in (bundle.hint_text or "")


def test_uot_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.uot_entropy import UoTEntropyRetriever

    retriever = UoTEntropyRetriever(
        top_k=1,
        per_round_k=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "uot_question" not in (bundle.hint_text or "")
