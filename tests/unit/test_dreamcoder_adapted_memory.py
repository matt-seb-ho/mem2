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
        "port": "dreamcoder",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "frontier_signature": "Alpha recurring frontier fragment for bridge subtree reuse.",
                "invented_primitive_candidate": {
                    "name_hint": "dc_bridge_fragment",
                    "arity_hint": 2,
                    "typed_inputs": ["grid", "object"],
                    "typed_output": "grid",
                    "reusable_behavior": "reuse a bridge subtree on a target object",
                },
                "compression_roles": [{"role": "shared_subtree", "text": "bridge subtree"}],
                "fragment_terms": ["bridge subtree", "frontier fragment"],
                "mdl_notes": "Could reduce repeated bridge subtree description.",
            },
            {
                "concept_id": "beta",
                "frontier_signature": "Beta unrelated fragment.",
                "invented_primitive_candidate": {
                    "name_hint": "dc_other",
                    "arity_hint": 1,
                    "typed_inputs": ["grid"],
                    "typed_output": "grid",
                    "reusable_behavior": "unrelated behavior",
                },
                "compression_roles": [{"role": "recognition_cue", "text": "other"}],
                "fragment_terms": ["unrelated"],
                "mdl_notes": "Unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_dreamcoder_adapted_memory_renders_fragment_cards(tmp_path: Path):
    from mem2.branches.memory_retriever.dreamcoder import DreamCoderAdaptedRetriever

    retriever = DreamCoderAdaptedRetriever(
        top_k=1,
        adapted_memory_path=_artifact(tmp_path / "dreamcoder_memory_v1.json"),
    )
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge subtree"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "dreamcoder_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["substrate_gap"] == "best_effort_non_executable_frontier_cards"
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "dreamcoder_frontier_signature" in (bundle.hint_text or "")


def test_dreamcoder_adapted_memory_falls_back_when_missing(tmp_path: Path):
    from mem2.branches.memory_retriever.dreamcoder import DreamCoderAdaptedRetriever

    retriever = DreamCoderAdaptedRetriever(top_k=1, adapted_memory_path=tmp_path / "missing.json")
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge subtree"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "dreamcoder_frontier_signature" not in (bundle.hint_text or "")
