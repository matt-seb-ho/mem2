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
        "port": "lilo",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "library_profile": "Alpha is a bridge abstraction for reusable object transformations.",
                "abstraction_proposal": {
                    "readable_name_hint": "bridge_object_transform",
                    "members_or_roles": ["bridge object", "target transform"],
                    "function_expression_hint": "bridge_object_transform(grid, object)",
                    "description": "Reusable bridge abstraction for target object transformations.",
                },
                "language_grounding": [{"phrase": "bridge object", "grounding": "source cue"}],
                "abstraction_terms": ["bridge abstraction", "target transform"],
                "iterative_growth_notes": "Propose this as one abstraction in the next LILO iteration.",
            },
            {
                "concept_id": "beta",
                "library_profile": "Beta is unrelated.",
                "abstraction_proposal": {
                    "readable_name_hint": "other",
                    "members_or_roles": ["other"],
                    "function_expression_hint": "other(grid)",
                    "description": "Unrelated abstraction.",
                },
                "language_grounding": [{"phrase": "other", "grounding": "source cue"}],
                "abstraction_terms": ["unrelated"],
                "iterative_growth_notes": "Unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_lilo_adapted_memory_renders_library_cards(tmp_path: Path):
    from mem2.branches.memory_retriever.lilo import LILOAdaptedRetriever

    retriever = LILOAdaptedRetriever(top_k=1, adapted_memory_path=_artifact(tmp_path / "lilo_memory_v1.json"))
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge abstraction"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "lilo_memory_v1"
    assert bundle.metadata["substrate_gap"] == "best_effort_non_executable_library_cards"
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "lilo_library_profile" in (bundle.hint_text or "")


def test_lilo_adapted_memory_falls_back_when_missing(tmp_path: Path):
    from mem2.branches.memory_retriever.lilo import LILOAdaptedRetriever

    retriever = LILOAdaptedRetriever(top_k=1, adapted_memory_path=tmp_path / "missing.json")
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge abstraction"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "lilo_library_profile" not in (bundle.hint_text or "")
