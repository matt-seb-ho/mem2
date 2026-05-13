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
        "port": "memp",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "procedure_card": "Alpha procedure handles bridge workflow and target object checks.",
                "workflow_steps": ["detect bridge object", "apply target check"],
                "success_conditions": ["bridge object present"],
                "failure_or_adjustment_signals": ["no bridge object"],
                "procedure_terms": ["bridge workflow", "target check"],
                "hit_success_notes": "Count hits when bridge workflow terms are retrieved.",
            },
            {
                "concept_id": "beta",
                "procedure_card": "Beta procedure is unrelated.",
                "workflow_steps": ["do other thing", "finish"],
                "success_conditions": ["other cue present"],
                "failure_or_adjustment_signals": ["bridge workflow"],
                "procedure_terms": ["unrelated"],
                "hit_success_notes": "Unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_memp_adapted_memory_renders_procedure_cards(tmp_path: Path):
    from mem2.branches.memory_retriever.memp import MempAdaptedRetriever

    retriever = MempAdaptedRetriever(top_k=1, adapted_memory_path=_artifact(tmp_path / "memp_memory_v1.json"))
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge workflow"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "memp_memory_v1"
    assert bundle.metadata["substrate_gap"] == "best_effort_no_trajectory_distillation"
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "memp_procedure_card" in (bundle.hint_text or "")


def test_memp_adapted_memory_falls_back_when_missing(tmp_path: Path):
    from mem2.branches.memory_retriever.memp import MempAdaptedRetriever

    retriever = MempAdaptedRetriever(top_k=1, adapted_memory_path=tmp_path / "missing.json")
    problem = ProblemSpec(uid="q", train_pairs=[], test_pairs=[], metadata={"prompt": "bridge workflow"})
    bundle = retriever.retrieve(_ctx(), _state(_mem()), problem, [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "memp_procedure_card" not in (bundle.hint_text or "")
