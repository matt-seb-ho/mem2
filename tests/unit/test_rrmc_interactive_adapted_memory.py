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
        cues=["bridge refinement cue"],
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
        uid="rrmc-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "run bridge refinement probes before the final commit"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "rrmc_interactive",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "selector_role": "refinement_probe",
                "round_1_relevance": 0.95,
                "round_2_relevance": 0.9,
                "coverage_targets": ["bridge refinement coverage"],
                "probe_plan": [
                    {
                        "round": 1,
                        "probe_question": "Which bridge refinement cue should seed the selector?",
                        "expected_signal": "The problem exposes bridge refinement evidence.",
                        "selector_update": "Add alpha as the seed selector concept.",
                    },
                    {
                        "round": 2,
                        "probe_question": "Does the bridge refinement cue survive after comparison?",
                        "expected_signal": "The cue remains consistent after round one.",
                        "selector_update": "Keep alpha and commit if no contradiction appears.",
                    },
                ],
                "convergence_signal": "Commit when bridge refinement evidence is stable.",
                "routing_keywords": ["bridge refinement", "selector commit"],
                "retrieval_notes": "Alpha is the relevant multi-round selector.",
            },
            {
                "concept_id": "beta",
                "selector_role": "other",
                "round_1_relevance": 0.1,
                "round_2_relevance": 0.1,
                "coverage_targets": ["unrelated coverage"],
                "probe_plan": [
                    {
                        "round": 1,
                        "probe_question": "Which unrelated cue appears?",
                        "expected_signal": "Unrelated evidence appears.",
                        "selector_update": "Add beta for unrelated tasks.",
                    },
                    {
                        "round": 2,
                        "probe_question": "Does unrelated evidence persist?",
                        "expected_signal": "Unrelated evidence persists.",
                        "selector_update": "Keep beta for unrelated tasks.",
                    },
                ],
                "convergence_signal": "Commit for unrelated evidence.",
                "routing_keywords": ["unrelated"],
                "retrieval_notes": "Beta is unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_rrmc_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.rrmc_interactive import RRMCInteractiveRetriever

    retriever = RRMCInteractiveRetriever(
        top_k=1,
        per_round_k=1,
        adapted_memory_path=_artifact(tmp_path / "rrmc_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "rrmc_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_selector_items_rendered"] == 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "rrmc_selector_role" in (bundle.hint_text or "")
    assert "Which bridge refinement cue should seed the selector?" in (bundle.hint_text or "")


def test_rrmc_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.rrmc_interactive import RRMCInteractiveRetriever

    retriever = RRMCInteractiveRetriever(
        top_k=1,
        per_round_k=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "rrmc_selector_role" not in (bundle.hint_text or "")
