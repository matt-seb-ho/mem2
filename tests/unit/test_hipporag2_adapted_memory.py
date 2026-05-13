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
        uid="filter-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "need bridge filter evidence for target object"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "hipporag2",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "ppr_passage": "Alpha passage stores bridge filter evidence for target object retrieval.",
                "candidate_profile": "Alpha should survive the HippoRAG2 filter for bridge target queries.",
                "query_filter_terms": ["bridge filter", "target object"],
                "filter_evidence": [
                    {"claim": "Alpha links bridge filters to target objects.", "supporting_text": "bridge target", "specificity": "high"}
                ],
                "reject_signals": ["unrelated color-only queries"],
                "rerank_notes": "Keep alpha when bridge evidence appears.",
            },
            {
                "concept_id": "beta",
                "ppr_passage": "Beta passage stores unrelated memory.",
                "candidate_profile": "Beta should rank lower for bridge filter queries.",
                "query_filter_terms": ["unrelated memory"],
                "filter_evidence": [
                    {"claim": "Beta is unrelated.", "supporting_text": "unrelated", "specificity": "low"}
                ],
                "reject_signals": ["bridge filter"],
                "rerank_notes": "Drop beta.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_hipporag2_prefers_adapted_filter_memory(tmp_path: Path):
    from mem2.branches.memory_retriever.hipporag2 import HippoRAG2FilterRetriever

    retriever = HippoRAG2FilterRetriever(
        first_stage_top_k=2,
        top_k=1,
        min_reset_overlap=1,
        adapted_memory_path=_artifact(tmp_path / "hipporag2_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "hipporag2_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_filter_cards_rendered"] == 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "hipporag2_ppr_passage" in (bundle.hint_text or "")
    assert "bridge filter" in (bundle.hint_text or "")


def test_hipporag2_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.hipporag2 import HippoRAG2FilterRetriever

    retriever = HippoRAG2FilterRetriever(
        first_stage_top_k=2,
        top_k=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "hipporag2_ppr_passage" not in (bundle.hint_text or "")
