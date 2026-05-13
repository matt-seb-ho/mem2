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
        uid="magma-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "why bridge cause entity anchor"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "source_typed_views": "data/arc_agi/concept_memory/shared/magma_typed_views_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "magma",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "event_node": {
                    "content": "Alpha is a bridge cause event that links entity anchors to a structural traversal.",
                    "timestamp_hint": "stable concept memory event",
                    "attributes": ["operation: bridge", "entity: anchor"],
                },
                "view_memberships": [
                    {
                        "view": "semantic",
                        "node_refs": ["concept::alpha"],
                        "edge_refs": ["semantic:bridge"],
                        "role": "semantic anchor for bridge entity queries",
                        "traversal_value": "expands from bridge keywords into the alpha concept",
                        "query_intents": ["SEMANTIC", "ENTITY"],
                    },
                    {
                        "view": "causal",
                        "node_refs": ["concept::alpha"],
                        "edge_refs": ["causal:bridge cause"],
                        "role": "causal bridge explanation",
                        "traversal_value": "answers why bridge cause queries should traverse alpha",
                        "query_intents": ["WHY"],
                    },
                ],
                "anchor_keywords": ["bridge cause", "entity anchor"],
                "policy_hints": {
                    "preferred_views": ["causal", "semantic"],
                    "why_signal": "bridge cause explanation",
                    "when_signal": "stable concept event",
                    "entity_signal": "entity anchor",
                },
                "graph_linearization_card": "<ref:concept::alpha> Alpha preserves bridge cause provenance for entity anchor traversal.",
                "salience_budget": {
                    "keep_full": ["causal bridge relation"],
                    "summarize_if_needed": ["semantic role"],
                },
            },
            {
                "concept_id": "beta",
                "event_node": {
                    "content": "Beta is a supporting event for unrelated structural traversal.",
                    "timestamp_hint": "stable concept memory event",
                    "attributes": ["operation: support"],
                },
                "view_memberships": [
                    {
                        "view": "structural",
                        "node_refs": ["concept::beta"],
                        "edge_refs": ["structural:support"],
                        "role": "supporting structural node",
                        "traversal_value": "weakly expands structural context",
                        "query_intents": ["STRUCTURAL"],
                    },
                    {
                        "view": "semantic",
                        "node_refs": ["concept::beta"],
                        "edge_refs": ["semantic:support"],
                        "role": "supporting semantic node",
                        "traversal_value": "adds backup semantic context",
                        "query_intents": ["SEMANTIC"],
                    },
                ],
                "anchor_keywords": ["support"],
                "policy_hints": {
                    "preferred_views": ["structural"],
                    "why_signal": "",
                    "when_signal": "stable concept event",
                    "entity_signal": "",
                },
                "graph_linearization_card": "<ref:concept::beta> Beta provides supporting structural context.",
                "salience_budget": {
                    "keep_full": ["structural support"],
                    "summarize_if_needed": ["semantic support"],
                },
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_magma_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.magma import MAGMAMultiGraphRetriever

    retriever = MAGMAMultiGraphRetriever(
        top_k_per_view=1,
        max_active_views=2,
        adapted_memory_path=_artifact(tmp_path / "magma_memory_v1.json"),
        typed_views_path=tmp_path / "missing_typed_views.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "magma_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_cards_rendered"] >= 1
    assert "causal" in bundle.metadata["active_views"]
    assert "Adapted MAGMA view records" in (bundle.hint_text or "")
    assert "bridge cause provenance" in (bundle.hint_text or "")


def test_magma_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.magma import MAGMAMultiGraphRetriever

    retriever = MAGMAMultiGraphRetriever(
        top_k_per_view=1,
        adapted_memory_path=tmp_path / "missing.json",
        typed_views_path=tmp_path / "missing_typed_views.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "Adapted MAGMA view records" not in (bundle.hint_text or "")
