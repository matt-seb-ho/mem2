from __future__ import annotations

import json
from pathlib import Path

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={}, output_dir=str(Path("/tmp/test_magma_typed_views")))


def _memory() -> ConceptMemory:
    mem = ConceptMemory()
    for name, desc in [
        ("transform_object", "transform object color using a shared pattern"),
        ("filter_color", "filter objects by color before transformation"),
    ]:
        concept = Concept(
            name=name,
            kind="routine",
            description=desc,
            cues=["color transform"],
            implementation=["apply color operation"],
            used_in=["task_a", "task_b"],
        )
        mem.concepts[name] = concept
        mem.categories[concept.kind].append(name)
    return mem


def _state(mem: ConceptMemory) -> MemoryState:
    return MemoryState(schema_name="arcmemo_ps", schema_version="v1", payload=mem.to_payload())


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="magma_typed_problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"description": "transform color object relation"},
    )


def _typed_views(tmp_path: Path) -> Path:
    path = tmp_path / "magma_typed_views_v1.json"
    path.write_text(json.dumps({
        "schema_version": "1",
        "model": "fixture",
        "views": {
            "semantic": {
                "nodes": [
                    {"node_id": "concept::transform_object", "label": "transform_object", "node_type": "concept"},
                    {"node_id": "entity::e1", "label": "color transformation", "node_type": "operation", "source_concept": "transform_object"},
                ],
                "edges": [{
                    "src": "concept::transform_object",
                    "dst": "entity::e1",
                    "edge_type": "operation",
                    "weight": 1.0,
                    "supporting_text": "transform object color",
                }],
            },
            "causal": {
                "nodes": [
                    {"node_id": "concept::transform_object", "label": "transform_object", "node_type": "concept"},
                    {"node_id": "concept::filter_color", "label": "filter_color", "node_type": "concept"},
                ],
                "edges": [{
                    "src": "concept::filter_color",
                    "dst": "concept::transform_object",
                    "edge_type": "shared_operation_predicate",
                    "weight": 1.0,
                    "predicates": ["transform", "filter"],
                }],
            },
            "structural": {
                "nodes": [
                    {"node_id": "concept::transform_object", "label": "transform_object", "node_type": "concept"},
                    {"node_id": "concept::filter_color", "label": "filter_color", "node_type": "concept"},
                ],
                "edges": [{
                    "src": "concept::filter_color",
                    "dst": "concept::transform_object",
                    "edge_type": "entity_co_mention_strength",
                    "weight": 2.0,
                    "relation_types": ["color", "object"],
                }],
            },
        },
        "stats": {"num_views": 3, "edges_per_view": {"semantic": 1, "causal": 1, "structural": 1}},
    }))
    return path


def test_magma_typed_views_activate_multiple_views(tmp_path: Path):
    from mem2.branches.memory_retriever.magma import MAGMAMultiGraphRetriever

    r = MAGMAMultiGraphRetriever(
        top_k_per_view=2,
        max_active_views=3,
        typed_views_path=_typed_views(tmp_path),
    )
    bundle = r.retrieve(_ctx(), _state(_memory()), _problem(), [])
    views_used = bundle.metadata.get("views_used", [])

    assert len(views_used) >= 2
    assert "semantic" in views_used
    assert bundle.metadata["typed_views_source"] == "magma_typed_views_v1"
