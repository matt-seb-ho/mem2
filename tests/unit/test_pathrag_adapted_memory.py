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
        uid="path-problem",
        train_pairs=[],
        test_pairs=[],
        metadata={"prompt": "need bridge path relation"},
    )


def _artifact(path: Path) -> Path:
    data = {
        "schema_version": "1",
        "source_seed": "data/arc_agi/concept_memory/compressed_v1.json",
        "model": "deepseek/deepseek-v4-flash",
        "port": "pathrag",
        "adapted_concepts": [
            {
                "concept_id": "alpha",
                "query_keywords": ["bridge path", "relation"],
                "path_nodes": [
                    {"node_id": "n1", "label": "bridge object", "text_chunk": "bridge object cue", "node_type": "object"},
                    {"node_id": "n2", "label": "target relation", "text_chunk": "target relation cue", "node_type": "concept"},
                ],
                "entity_paths": [
                    {
                        "path_id": "p1",
                        "nodes": ["n1", "n2"],
                        "edges": [{"src": "n1", "dst": "n2", "relation": "connects_to", "text_chunk": "bridge object connects to target relation"}],
                        "textual_path": "bridge object cue; connects_to; target relation cue",
                        "reliability_hint": 0.9,
                        "pruning_rationale": "direct bridge path",
                    }
                ],
                "answer_generation_notes": "Keep this as a path.",
            },
            {
                "concept_id": "beta",
                "query_keywords": ["unrelated"],
                "path_nodes": [
                    {"node_id": "n1", "label": "other", "text_chunk": "other", "node_type": "concept"},
                    {"node_id": "n2", "label": "node", "text_chunk": "node", "node_type": "concept"},
                ],
                "entity_paths": [
                    {
                        "path_id": "p1",
                        "nodes": ["n1", "n2"],
                        "edges": [{"src": "n1", "dst": "n2", "relation": "mentions", "text_chunk": "other mentions node"}],
                        "textual_path": "other; mentions; node",
                        "reliability_hint": 0.2,
                        "pruning_rationale": "weak path",
                    }
                ],
                "answer_generation_notes": "Unrelated.",
            },
        ],
        "stats": {"num_concepts": 2, "num_failures": 0},
    }
    path.write_text(json.dumps(data))
    return path


def test_pathrag_prefers_adapted_memory_when_present(tmp_path: Path):
    from mem2.branches.memory_retriever.pathrag import PathRAGRetriever

    retriever = PathRAGRetriever(
        top_k_seeds=1,
        max_paths_rendered=2,
        adapted_memory_path=_artifact(tmp_path / "pathrag_memory_v1.json"),
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "pathrag_memory_v1"
    assert bundle.metadata["adapted_records_loaded"] == 2
    assert bundle.metadata["adapted_paths_rendered"] >= 1
    assert bundle.retrieved_items[0]["name"] == "alpha"
    assert "adapted PathRAG relational paths" in (bundle.hint_text or "")
    assert "bridge object cue; connects_to; target relation cue" in (bundle.hint_text or "")


def test_pathrag_falls_back_when_adapted_memory_absent(tmp_path: Path):
    from mem2.branches.memory_retriever.pathrag import PathRAGRetriever

    retriever = PathRAGRetriever(
        top_k_seeds=1,
        adapted_memory_path=tmp_path / "missing.json",
    )
    bundle = retriever.retrieve(_ctx(), _state(_mem()), _problem(), [])

    assert bundle.metadata["adapted_memory_source"] == "flat"
    assert bundle.metadata["adapted_records_loaded"] == 0
    assert "adapted PathRAG relational paths" not in (bundle.hint_text or "")
