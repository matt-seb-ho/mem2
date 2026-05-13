from __future__ import annotations

import json
from pathlib import Path

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={}, output_dir=str(Path("/tmp/test_prep_artifacts")))


def _memory() -> ConceptMemory:
    mem = ConceptMemory()
    for name in ("extract_objects", "filter_objects", "recolor_objects"):
        c = Concept(
            name=name,
            kind="routine",
            description=f"{name} routine for grid object reasoning",
            cues=[f"use {name} for object tasks"],
            implementation=[],
            used_in=["task_a", "task_b"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _fact_memory() -> ConceptMemory:
    mem = ConceptMemory()
    for idx, name in enumerate(("extract_objects", "filter_objects", "recolor_objects")):
        c = Concept(
            name=name,
            kind="routine",
            description=f"{name} routine for grid object reasoning",
            cues=[f"use {name} for object tasks"],
            implementation=[],
            used_in=[f"task_{idx}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _state(mem: ConceptMemory) -> MemoryState:
    return MemoryState(schema_name="arcmemo_ps", schema_version="v1", payload=mem.to_payload())


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="prep_artifact_problem",
        train_pairs=[{"input": "object extraction and recoloring"}],
        test_pairs=[],
        metadata={"description": "extract and recolor objects"},
    )


def _summary_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "community_summaries_v1.json"
    path.write_text(json.dumps({
        "schema_version": "1",
        "source_seed": "fixture",
        "source_graph": "co_activation_louvain",
        "model": "fixture",
        "communities": [{
            "community_id": "community_fixture",
            "seed_concept": "extract_objects",
            "member_concepts": ["extract_objects", "filter_objects", "recolor_objects"],
            "member_digest": "object routines",
            "llm_summary": "LLM artifact summary for object extraction, filtering, and recoloring.",
            "summary_tokens": 9,
        }],
        "stats": {},
    }))
    return path


def _openie_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "concept_facts_openie_v1.json"
    path.write_text(json.dumps({
        "schema_version": "1",
        "source_seed": "fixture",
        "model": "fixture",
        "facts": [{
            "fact_id": "openie_fixture_00001",
            "source_concept": "extract_objects",
            "subject": "extract_objects",
            "predicate": "supports",
            "object": "filter_objects",
            "confidence": 0.95,
            "supporting_text": "extract objects supports later filtering",
            "linked_concepts": ["extract_objects", "filter_objects"],
            "relation_kind": "uses",
        }],
        "stats": {"num_facts": 1},
    }))
    return path


def test_graphrag_reports_llm_summary_source_when_artifact_present(tmp_path: Path):
    from mem2.branches.memory_retriever.graphrag import GraphRAGRetriever

    mem = _memory()
    r = GraphRAGRetriever(
        top_k_communities=1,
        min_community_size=2,
        community_summaries_path=_summary_artifact(tmp_path),
    )
    bundle = r.retrieve(_ctx(), _state(mem), _problem(), [])

    assert bundle.metadata["summary_source"] == "llm_summaries_v1"
    assert "LLM artifact summary" in (bundle.hint_text or "")


def test_raptor_renders_llm_summary_when_artifact_present(tmp_path: Path):
    from mem2.branches.memory_retriever.raptor import RAPTORRetriever

    mem = _memory()
    r = RAPTORRetriever(
        top_k=2,
        parent_ratio=0.5,
        min_community_size=2,
        community_summaries_path=_summary_artifact(tmp_path),
    )
    bundle = r.retrieve(_ctx(), _state(mem), _problem(), [])

    assert bundle.metadata["summary_source"] == "llm_summaries_v1"
    assert "LLM artifact summary" in (bundle.hint_text or "")


def test_hipporag_ppr_reports_openie_fact_edges_when_artifact_present(tmp_path: Path, monkeypatch):
    from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever
    from mem2.concepts import artifacts

    monkeypatch.setattr(artifacts, "OPENIE_FACTS_PATH", _openie_artifact(tmp_path))
    mem = _fact_memory()
    r = HippoRAGPPRRetriever(top_k=2, min_reset_overlap=1)
    bundle = r.retrieve(_ctx(), _state(mem), _problem(), [])

    assert bundle.metadata["fact_graph_edges_used"] > 0
    assert bundle.metadata["num_fact_reset_matches"] > 0


def test_pathrag_renders_openie_fact_path_when_artifact_present(tmp_path: Path, monkeypatch):
    from mem2.branches.memory_retriever.pathrag import PathRAGRetriever
    from mem2.concepts import artifacts

    monkeypatch.setattr(artifacts, "OPENIE_FACTS_PATH", _openie_artifact(tmp_path))
    mem = _fact_memory()
    r = PathRAGRetriever(
        top_k_seeds=3,
        max_path_length=2,
        min_reliability=0.01,
        max_paths_rendered=1,
    )
    bundle = r.retrieve(_ctx(), _state(mem), _problem(), [])

    assert bundle.metadata["paths_rendered"] == 1
    assert "--supports--" in (bundle.hint_text or "")
    assert "extract_objects" in (bundle.hint_text or "")
    assert "filter_objects" in (bundle.hint_text or "")
