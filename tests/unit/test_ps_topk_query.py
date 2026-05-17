from __future__ import annotations

from pathlib import Path

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={}, output_dir=str(Path("/tmp/test_ps_topk_query")))


def _problem(uid: str, prompt: str) -> ProblemSpec:
    return ProblemSpec(
        uid=uid,
        train_pairs=[{"input": [[0]], "output": [[1]]}],
        test_pairs=[{"input": [[0]]}],
        metadata={"prompt": prompt},
    )


def _memory_state() -> MemoryState:
    mem = ConceptMemory()
    concepts = [
        Concept(
            name="z_popular_global",
            kind="routine",
            description="general fallback grid routine",
            used_in=["p1", "p2", "p3", "p4"],
        ),
        Concept(
            name="shape_resize",
            kind="routine",
            description="resize shape by extending object boundary",
            used_in=["p1"],
        ),
        Concept(
            name="color_recolor",
            kind="routine",
            description="recolor object cells using palette mapping",
            used_in=["p2"],
        ),
    ]
    for concept in concepts:
        mem.concepts[concept.name] = concept
        mem.categories[concept.kind].append(concept.name)
    return MemoryState(
        schema_name="arcmemo_ps",
        schema_version="v1",
        payload=mem.to_payload(),
    )


def _names(bundle) -> list[str]:
    return [item["name"] for item in bundle.retrieved_items]


def test_query_conditioned_order_differs_across_distinct_problems():
    from mem2.branches.memory_retriever.ps_topk_query import PsTopKQueryRetriever

    retriever = PsTopKQueryRetriever(top_k=2, alpha=0.9, beta=0.1)
    state = _memory_state()

    shape_bundle = retriever.retrieve(_ctx(), state, _problem("shape", "resize object boundary"), [])
    color_bundle = retriever.retrieve(_ctx(), state, _problem("color", "recolor object palette"), [])

    assert _names(shape_bundle) != _names(color_bundle)
    assert _names(shape_bundle)[0] == "shape_resize"
    assert _names(color_bundle)[0] == "color_recolor"


def test_alpha_zero_degenerates_to_popularity_only_order():
    from mem2.branches.memory_retriever.ps_topk import PsTopKRetriever
    from mem2.branches.memory_retriever.ps_topk_query import PsTopKQueryRetriever

    state = _memory_state()
    problem = _problem("shape", "resize object boundary")
    popularity_names = _names(PsTopKRetriever(top_k=3).retrieve(_ctx(), state, problem, []))
    query_names = _names(PsTopKQueryRetriever(top_k=3, alpha=0.0, beta=1.0).retrieve(_ctx(), state, problem, []))

    assert query_names == popularity_names


def test_metadata_records_scoring_parameters():
    from mem2.branches.memory_retriever.ps_topk_query import PsTopKQueryRetriever

    bundle = PsTopKQueryRetriever(top_k=2, alpha=0.6, beta=0.4).retrieve(
        _ctx(),
        _memory_state(),
        _problem("shape", "resize object boundary"),
        [],
    )

    assert bundle.metadata["scoring_mode"] == "ps_topk_query"
    assert bundle.metadata["alpha"] == 0.6
    assert bundle.metadata["beta"] == 0.4
    assert bundle.metadata["query_token_count"] > 0
    assert bundle.metadata["num_concepts_scored"] == 3
