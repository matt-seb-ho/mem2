from __future__ import annotations

from pathlib import Path

from mem2.branches.memory_builder.variant_formats import RENDER_FLAGS
from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={}, output_dir=str(Path("/tmp/test_variant_rendering")))


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="variant_problem",
        train_pairs=[{"input": "object color transform"}],
        test_pairs=[],
        metadata={"description": "object color transform"},
    )


def _memory() -> ConceptMemory:
    mem = ConceptMemory()
    concepts = [
        Concept(
            name="object_transform",
            kind="routine",
            description="transform object colors based on a pattern",
            cues=["object color cue"],
            implementation=["apply transform"],
            used_in=["task_a", "task_b"],
        ),
        Concept(
            name="shape_principle",
            kind="structure",
            description="shape components define the object layout",
            cues=["shape cue"],
            implementation=[],
            used_in=["task_a"],
        ),
    ]
    for concept in concepts:
        mem.concepts[concept.name] = concept
        mem.categories[concept.kind].append(concept.name)
    return mem


def _state(mem: ConceptMemory, *, variant: str, render_flags: dict) -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps",
        schema_version="v1",
        payload=mem.to_payload(),
        metadata={"variant": variant, "render_flags": dict(render_flags)},
    )


def test_variant_free_text_is_paragraph_not_minimal_structure():
    from mem2.branches.memory_retriever.ps_topk import PsTopKRetriever

    mem = _memory()
    retriever = PsTopKRetriever(top_k=2, usage_threshold=0)
    free_bundle = retriever.retrieve(
        _ctx(),
        _state(mem, variant="free_text", render_flags=RENDER_FLAGS["free_text"]),
        _problem(),
        [],
    )
    minimal_bundle = retriever.retrieve(
        _ctx(),
        _state(mem, variant="minimal", render_flags=RENDER_FLAGS["minimal"]),
        _problem(),
        [],
    )

    free_text = free_bundle.hint_text or ""
    minimal_text = minimal_bundle.hint_text or ""
    assert free_text != minimal_text
    assert free_text.startswith("Recall the following concepts")
    assert "- " not in free_text
    assert "* " not in free_text
    assert "1." not in free_text
    assert "- concept:" in minimal_text


def test_variant_parse_kind_overrides_skip_routines_in_ps_topk():
    from mem2.branches.memory_retriever.ps_topk import PsTopKRetriever

    mem = _memory()
    flags = dict(RENDER_FLAGS["structured_routine"])
    flags["parse_kind_overrides"] = {"routine": "skip", "structure": "compact"}
    bundle = PsTopKRetriever(top_k=2, usage_threshold=0).retrieve(
        _ctx(),
        _state(mem, variant="parse_refined_structured_routine", render_flags=flags),
        _problem(),
        [],
    )
    hint = bundle.hint_text or ""
    assert "object_transform" not in hint
    assert "shape_principle" in hint


def test_variant_parse_kind_overrides_skip_routines_in_ps_selector():
    from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

    mem = _memory()
    flags = dict(RENDER_FLAGS["structured_routine"])
    flags["parse_kind_overrides"] = {"routine": "skip", "structure": "compact"}
    bundle = PsSelectorRetriever(use_llm_selector=False, render_mode="full").retrieve(
        _ctx(),
        _state(mem, variant="parse_refined_structured_routine", render_flags=flags),
        _problem(),
        [],
    )
    hint = bundle.hint_text or ""
    assert "object_transform" not in hint
    assert "shape_principle" in hint
