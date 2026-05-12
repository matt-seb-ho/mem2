"""Behavioral tests for D.4 variant_dspy_opt (DSPy-COPRO format optimizer).

These tests are the no-op replacement guard from Path-C rebuild (doc 75 §2.5).
They MUST FAIL if the optimizer's winning variant never reaches the retriever.

The paper's core mechanism: COPRO compiles the winning instruction into the
predictor's signature — the analog in mem2 is compiling the winning variant's
render flags into the memory state so the retriever uses them at render time.
The original implementation (doc 74 §0 finding 1) wrote the winner to
memory.metadata["variant"] but ps_selector never read it — optimization was
theatre.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={},
                      output_dir=str(Path("/tmp/test_d4")))


def _concept_mem_with_cues() -> ConceptMemory:
    mem = ConceptMemory()
    for i in range(3):
        c = Concept(
            name=f"concept_{i}", kind="routine",
            description=f"Technique {i} for grid transformation",
            cues=[f"look for pattern type {i} in the grid"],
            implementation=[f"apply_transform_{i}(grid)"],
            used_in=[f"task_{i}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


# -------------------------------------------------------------------- #
#  Test 1: NO-OP REPLACEMENT GUARD — optimizer winner must be written
#  to memory.metadata["render_flags"] so the retriever can consume it.
# -------------------------------------------------------------------- #

def test_dspy_opt_writes_render_flags_to_metadata():
    """The D.4 optimizer must write the winning variant's render flags to
    memory.metadata['render_flags'] — matching what D.3x (variant_format)
    already does. Without this, ps_selector can't consume the optimization
    result."""
    from mem2.branches.memory_builder.variant_dspy_opt import DSPyOptFormatBuilder

    b = DSPyOptFormatBuilder(breadth=3, depth=1)
    problems = {f"task_{i}": ProblemSpec(uid=f"task_{i}", train_pairs=[], test_pairs=[])
                for i in range(3)}
    ms = b.initialize(_ctx(), problems)

    assert "variant" in ms.metadata, "winner variant name missing from metadata"
    assert "render_flags" in ms.metadata, (
        "render_flags missing from metadata — the retriever can't consume "
        "the optimization result without it"
    )
    flags = ms.metadata["render_flags"]
    assert isinstance(flags, dict)
    assert "skip_cues" in flags or "skip_implementation" in flags


# -------------------------------------------------------------------- #
#  Test 2: MECHANISM-FIRES — optimization actually explores variants.
# -------------------------------------------------------------------- #

def test_dspy_opt_explores_multiple_variants():
    """The optimizer must evaluate more than 1 variant (breadth > 1 is the
    COPRO distinctive: explore a population, not just the default)."""
    from mem2.branches.memory_builder.variant_dspy_opt import DSPyOptFormatBuilder

    b = DSPyOptFormatBuilder(breadth=5, depth=2)
    problems = {f"task_{i}": ProblemSpec(uid=f"task_{i}", train_pairs=[], test_pairs=[])
                for i in range(3)}
    ms = b.initialize(_ctx(), problems)

    opt_info = ms.metadata.get("dspy_opt", {})
    assert opt_info.get("history_len", 0) > 1, (
        "optimizer must evaluate multiple variants (breadth > 1)"
    )


# -------------------------------------------------------------------- #
#  Test 3: WINNER IS CONSUMABLE — ps_selector uses render_flags from
#  memory metadata when present.
# -------------------------------------------------------------------- #

def test_ps_selector_uses_variant_render_flags():
    """When memory.metadata['render_flags'] is set with skip_cues=True,
    ps_selector's rendered hint must NOT contain the concept cues.
    This verifies the optimizer→retriever wire is connected."""
    from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

    mem = _concept_mem_with_cues()
    ms_with_skip = MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload=mem.to_payload(),
        metadata={"variant": "minimal", "render_flags": {
            "skip_cues": True, "skip_implementation": True,
            "skip_parameters": True, "include_description": True,
        }},
    )
    ms_without = MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload=mem.to_payload(),
        metadata={},
    )

    r = PsSelectorRetriever(
        top_k=3, use_llm_selector=False, render_mode="full",
    )
    bundle_with = r.retrieve(_ctx(), ms_with_skip,
                              ProblemSpec(uid="p1", train_pairs=[{"input": "grid pattern"}],
                                         test_pairs=[], metadata={}), [])
    bundle_without = r.retrieve(_ctx(), ms_without,
                                 ProblemSpec(uid="p1", train_pairs=[{"input": "grid pattern"}],
                                            test_pairs=[], metadata={}), [])

    hint_with = bundle_with.hint_text or ""
    hint_without = bundle_without.hint_text or ""

    assert "apply_transform" not in hint_with.lower(), (
        "With skip_implementation=True in render_flags, implementation should "
        "be omitted from hint"
    )
    assert "apply_transform" in hint_without.lower(), (
        "Without render_flags, full render_mode should include implementation"
    )


# -------------------------------------------------------------------- #
#  Test 4: CONTROL — without render_flags in metadata, ps_selector
#  uses its default render_mode.
# -------------------------------------------------------------------- #

def test_ps_selector_ignores_empty_metadata():
    """When memory.metadata has no render_flags, ps_selector uses its
    default render_mode='full' which includes everything."""
    from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever

    mem = _concept_mem_with_cues()
    ms = MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload=mem.to_payload(),
        metadata={},
    )
    r = PsSelectorRetriever(top_k=3, use_llm_selector=False, render_mode="full")
    bundle = r.retrieve(_ctx(), ms,
                         ProblemSpec(uid="p1", train_pairs=[{"input": "grid"}],
                                    test_pairs=[], metadata={}), [])
    hint = bundle.hint_text or ""
    assert "apply_transform" in hint.lower(), (
        "Full render_mode should include implementation content"
    )
