"""Behavioral tests for A.2 reorg_dreamcoder (DreamCoder-style fragment compression).

These tests are the no-op replacement guard from Path-C rebuild (doc 75 §2.1).
They MUST FAIL if `consolidate()` is replaced with `return memory`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={},
                      output_dir=str(Path("/tmp/test_a2")))


SHARED_CUES = [
    "When the input grid has a repeating pattern, identify the period",
    "Count the number of distinct colors in the repeating block",
]
SHARED_IMPL = [
    "for each row in grid: find_period(row) and mark boundaries",
]


def _mem_with_shared_fragments(n_concepts: int = 3) -> ConceptMemory:
    """Build a ConceptMemory where n_concepts share SHARED_CUES + SHARED_IMPL
    plus each has unique content. All share at least one used_in problem."""
    mem = ConceptMemory()
    for i in range(n_concepts):
        c = Concept(
            name=f"concept_{i}",
            kind="routine",
            description=f"Test concept {i} with shared fragment lines",
            cues=SHARED_CUES + [f"unique_cue_{i}_alpha", f"unique_cue_{i}_beta"],
            implementation=SHARED_IMPL + [f"unique_impl_{i}"],
            used_in=[f"task_{i}", f"task_{(i + 1) % n_concepts}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _mem_no_shared_content() -> ConceptMemory:
    """Build a ConceptMemory where concepts share NO lines at all."""
    mem = ConceptMemory()
    for i in range(4):
        c = Concept(
            name=f"isolated_{i}",
            kind="routine",
            description=f"Isolated concept {i}",
            cues=[f"only_cue_{i}_x", f"only_cue_{i}_y"],
            implementation=[f"only_impl_{i}"],
            used_in=[f"task_{i}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _make_ms(mem: ConceptMemory, step: int = 20) -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload={
            **mem.to_payload(),
            "dreamcoder_reorg": {"step": step, "history": [],
                                  "outcomes": [], "trigger": "every_k",
                                  "scope": "global_rebuild"},
        },
    )


# -------------------------------------------------------------------- #
#  Test 1: NO-OP REPLACEMENT GUARD — this MUST fail if consolidate()
#  is replaced with `return memory`.
# -------------------------------------------------------------------- #

def test_reorg_dreamcoder_is_not_a_noop():
    """Replacing consolidate() with `return memory` MUST break this test.
    This is the structural defense against shortcut implementations."""
    from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder

    b = DreamCoderReorgBuilder(
        trigger="every_k", every_k=1,
        scope="global_rebuild",
        min_shared_lines=2, min_fragment_frequency=2,
    )
    mem = _mem_with_shared_fragments(n_concepts=3)
    ms = _make_ms(mem, step=1)
    before_names = set(ConceptMemory.from_payload(ms.payload).concepts.keys())

    out = b.consolidate(_ctx(), ms)

    after_mem = ConceptMemory.from_payload(out.payload)
    after_names = set(after_mem.concepts.keys())
    new_names = after_names - before_names
    assert len(new_names) >= 1, (
        "consolidate() must create at least one fragment concept — "
        "if this fails, the mechanism is a no-op"
    )
    fragment = after_mem.concepts[next(iter(new_names))]
    assert fragment.routine_subtype == "fragment"


# -------------------------------------------------------------------- #
#  Test 2: MECHANISM-FIRES — fragment compression actually removes
#  shared lines from children (the DreamCoder-distinctive behavior).
# -------------------------------------------------------------------- #

def test_reorg_dreamcoder_removes_shared_lines_from_children():
    """In global_rebuild scope, shared lines must be REMOVED from child
    concepts after fragment extraction. This is the paper's core mechanism:
    programs are rewritten using the new library, removing inlined code."""
    from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder

    b = DreamCoderReorgBuilder(
        trigger="every_k", every_k=1,
        scope="global_rebuild",
        min_shared_lines=2, min_fragment_frequency=2,
    )
    mem = _mem_with_shared_fragments(n_concepts=3)
    ms = _make_ms(mem, step=1)

    out = b.consolidate(_ctx(), ms)
    after_mem = ConceptMemory.from_payload(out.payload)

    for name in ["concept_0", "concept_1", "concept_2"]:
        c = after_mem.concepts[name]
        for shared_cue in SHARED_CUES:
            assert shared_cue not in (c.cues or []), (
                f"{name} should not contain shared cue '{shared_cue}' "
                "after global_rebuild fragment extraction"
            )
        for shared_impl in SHARED_IMPL:
            assert shared_impl not in (c.implementation or []), (
                f"{name} should not contain shared impl '{shared_impl}' "
                "after global_rebuild fragment extraction"
            )


def test_reorg_dreamcoder_fragment_contains_shared_lines():
    """The fragment concept must contain exactly the shared lines."""
    from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder

    b = DreamCoderReorgBuilder(
        trigger="every_k", every_k=1,
        scope="global_rebuild",
        min_shared_lines=2, min_fragment_frequency=2,
    )
    mem = _mem_with_shared_fragments(n_concepts=3)
    ms = _make_ms(mem, step=1)

    out = b.consolidate(_ctx(), ms)
    after_mem = ConceptMemory.from_payload(out.payload)

    fragments = [c for c in after_mem.concepts.values()
                 if getattr(c, 'routine_subtype', None) == "fragment"]
    assert len(fragments) >= 1
    frag = fragments[0]
    assert set(frag.cues) == set(SHARED_CUES)
    assert set(frag.implementation) == set(SHARED_IMPL)


# -------------------------------------------------------------------- #
#  Test 3: CONTROL — no shared content → no fragments created.
# -------------------------------------------------------------------- #

def test_reorg_dreamcoder_no_fragments_on_no_shared_content():
    """When no concepts share lines, consolidate must return memory unchanged."""
    from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder

    b = DreamCoderReorgBuilder(
        trigger="every_k", every_k=1,
        scope="global_rebuild",
        min_shared_lines=2, min_fragment_frequency=2,
    )
    mem = _mem_no_shared_content()
    ms = _make_ms(mem, step=1)
    before_count = len(ConceptMemory.from_payload(ms.payload).concepts)

    out = b.consolidate(_ctx(), ms)
    after_count = len(ConceptMemory.from_payload(out.payload).concepts)
    assert after_count == before_count


# -------------------------------------------------------------------- #
#  Test 4: MDL IMPROVEMENT — consolidation actually reduces MDL score.
# -------------------------------------------------------------------- #

def test_reorg_dreamcoder_reduces_mdl():
    """Fragment compression in global_rebuild mode must reduce MDL score.
    This validates the MDL gate fires correctly (doc 74 §0 finding 3)."""
    from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder
    from mem2.scoring.mdl import MDLScorer

    b = DreamCoderReorgBuilder(
        trigger="every_k", every_k=1,
        scope="global_rebuild",
        min_shared_lines=2, min_fragment_frequency=2,
    )
    mem_before = _mem_with_shared_fragments(n_concepts=3)
    scorer = MDLScorer(per_concept_overhead=32.0)
    mdl_before = scorer.score(mem_before).total

    ms = _make_ms(mem_before, step=1)
    out = b.consolidate(_ctx(), ms)
    mem_after = ConceptMemory.from_payload(out.payload)
    mdl_after = scorer.score(mem_after).total

    assert mdl_after < mdl_before, (
        f"MDL should decrease after fragment extraction: "
        f"before={mdl_before:.1f}, after={mdl_after:.1f}"
    )


# -------------------------------------------------------------------- #
#  Test 5: ACCRETIVE MODE REJECTION — confirms the original bug is real.
#  This test documents that accretive mode correctly prevents fragment
#  creation (MDL always increases when content is only added, never
#  removed). It's a regression test for the *understanding* of the bug.
# -------------------------------------------------------------------- #

def test_reorg_dreamcoder_yaml_default_is_global_rebuild():
    """The axis 2 YAML must set scope=global_rebuild for the DreamCoder
    condition. This test verifies the YAML fix from doc 74 §0 finding 3
    is actually wired — not just that the builder works in that mode."""
    from mem2.sweeps.catalog import load_axis_catalog
    cat = load_axis_catalog("2", Path(__file__).resolve().parents[2] / "configs" / "axes")
    dc = next((c for c in cat.conditions if c.label == "reorg_dreamcoder"), None)
    assert dc is not None, "reorg_dreamcoder condition missing from axis 2"
    assert dc.builder_cfg.get("scope") == "global_rebuild", (
        f"YAML scope must be global_rebuild, got {dc.builder_cfg.get('scope')!r}"
    )


def test_reorg_dreamcoder_accretive_mode_creates_no_fragments():
    """Accretive mode preserves child content → MDL can only increase →
    no fragments pass the gate. This is expected behavior, NOT a bug in
    accretive mode — the bug was using accretive as the YAML DEFAULT for
    a mechanism that's inherently global-rebuild (doc 74 §0 finding 3)."""
    from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder

    b = DreamCoderReorgBuilder(
        trigger="every_k", every_k=1,
        scope="accretive",
        min_shared_lines=2, min_fragment_frequency=2,
    )
    mem = _mem_with_shared_fragments(n_concepts=3)
    ms = _make_ms(mem, step=1)
    before_count = len(ConceptMemory.from_payload(ms.payload).concepts)

    out = b.consolidate(_ctx(), ms)
    after_count = len(ConceptMemory.from_payload(out.payload).concepts)
    assert after_count == before_count, (
        "Accretive mode should create zero fragments (MDL gate rejects all)"
    )
