"""Behavioral tests for A.6 reorg_amem (A-MEM per-note evolution).

No-op replacement guard from Path-C rebuild (doc 75 §2.2).
The paper's core mechanism: per-note link AND tag enrichment.
Doc 74 §0 finding 5: template-mode tag enrichment was a no-op
(_evolve_via_template returned tags_to_update: []).
"""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={},
                      output_dir=str(Path("/tmp/test_a6")))


def _amem_mem() -> ConceptMemory:
    mem = ConceptMemory()
    for i in range(6):
        c = Concept(
            name=f"concept_{i}", kind="routine",
            description=f"Grid transformation technique {i} for color pattern analysis",
            cues=[f"cue_{i}"], implementation=[f"impl_{i}"],
            used_in=[f"task_{j}" for j in range(i % 3 + 1)],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _make_ms(mem: ConceptMemory) -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload={**mem.to_payload(), "reorg": {"step": 20, "history": []}},
    )


def _amem_link_graph(tmp_path: Path) -> Path:
    path = tmp_path / "amem_link_graph_v1.json"
    path.write_text(json.dumps({
        "schema_version": "1",
        "source_seed": "fixture",
        "model": "fixture",
        "links": [
            {
                "source_concept": "concept_0",
                "target_concept": "concept_1",
                "link_type": "applied_with",
                "rationale": "both support color pattern analysis",
                "confidence": 0.9,
            },
            {
                "source_concept": "concept_1",
                "target_concept": "concept_2",
                "link_type": "specializes",
                "rationale": "concept_2 is a more specific pattern case",
                "confidence": 0.8,
            },
        ],
        "stats": {"num_links": 2},
    }))
    return path


def test_amem_template_mode_produces_nonempty_tags():
    """Template-mode _evolve_via_template must return non-empty tags_to_update.
    This MUST fail if tags_to_update is hardcoded to []."""
    from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder

    b = AMEMAgenticMemoryBuilder(
        every_k=20, trigger="every_k",
        k_neighbors=3, max_notes_per_pass=6, min_neighbor_strength=0.0,
    )
    mem = _amem_mem()
    ms = _make_ms(mem)
    out = b.consolidate(_ctx(), ms)

    history = out.payload["reorg"]["history"]
    assert len(history) >= 1
    actions_log = history[0].get("actions", [])
    tag_actions = [app for entry in actions_log
                   for app in entry.get("applied", [])
                   if app.get("type") == "tags"]
    assert len(tag_actions) >= 1, (
        "Template mode must produce at least one tag action — "
        "if zero, tags_to_update is still empty (doc 74 finding 5)"
    )
    for ta in tag_actions:
        assert len(ta.get("tags", [])) >= 1


def test_amem_link_enrichment_fires():
    """A-MEM link enrichment must add [A-MEM linked: ...] markers."""
    from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder

    b = AMEMAgenticMemoryBuilder(
        every_k=20, trigger="every_k",
        k_neighbors=3, max_notes_per_pass=6, min_neighbor_strength=0.0,
    )
    mem = _amem_mem()
    ms = _make_ms(mem)
    out = b.consolidate(_ctx(), ms)

    after_mem = ConceptMemory.from_payload(out.payload)
    linked = [c for c in after_mem.concepts.values()
              if "[A-MEM linked:" in (c.description or "")]
    assert len(linked) >= 1, "At least one concept should have a link marker"


def test_amem_concept_count_unchanged():
    """A-MEM invariant: concept count must NOT change (enrichment only)."""
    from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder

    mem = _amem_mem()
    before = len(mem.concepts)
    b = AMEMAgenticMemoryBuilder(
        every_k=20, trigger="every_k",
        k_neighbors=3, max_notes_per_pass=6, min_neighbor_strength=0.0,
    )
    ms = _make_ms(mem)
    out = b.consolidate(_ctx(), ms)
    after = len(ConceptMemory.from_payload(out.payload).concepts)
    assert after == before


def test_amem_no_evolution_below_threshold():
    """With min_neighbor_strength very high, no evolution should fire."""
    from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder

    b = AMEMAgenticMemoryBuilder(
        every_k=20, trigger="every_k",
        k_neighbors=3, max_notes_per_pass=6, min_neighbor_strength=999.0,
    )
    mem = _amem_mem()
    ms = _make_ms(mem)
    out = b.consolidate(_ctx(), ms)

    after_mem = ConceptMemory.from_payload(out.payload)
    linked = [c for c in after_mem.concepts.values()
              if "[A-MEM linked:" in (c.description or "")]
    assert len(linked) == 0


def test_amem_link_graph_persists_zettelkasten_links(tmp_path: Path):
    """A-MEM should persist typed Zettelkasten links on evolved concepts."""
    from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder

    valid_types = {
        "generalizes",
        "specializes",
        "prerequisite_of",
        "contrast_with",
        "applied_with",
        "related_to",
    }
    b = AMEMAgenticMemoryBuilder(
        every_k=20,
        trigger="every_k",
        k_neighbors=3,
        max_notes_per_pass=6,
        min_neighbor_strength=0.0,
        link_graph_path=_amem_link_graph(tmp_path),
    )
    mem = _amem_mem()
    out = b.consolidate(_ctx(), _make_ms(mem))
    after_mem = ConceptMemory.from_payload(out.payload)

    linked = [
        link
        for concept in after_mem.concepts.values()
        for link in concept.links
    ]
    assert linked
    assert all(link["link_type"] in valid_types for link in linked)
    assert any(link["link_type"] == "applied_with" for link in linked)
