"""Behavioral tests for B.7 lightrag (dual-level entity + relationship retriever).

No-op replacement guard from Path-C rebuild (doc 75 §2.4).
The paper's core mechanism: dual-level retrieval — entity-level (local) AND
relationship-level (global) content appears in the hint. The mem2 port
substitutes token-overlap for embedding/LLM keyword extraction and always
runs in "hybrid" mode (no mode toggle). Graded reduced-but-honest.
"""
from __future__ import annotations

from pathlib import Path
import json

import pytest
import numpy as np

from mem2.concepts.data import Concept
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import MemoryState, ProblemSpec, RunContext


def _ctx() -> RunContext:
    return RunContext(run_id="unit", seed=0, config={},
                      output_dir=str(Path("/tmp/test_b7")))


def _lightrag_mem() -> ConceptMemory:
    """Memory with concepts that share used_in → co-activation edges exist."""
    mem = ConceptMemory()
    for i in range(4):
        c = Concept(
            name=f"transform_{i}", kind="routine",
            description=f"Grid color transformation technique {i}",
            cues=[f"apply color transform {i} to grid"],
            implementation=[f"color_transform_{i}(grid)"],
            used_in=["task_1", f"task_{i + 2}"],
        )
        mem.concepts[c.name] = c
        mem.categories[c.kind].append(c.name)
    return mem


def _problem() -> ProblemSpec:
    return ProblemSpec(
        uid="test_lr_1",
        train_pairs=[{"input": "grid with color transformation pattern"}],
        test_pairs=[{"input": "another color grid"}],
        metadata={"description": "color grid task"},
    )


def _make_ms(mem: ConceptMemory) -> MemoryState:
    return MemoryState(
        schema_name="arcmemo_ps", schema_version="v1",
        payload=mem.to_payload(),
    )


def _lightrag_embedding_artifact(tmp_path: Path, mem: ConceptMemory) -> tuple[Path, Path]:
    npz_path = tmp_path / "lightrag_embed_v1.npz"
    meta_path = tmp_path / "lightrag_embed_v1.json"
    names = sorted(mem.concepts.keys())
    concept_embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    concept_embeddings /= np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    entity_sources = [names[0], names[1]]
    entity_embeddings = np.asarray([[1.0, 0.0, 0.0], [0.85, 0.15, 0.0]], dtype=np.float32)
    entity_embeddings /= np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
    np.savez_compressed(
        npz_path,
        concept_embeddings=concept_embeddings,
        entity_embeddings=entity_embeddings,
    )
    meta_path.write_text(json.dumps({
        "schema_version": "1",
        "model": "fixture",
        "dim": 3,
        "concept_ids": names,
        "entity_keys": ["ent0", "ent1"],
        "entity_sources": entity_sources,
        "entity_mentions": ["color transformation", "grid color"],
        "entity_types": ["operation", "target_object"],
        "stats": {"num_concepts": len(names), "num_entities": 2},
    }))
    return npz_path, meta_path


def test_lightrag_is_not_a_noop():
    """Replacing retrieve() with 'return empty bundle' MUST break this.
    The hint must contain BOTH entity AND relationship content."""
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    r = LightRAGRetriever(top_k_entities=3, top_m_relationships=3, min_edge_weight=0.5)
    bundle = r.retrieve(_ctx(), _make_ms(_lightrag_mem()), _problem(), [])

    assert bundle.hint_text is not None
    hint = bundle.hint_text
    assert "entities (local)" in hint, "Must have entity-level (local) section"
    assert "relationships (global)" in hint, "Must have relationship-level (global) section"


def test_lightrag_dual_level_items():
    """retrieved_items must contain both entity and edge types."""
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    r = LightRAGRetriever(top_k_entities=3, top_m_relationships=3, min_edge_weight=0.5)
    bundle = r.retrieve(_ctx(), _make_ms(_lightrag_mem()), _problem(), [])

    entity_items = [it for it in bundle.retrieved_items if it.get("type") == "entity"]
    edge_items = [it for it in bundle.retrieved_items if it.get("type") == "edge"]
    assert len(entity_items) >= 1
    assert len(edge_items) >= 1


def test_lightrag_empty_memory_returns_none():
    """Empty memory → no hint."""
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    r = LightRAGRetriever(top_k_entities=3, top_m_relationships=3)
    empty = ConceptMemory()
    bundle = r.retrieve(_ctx(), _make_ms(empty), _problem(), [])
    assert bundle.hint_text is None


def test_lightrag_no_edges_above_threshold():
    """With very high min_edge_weight, no relationships appear but entities do."""
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    r = LightRAGRetriever(top_k_entities=3, top_m_relationships=3, min_edge_weight=9999.0)
    bundle = r.retrieve(_ctx(), _make_ms(_lightrag_mem()), _problem(), [])

    hint = bundle.hint_text or ""
    assert "entities (local)" in hint
    edge_items = [it for it in bundle.retrieved_items if it.get("type") == "edge"]
    assert len(edge_items) == 0


def test_lightrag_uses_dual_dense_artifact_when_present(tmp_path: Path):
    """Embedding artifact should activate dense primary plus sparse secondary."""
    from mem2.branches.memory_retriever.lightrag import LightRAGRetriever

    mem = _lightrag_mem()
    npz_path, meta_path = _lightrag_embedding_artifact(tmp_path, mem)
    r = LightRAGRetriever(
        top_k_entities=3,
        top_m_relationships=3,
        min_edge_weight=0.5,
        embedding_npz_path=npz_path,
        embedding_meta_path=meta_path,
    )
    bundle = r.retrieve(_ctx(), _make_ms(mem), _problem(), [])

    assert bundle.metadata["dual_signal_used"] is True
    assert bundle.metadata["num_dense_candidates"] >= 1
    assert bundle.metadata["scoring_detail"] == "dense_primary_sparse_secondary"
