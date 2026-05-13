"""LightRAG dual-level retriever - axis B.7.

Port of the dual-level retrieval pattern from LightRAG (Guo et al.).

Paper: literature/2410.05779.pdf
Repo:  third_party/lightrag/ (entry: lightrag/operate.py::kg_query)

Specifically ported:
    - Dual-level retrieval: "local" (entity-level) + "global" (relationship-level),
      merged into a single context.

Deliberate simplifications (LLM-free, embed-free):
    - Entity scoring: paper uses entity-embedding cosine sim -> token overlap.
    - Relationship scoring: paper uses per-edge LLM summary + embedding sim ->
      edge score = product of endpoint token-overlap times edge weight.
    - No vector DBs, no LLM summary generation.
    - Hint shape: two blocks, `## entities (local)` + `## relationships (global)`,
      distinguishing this axis-B condition from node-only or community-only
      retrievers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from mem2.concepts.artifacts import CONCEPT_MEMORY_DIR
from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_EMB_NPZ = _REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "lightrag_embed_v1.npz"
_DEFAULT_EMB_META = _REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "shared" / "lightrag_embed_v1.json"
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "lightrag_memory_v1.json"


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


def _problem_text(problem: ProblemSpec) -> str:
    parts: list[str] = []
    if problem.metadata:
        parts.extend(v for v in problem.metadata.values() if isinstance(v, str))
    for p in (problem.train_pairs or []):
        if isinstance(p, dict):
            parts.extend(v for v in p.values() if isinstance(v, str))
    for p in (problem.test_pairs or []):
        if isinstance(p, dict):
            parts.extend(v for v in p.values() if isinstance(v, str))
    parts.append(problem.uid)
    return "\n".join(parts)


class LightRAGRetriever:
    """Dual-level: top-k entities (concepts) + top-m relationships (edges)."""

    name = "lightrag"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k_entities: int = 3,
        top_m_relationships: int = 3,
        min_edge_weight: float = 1.0,
        embedding_npz_path: str | Path | None = None,
        embedding_meta_path: str | Path | None = None,
        adapted_memory_path: str | Path | None = None,
    ) -> None:
        self.top_k_entities = int(top_k_entities)
        self.top_m_relationships = int(top_m_relationships)
        self.min_edge_weight = float(min_edge_weight)
        self.embedding_npz_path = Path(embedding_npz_path) if embedding_npz_path else _DEFAULT_EMB_NPZ
        self.embedding_meta_path = Path(embedding_meta_path) if embedding_meta_path else _DEFAULT_EMB_META
        self.adapted_memory_path = self._resolve_path(
            adapted_memory_path,
            _DEFAULT_ADAPTED_MEMORY_PATH,
        )

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        mem = ConceptMemory.from_payload(memory.payload)
        if not mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid, hint_text=None, retrieved_items=[],
                metadata={"retriever": self.name, "reason": "empty_memory"},
            )

        q_tokens = _tokenize(_problem_text(problem))
        adapted_records, adapted_source = self._load_adapted_records(mem)

        # --- Local (entity-level) scoring -------------------------------
        token_scores: dict[str, int] = {}
        for name, concept in mem.concepts.items():
            c_tokens = _tokenize(concept.to_string(include_description=True))
            token_scores[name] = len(q_tokens & c_tokens)

        dense_scores, dense_meta = self._dense_entity_scores(mem, q_tokens)
        entity_scores: dict[str, float] = {}
        if dense_scores:
            for name in mem.concepts:
                entity_scores[name] = (
                    dense_scores.get(name, 0.0)
                    + 0.05 * token_scores.get(name, 0)
                    + self._adapted_local_score(adapted_records.get(name), q_tokens)
                )
            ranked_entities = sorted(entity_scores.items(), key=lambda kv: kv[1], reverse=True)
            scoring_detail = "dense_primary_sparse_secondary"
        else:
            entity_scores = {
                name: float(score) + self._adapted_local_score(adapted_records.get(name), q_tokens)
                for name, score in token_scores.items()
            }
            ranked_entities = sorted(entity_scores.items(), key=lambda kv: kv[1], reverse=True)
            scoring_detail = "token_overlap_fallback"
        top_entities = [name for name, _ in ranked_entities[: self.top_k_entities]]

        # --- Global (relationship-level) scoring ------------------------
        graph = ConceptGraph.build_from_memory(
            mem,
            min_co_overlap=1,
            load_openie_edges=True,
            load_entity_edges=True,
        )
        edge_scores: list[tuple[str, str, float, str, str]] = []
        for edge in graph.edges():
            if edge.kind not in {"co_activation", "openie_fact", "entity_relation"}:
                continue
            if edge.kind == "co_activation" and edge.weight < self.min_edge_weight:
                continue
            e1_score = entity_scores.get(edge.src, 0.0)
            e2_score = entity_scores.get(edge.dst, 0.0)
            if e1_score == 0 and e2_score == 0:
                continue
            # product of endpoint relevance times edge weight; add 1 to endpoint
            # scores to avoid killing asymmetric edges where one endpoint
            # has zero overlap.
            combined = float((e1_score + 1) * (e2_score + 1) * edge.weight)
            combined += self._adapted_relationship_score(
                adapted_records.get(edge.src),
                edge.dst,
                q_tokens,
            )
            combined += self._adapted_relationship_score(
                adapted_records.get(edge.dst),
                edge.src,
                q_tokens,
            )
            label = self._relationship_label(edge)
            edge_scores.append((edge.src, edge.dst, combined, edge.kind, label))
        edge_scores.sort(key=lambda t: t[2], reverse=True)
        top_edges = edge_scores[: self.top_m_relationships]

        # --- Render hint ------------------------------------------------
        if adapted_records:
            hint = self._render_adapted_hint(mem, top_entities, top_edges, adapted_records)
        else:
            lines: list[str] = []
            if top_entities:
                lines.append("## entities (local)")
                for name in top_entities:
                    c = mem.concepts.get(name)
                    desc = (c.description or "") if c else ""
                    lines.append(f"- {name}: {desc}" if desc else f"- {name}")
            if top_edges:
                lines.append("")
                lines.append("## relationships (global)")
                for src, dst, score, kind, label in top_edges:
                    lines.append(f"- {src} --{label}-- {dst}  ({kind} score {score:.1f})")
            hint = "\n".join(lines) if lines else None

        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint,
            retrieved_items=[
                *[{"type": "entity", "name": n} for n in top_entities],
                *[
                    {"type": "edge", "src": s, "dst": d, "score": sc, "edge_kind": k, "label": label}
                    for s, d, sc, k, label in top_edges
                ],
            ],
            metadata={
                "retriever": self.name,
                "scoring_mode": "lightrag_dual",
                "scoring_detail": scoring_detail,
                "dual_signal_used": bool(dense_scores and edge_scores),
                "num_dense_candidates": int(dense_meta.get("num_dense_candidates", 0)),
                "num_dense_entity_hits": int(dense_meta.get("num_dense_entity_hits", 0)),
                "top_k_entities": self.top_k_entities,
                "top_m_relationships": self.top_m_relationships,
                "num_concepts_total": len(mem.concepts),
                "num_edges_considered": len(graph.edges()),
                "num_edges_ranked": len(edge_scores),
                "num_entities_selected": len(top_entities),
                "num_edges_selected": len(top_edges),
                "adapted_memory_source": adapted_source,
                "adapted_records_loaded": len(adapted_records),
                "adapted_local_items_rendered": sum(1 for n in top_entities if n in adapted_records),
                "adapted_relationship_items_rendered": sum(
                    1
                    for src, dst, *_ in top_edges
                    if src in adapted_records or dst in adapted_records
                ),
            },
        )

    async def async_retrieve(
        self,
        *,
        ctx: RunContext,
        provider,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
        selector_model: str = "",
    ) -> RetrievalBundle:
        return self.retrieve(ctx, memory, problem, previous_attempts)

    def _relationship_label(self, edge) -> str:
        if edge.kind == "entity_relation":
            edge_type = str((edge.metadata or {}).get("edge_type") or "").strip()
            if edge_type:
                return edge_type
        if edge.kind == "openie_fact":
            predicate = str((edge.metadata or {}).get("predicate") or "").strip()
            if predicate:
                return predicate
        return "co-activates-with"

    @staticmethod
    def _resolve_path(path: str | Path | None, default: Path) -> Path:
        if path is None:
            return default
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _load_adapted_records(
        self,
        mem: ConceptMemory,
    ) -> tuple[dict[str, dict], str]:
        path = self.adapted_memory_path
        if not path.exists():
            return {}, "flat"
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - corrupted local artifact should not be silent
            raise RuntimeError(f"invalid LightRAG adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != self.name:
            raise RuntimeError(f"invalid LightRAG adapted memory schema: {path}")
        records: dict[str, dict] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            if "local_entities" not in raw or "global_relationships" not in raw:
                raise RuntimeError(f"adapted memory missing LightRAG fields for {concept_id}")
            records[concept_id] = raw
        if not records:
            return {}, "flat"
        return records, "lightrag_memory_v1"

    def _adapted_local_score(
        self,
        record: dict | None,
        q_tokens: set[str],
    ) -> float:
        if not record:
            return 0.0
        text = self._adapted_record_text(record, local_only=True)
        return float(len(q_tokens & _tokenize(text))) + 0.1 * len(record.get("local_entities") or [])

    def _adapted_relationship_score(
        self,
        record: dict | None,
        target: str,
        q_tokens: set[str],
    ) -> float:
        if not record:
            return 0.0
        score = 0.0
        target_tokens = _tokenize(target)
        for relationship in record.get("global_relationships") or []:
            if not isinstance(relationship, dict):
                continue
            rel_text = " ".join(
                str(relationship.get(key) or "")
                for key in ("relation", "target_concept", "relation_summary")
            )
            rel_tokens = _tokenize(rel_text)
            if target_tokens & rel_tokens:
                score += 1.0
            score += float(len(q_tokens & rel_tokens))
            try:
                score += float(relationship.get("strength", 0.0))
            except (TypeError, ValueError):
                pass
        return score

    @staticmethod
    def _adapted_record_text(record: dict, *, local_only: bool = False) -> str:
        parts: list[str] = [
            str(record.get("entity_value_summary") or ""),
            str(record.get("chunk_reference") or ""),
            str(record.get("retrieval_notes") or ""),
        ]
        parts.extend(str(item) for item in record.get("low_level_keywords") or [])
        for entity in record.get("local_entities") or []:
            if isinstance(entity, dict):
                parts.extend(
                    str(entity.get(key) or "")
                    for key in ("mention", "entity_type", "entity_summary")
                )
        if not local_only:
            parts.append(str(record.get("relation_value_summary") or ""))
            parts.extend(str(item) for item in record.get("high_level_keywords") or [])
            parts.extend(str(item) for item in record.get("one_hop_neighbors") or [])
            for relationship in record.get("global_relationships") or []:
                if isinstance(relationship, dict):
                    parts.extend(
                        str(relationship.get(key) or "")
                        for key in ("relation", "target_concept", "relation_summary")
                    )
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _render_adapted_hint(
        mem: ConceptMemory,
        top_entities: list[str],
        top_edges: list[tuple[str, str, float, str, str]],
        adapted_records: dict[str, dict],
    ) -> str | None:
        lines: list[str] = []
        if top_entities:
            lines.append("## entities (local)")
            for name in top_entities:
                record = adapted_records.get(name)
                if not record:
                    c = mem.concepts.get(name)
                    desc = (c.description or "") if c else ""
                    lines.append(f"- {name}: {desc}" if desc else f"- {name}")
                    continue
                local_mentions = [
                    str(entity.get("mention") or "").strip()
                    for entity in record.get("local_entities") or []
                    if isinstance(entity, dict) and str(entity.get("mention") or "").strip()
                ]
                keywords = [
                    str(item).strip()
                    for item in record.get("low_level_keywords") or []
                    if str(item).strip()
                ]
                lines.append(f"- {name}")
                lines.append("  lightrag_local_entities: " + ", ".join(local_mentions[:6]))
                if keywords:
                    lines.append("  low_level_keywords: " + ", ".join(keywords[:6]))
                summary = str(record.get("entity_value_summary") or "").strip()
                if summary:
                    lines.append(f"  entity_value_summary: {summary}")
        if top_edges:
            lines.append("")
            lines.append("## relationships (global)")
            for src, dst, score, kind, label in top_edges:
                lines.append(f"- {src} --{label}-- {dst}  ({kind} score {score:.1f})")
                for concept_name, target in ((src, dst), (dst, src)):
                    record = adapted_records.get(concept_name)
                    if not record:
                        continue
                    rendered = False
                    for relationship in record.get("global_relationships") or []:
                        if not isinstance(relationship, dict):
                            continue
                        target_text = str(relationship.get("target_concept") or "")
                        if target not in target_text and not (_tokenize(target) & _tokenize(target_text)):
                            continue
                        rel = str(relationship.get("relation") or "").strip()
                        summary = str(relationship.get("relation_summary") or "").strip()
                        lines.append(f"  lightrag_global_relationships: {concept_name} {rel} {target_text}")
                        if summary:
                            lines.append(f"  relation_value_summary: {summary}")
                        rendered = True
                        break
                    if rendered:
                        break
        return "\n".join(lines) if lines else None

    def _dense_entity_scores(self, mem: ConceptMemory, q_tokens: set[str]) -> tuple[dict[str, float], dict]:
        if not q_tokens:
            return {}, {"num_dense_candidates": 0, "num_dense_entity_hits": 0}
        if not self.embedding_npz_path.exists() or not self.embedding_meta_path.exists():
            return {}, {"num_dense_candidates": 0, "num_dense_entity_hits": 0}
        try:
            import numpy as np

            meta = json.loads(self.embedding_meta_path.read_text())
            npz = np.load(self.embedding_npz_path)
            concept_ids = list(meta.get("concept_ids") or [])
            entity_sources = list(meta.get("entity_sources") or [])
            entity_mentions = list(meta.get("entity_mentions") or [])
            entity_types = list(meta.get("entity_types") or [])
            concept_embeddings = np.asarray(npz["concept_embeddings"], dtype=np.float32)
            entity_embeddings = np.asarray(npz["entity_embeddings"], dtype=np.float32)
        except Exception:
            return {}, {"num_dense_candidates": 0, "num_dense_entity_hits": 0}

        seed_vecs: list = []
        for idx, name in enumerate(concept_ids):
            if idx >= len(concept_embeddings) or name not in mem.concepts:
                continue
            concept = mem.concepts[name]
            text = concept.to_string(include_description=True)
            overlap = len(q_tokens & _tokenize(text))
            if overlap:
                seed_vecs.append(concept_embeddings[idx] * float(overlap))
        for idx, source in enumerate(entity_sources):
            if idx >= len(entity_embeddings):
                continue
            if source not in mem.concepts:
                continue
            text = " ".join([
                str(entity_mentions[idx] if idx < len(entity_mentions) else ""),
                str(entity_types[idx] if idx < len(entity_types) else ""),
                source,
            ])
            overlap = len(q_tokens & _tokenize(text))
            if overlap:
                seed_vecs.append(entity_embeddings[idx] * float(overlap))
        if not seed_vecs:
            return {}, {"num_dense_candidates": 0, "num_dense_entity_hits": 0}
        query = np.sum(np.stack(seed_vecs), axis=0)
        norm = np.linalg.norm(query)
        if not norm:
            return {}, {"num_dense_candidates": 0, "num_dense_entity_hits": 0}
        query = (query / norm).astype(np.float32)

        scores: dict[str, float] = {}
        concept_scores = concept_embeddings @ query
        for idx, score in enumerate(concept_scores):
            if idx < len(concept_ids) and concept_ids[idx] in mem.concepts:
                scores[concept_ids[idx]] = max(scores.get(concept_ids[idx], 0.0), float(score))
        entity_scores = entity_embeddings @ query
        dense_entity_hits = 0
        for idx, score in enumerate(entity_scores):
            if idx < len(entity_sources) and entity_sources[idx] in mem.concepts:
                dense_entity_hits += 1
                source = entity_sources[idx]
                scores[source] = max(scores.get(source, 0.0), float(score))
        return scores, {
            "num_dense_candidates": len(scores),
            "num_dense_entity_hits": dense_entity_hits,
        }
