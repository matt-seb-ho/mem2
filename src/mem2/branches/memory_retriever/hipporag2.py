"""HippoRAG 2 linking-score rerank over PPR — axis B.5.

Port of HippoRAG 2 (Gutiérrez, Su et al., 2025; arxiv 2502.14802).

Paper: literature/2502.14802.pdf
Repo:  third_party/hipporag/ (entry: src/hipporag/rerank.py::DSPyFilter.rerank)

Specifically ported:
    - The *two-stage retrieval* pattern: PPR produces a wide candidate pool
      (as in HippoRAG 1, already axis B.4), then a second-stage *LLM fact
      filter* (DSPyFilter) reranks and trims. HippoRAG 2's distinctive gain
      over HippoRAG 1 is the filter stage — without it, B.5 reduces to B.4.
    - The filter's top-N trim behavior (`len_after_rerank`).

Deliberate simplifications (LLM-optional):
    - The LLM DSPy filter is replaced by a template scorer when no
      `_meta_edit_provider` is wired: each PPR candidate is scored by
      query-token overlap (same MaxSim-style scoring as B.8 ColBERT, but
      applied AFTER PPR instead of first-stage). Template-mode B.5 is
      distinct from B.8 because B.8's first-stage pool is frequency-ranked
      (ps_topk-style) while B.5's is PPR-ranked.
    - When a provider IS wired, the filter call mimics DSPyFilter's
      signature: sends query + candidate list, expects a JSON of kept
      candidate-names back.

B.5 vs B.4 vs B.8:
    - B.4 HippoRAG 1: PPR-only, no filter.
    - B.5 HippoRAG 2 (this module): PPR + post-filter.
    - B.8 ColBERT: frequency + MaxSim filter.
    - The axis-B ablation cleanly isolates: (i) structural vs lexical
      first-stage (B.4 vs B.8), and (ii) whether a post-PPR filter adds
      signal (B.5 vs B.4).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever
from mem2.concepts.artifacts import CONCEPT_MEMORY_DIR, load_openie_facts
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"\w+")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_HIPPO2_ADAPTED_MEMORY_PATH = (
    CONCEPT_MEMORY_DIR / "ports" / "hipporag2_memory_v1.json"
)

HIPPO2_FILTER_SYSTEM = (
    "You are a fact filter. You will see a query and a list of candidate "
    "concepts (names + descriptions). Output the subset that is directly "
    "relevant. Respond with a JSON list of concept names only: "
    '["concept_a", "concept_b", ...]. Do not include unrelated ones.'
)


class HippoRAG2FilterRetriever(HippoRAGPPRRetriever):
    """PPR first stage + filter rerank (LLM or token-overlap)."""

    name = "hipporag2_filter"

    def __init__(
        self,
        *,
        first_stage_top_k: int = 10,
        top_k: int = 3,
        adapted_memory_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        # First stage runs PPR with the wider pool.
        kwargs["top_k"] = first_stage_top_k
        kwargs["adapted_memory_path"] = self._resolve_path(
            adapted_memory_path,
            _DEFAULT_HIPPO2_ADAPTED_MEMORY_PATH,
        )
        super().__init__(**kwargs)
        self.first_stage_top_k = int(first_stage_top_k)
        self.final_top_k = int(top_k)

    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        # Stage 1 — PPR (via parent).
        first_bundle = super().retrieve(ctx, memory, problem, previous_attempts)
        if not first_bundle.retrieved_items:
            return first_bundle

        candidates = [it["name"] for it in first_bundle.retrieved_items]
        mem = ConceptMemory.from_payload(memory.payload)
        facts_by_source = self._facts_by_source(mem)
        adapted_records, _ = self._load_adapted_records(mem)
        query_text = self._build_query_text(problem, previous_attempts)
        provider = self._resolve_provider(ctx)

        # Stage 2 — filter rerank
        if provider is not None:
            filtered = self._filter_via_llm(
                provider, query_text, candidates, mem, facts_by_source, adapted_records,
            ) or candidates  # fall back to PPR ordering on failure
            filter_method = "llm"
        else:
            filtered = self._filter_via_template(query_text, candidates, mem, facts_by_source)
            filter_method = "token_overlap"

        selected = filtered[: self.final_top_k]
        if adapted_records:
            hint = self._render_adapted_hint(selected, adapted_records)
            if not hint:
                hint = mem.to_string(
                    concept_names=selected,
                    include_description=self.include_description,
                    skip_cues=self.skip_cues,
                    skip_implementation=self.skip_implementation,
                    usage_threshold=self.usage_threshold,
                )
        else:
            hint = mem.to_string(
                concept_names=selected,
                include_description=self.include_description,
                skip_cues=self.skip_cues,
                skip_implementation=self.skip_implementation,
                usage_threshold=self.usage_threshold,
            )

        metadata = dict(first_bundle.metadata or {})
        metadata.update({
            "retriever": self.name,
            "scoring_mode": "hipporag2_filter",
            "first_stage_top_k": self.first_stage_top_k,
            "final_top_k": self.final_top_k,
            "filter_method": filter_method,
            "fact_aware_filter": bool(facts_by_source),
            "post_filter_pool": len(filtered),
            "num_selected": len(selected),
            "adapted_filter_cards_rendered": sum(1 for name in selected if name in adapted_records),
        })
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in selected],
            metadata=metadata,
        )

    # ----------------------------------------------------------------- #
    def _resolve_provider(self, ctx: RunContext):
        try:
            return (ctx.config or {}).get("_meta_edit_provider")
        except AttributeError:
            return None

    def _build_query_text(
        self, problem: ProblemSpec, previous_attempts: list[AttemptRecord],
    ) -> str:
        parts: list[str] = [str(getattr(problem, "uid", ""))]
        meta = getattr(problem, "metadata", {}) or {}
        for key in ("description", "instructions", "prompt", "query"):
            if meta.get(key):
                parts.append(str(meta[key]))
        return " \n ".join(parts)

    def _filter_via_llm(
        self,
        provider,
        query: str,
        candidates: list[str],
        mem: ConceptMemory,
        facts_by_source: dict[str, list[dict[str, Any]]],
        adapted_records: dict[str, dict[str, Any]],
    ) -> list[str] | None:
        cand_block = "\n".join(
            f"- {name}: {(mem.concepts[name].description or '')[:140]}"
            f"{self._render_candidate_facts(name, facts_by_source)}"
            f"{self._render_candidate_adapted_profile(name, adapted_records)}"
            for name in candidates if name in mem.concepts
        )
        prompt = (
            f"{HIPPO2_FILTER_SYSTEM}\n\n"
            f"Query: {query}\n\nCandidates:\n{cand_block}\n\n"
            "Output JSON list only."
        )
        try:
            completions = provider.generate(prompt, model=getattr(provider, "model", ""))
            raw = completions[0] if completions else "[]"
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return None
            # Preserve PPR ordering among the names the LLM kept.
            kept = [c for c in candidates if c in parsed]
            return kept if kept else None
        except Exception as exc:  # pragma: no cover
            logger.warning("hipporag2 LLM filter failed: %s", exc)
            return None

    def _filter_via_template(
        self,
        query: str,
        candidates: list[str],
        mem: ConceptMemory,
        facts_by_source: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """Token-overlap rerank. Preserves PPR order as tiebreak via index."""
        q_toks = {tok.lower() for tok in WORD_RE.findall(query)}
        adapted_records, _ = self._load_adapted_records(mem)
        scored: list[tuple[float, int, str]] = []
        for idx, name in enumerate(candidates):
            c = mem.concepts.get(name)
            if c is None:
                continue
            doc_toks = set()
            doc_toks.add(name.lower())
            if c.description:
                doc_toks.update(
                    tok.lower() for tok in WORD_RE.findall(c.description)
                )
            for cue in c.cues or []:
                doc_toks.update(tok.lower() for tok in WORD_RE.findall(cue))
            if name in adapted_records:
                doc_toks.update(
                    tok.lower()
                    for tok in WORD_RE.findall(self._adapted_record_text(adapted_records[name]))
                )
            for fact in facts_by_source.get(name, []):
                fact_text = " ".join(
                    str(fact.get(k) or "")
                    for k in ("subject", "predicate", "object", "supporting_text")
                )
                doc_toks.update(tok.lower() for tok in WORD_RE.findall(fact_text))
            overlap = len(q_toks & doc_toks) if q_toks else 0.0
            # Primary: overlap (descending). Secondary: PPR rank (index, ascending).
            scored.append((-overlap, idx, name))
        scored.sort()
        return [name for _, _, name in scored]

    @staticmethod
    def _resolve_path(path: str | Path | None, default: Path) -> Path:
        if path is None:
            return default
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _load_adapted_records(
        self,
        mem: ConceptMemory,
    ) -> tuple[dict[str, dict[str, Any]], str]:
        path = self.adapted_memory_path
        if not path.exists():
            return {}, "flat"
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"invalid HippoRAG2 adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != "hipporag2":
            raise RuntimeError(f"invalid HippoRAG2 adapted memory schema: {path}")
        records: dict[str, dict[str, Any]] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            passage = raw.get("ppr_passage")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            if not isinstance(passage, str) or not passage.strip():
                raise RuntimeError(f"adapted memory missing ppr_passage for {concept_id}")
            records[concept_id] = raw
        if not records:
            return {}, "flat"
        return records, "hipporag2_memory_v1"

    @staticmethod
    def _adapted_record_text(record: dict[str, Any]) -> str:
        parts: list[str] = [
            str(record.get("ppr_passage") or ""),
            str(record.get("candidate_profile") or ""),
            str(record.get("rerank_notes") or ""),
            " ".join(str(t) for t in record.get("query_filter_terms") or []),
            " ".join(str(t) for t in record.get("reject_signals") or []),
        ]
        for evidence in record.get("filter_evidence") or []:
            if isinstance(evidence, dict):
                parts.append(" ".join([
                    str(evidence.get("claim") or ""),
                    str(evidence.get("supporting_text") or ""),
                    str(evidence.get("specificity") or ""),
                ]))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _render_adapted_hint(
        top_names: list[str],
        adapted_records: dict[str, dict[str, Any]],
    ) -> str:
        blocks: list[str] = []
        for name in top_names:
            record = adapted_records.get(name)
            if not record:
                continue
            lines = [f"- concept: {name}"]
            passage = str(record.get("ppr_passage") or "").strip()
            if passage:
                lines.append(f"  hipporag2_ppr_passage: {passage}")
            profile = str(record.get("candidate_profile") or "").strip()
            if profile:
                lines.append(f"  filter_profile: {profile}")
            terms = [str(t).strip() for t in (record.get("query_filter_terms") or []) if str(t).strip()]
            if terms:
                lines.append("  filter_terms: " + ", ".join(terms[:8]))
            evidence_lines: list[str] = []
            for evidence in record.get("filter_evidence") or []:
                if isinstance(evidence, dict) and evidence.get("claim"):
                    evidence_lines.append(str(evidence["claim"]).strip())
            if evidence_lines:
                lines.append("  filter_evidence: " + "; ".join(evidence_lines[:5]))
            rejects = [str(t).strip() for t in (record.get("reject_signals") or []) if str(t).strip()]
            if rejects:
                lines.append("  reject_signals: " + "; ".join(rejects[:4]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _render_candidate_adapted_profile(
        name: str,
        adapted_records: dict[str, dict[str, Any]],
    ) -> str:
        record = adapted_records.get(name)
        if not record:
            return ""
        parts = [
            str(record.get("candidate_profile") or ""),
            " ".join(str(t) for t in record.get("query_filter_terms") or []),
        ]
        return " | adapted: " + " ".join(part for part in parts if part).strip()[:260]

    def _facts_by_source(self, mem: ConceptMemory) -> dict[str, list[dict[str, Any]]]:
        facts_by_source: dict[str, list[dict[str, Any]]] = {}
        for fact in load_openie_facts(valid_concepts=mem.concepts.keys()):
            source = fact.get("source_concept")
            if isinstance(source, str) and source in mem.concepts:
                facts_by_source.setdefault(source, []).append(fact)
        return facts_by_source

    def _render_candidate_facts(
        self, name: str, facts_by_source: dict[str, list[dict[str, Any]]],
    ) -> str:
        facts = facts_by_source.get(name, [])[:3]
        if not facts:
            return ""
        rendered = []
        for fact in facts:
            rendered.append(
                f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
            )
        return " | facts: " + "; ".join(rendered)
