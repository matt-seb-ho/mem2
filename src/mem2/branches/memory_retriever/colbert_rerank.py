"""ColBERT late-interaction lexical rerank — axis B.8.

Port of ColBERT (Khattab & Zaharia, SIGIR'20; arxiv 2004.12832).

Paper: literature/2004.12832.pdf
Repo:  third_party/colbert/ (entry: colbert/modeling/colbert.py::colbert_score_reduce + MaxSim)

Specifically ported:
    - The *MaxSim* aggregation (`colbert_score_reduce`): for each query token,
      take the max similarity against all document tokens; sum these per-query
      maxes. This is the late-interaction distinctive scoring pattern.
    - The two-stage pipeline: first-stage wide retrieval + second-stage MaxSim
      rerank over a short candidate list.

Embedding mode (preferred, when prereq file exists):
    - Loads `data/arc_agi/concept_memory/concept_embeddings_v1.npz` (built
      by `scripts/prereq/axis_1/colbert_rerank/build_concept_embeddings.py`).
    - Concept similarity = cosine(query_embedding, concept_embedding) using
      the sentence-transformer that produced the prereq file.
    - This is concept-level (one vec/concept), not full ColBERT's token-level
      MaxSim — a deliberate simplification documented in doc 52 as Reduced.

Deliberate simplifications when prereq file is absent (token-overlap fallback):
    - Tokens here are lowercase words (token overlap signal), NOT BERT-contextual
      embeddings. "Similarity" = exact token match (1.0) or not (0.0). This
      reduces MaxSim to a set-intersection: for each query token, 1 if the
      document contains it, else 0. The SUM then counts unique query tokens
      covered by the document — a standard unigram coverage score that shares
      the MaxSim algebraic structure (per-query-token max aggregation).
    - TF-IDF weighting is NOT added — it would confuse the ablation ("is
      this token-level matching or TF-IDF?"). The distinctive mechanic is
      the per-query-token MAX (not sum), which rewards coverage over
      repetition.
    - First-stage retrieval is `ps_topk` with `top_k * expansion_factor`;
      second-stage reranks to final `top_k`. Identity with ps_topk when
      `expansion_factor == 1` (sanity check).

B.8 vs B.1-B.7:
    - B.1 ps_topk: frequency ranking only.
    - B.2 graph_traversal: BFS on graph.
    - B.4 hipporag_ppr: PPR from query seeds.
    - B.6 raptor: community summaries + leaves.
    - B.3 graphrag: community summaries only.
    - B.7 lightrag: entities + edges dual.
    - B.8 (this module): *token-level MaxSim late-interaction* — a different
      scoring algebra (per-query-term max, not cumulative frequency or
      structural).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

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

# Resolve repo root once (mem2/) for prereq file lookup.
# parents[0]=memory_retriever, [1]=branches, [2]=mem2, [3]=src, [4]=mem2 root
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]  # mem2/

_DEFAULT_EMB_PATH = _REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "concept_embeddings_v1.npz"
_DEFAULT_EMB_META = _REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "concept_embeddings_v1.meta.json"


# Lazy-loaded embedding cache. Shared across retriever instances within a
# process; the embedding matrix and the sentence-transformer model are
# loaded at most once.
_EMB_CACHE: dict | None = None


def _load_embedding_cache() -> dict | None:
    """Load embedding matrix + sentence-transformer model on first use.

    Returns a dict with keys:
      - "names": np.array of concept names (sorted, shape (N,))
      - "name_to_idx": {name: int}
      - "embeddings": (N, dim) float32 normalized
      - "model": SentenceTransformer instance for query encoding
      - "model_id": str
    Returns None on any failure (missing file, missing dep, etc.).
    """
    global _EMB_CACHE
    if _EMB_CACHE is not None:
        return _EMB_CACHE if _EMB_CACHE.get("ok") else None

    sentinel = {"ok": False}
    if not _DEFAULT_EMB_PATH.exists():
        _EMB_CACHE = sentinel
        return None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("colbert_rerank: numpy or sentence-transformers missing → token-overlap fallback")
        _EMB_CACHE = sentinel
        return None

    try:
        npz = np.load(_DEFAULT_EMB_PATH, allow_pickle=True)
        names = list(npz["names"])
        embeddings = npz["embeddings"]
        meta = {}
        if _DEFAULT_EMB_META.exists():
            meta = json.loads(_DEFAULT_EMB_META.read_text())
        model_id = meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        device = "cpu"  # query encoding is small; stick to CPU to avoid MPS init churn
        model = SentenceTransformer(model_id, device=device)
        _EMB_CACHE = {
            "ok": True,
            "names": names,
            "name_to_idx": {n: i for i, n in enumerate(names)},
            "embeddings": embeddings,
            "model": model,
            "model_id": model_id,
        }
        logger.info(f"colbert_rerank: loaded embeddings from {_DEFAULT_EMB_PATH.name} ({len(names)} concepts, dim={embeddings.shape[1]}, model={model_id})")
        return _EMB_CACHE
    except Exception as e:
        logger.warning(f"colbert_rerank: failed to load embeddings → fallback. Error: {e}")
        _EMB_CACHE = sentinel
        return None


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok.lower() for tok in WORD_RE.findall(text)]


def _maxsim_score(query_toks: list[str], doc_toks: Iterable[str]) -> float:
    """Lexical MaxSim: for each query token, 1.0 if any doc token matches, else 0.
    Sum over query tokens. This is the SUM-of-MAX-over-doc-tokens reduction
    from `colbert_score_reduce` — simplified to exact-match similarity.
    """
    if not query_toks:
        return 0.0
    doc_set = set(doc_toks)
    return sum(1.0 if qt in doc_set else 0.0 for qt in query_toks)


class ColBERTRerankRetriever:
    """Two-stage: ps_topk-style wide retrieval, then MaxSim rerank."""

    name = "colbert_rerank"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        expansion_factor: int = 4,
        usage_threshold: int = 0,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        skip_parameters: bool = False,
        skip_parameter_description: bool = True,
    ) -> None:
        self.top_k = int(top_k)
        self.expansion_factor = max(1, int(expansion_factor))
        self.usage_threshold = int(usage_threshold)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.skip_parameters = bool(skip_parameters)
        self.skip_parameter_description = bool(skip_parameter_description)

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
                problem_uid=problem.uid, hint_text=None,
                retrieved_items=[],
                metadata={"retriever": self.name, "stage": "empty"},
            )

        # Stage 1 — wide frequency ranking (ps_topk-style).
        first_stage = sorted(
            mem.concepts.values(),
            key=lambda c: (len(c.used_in), c.name), reverse=True,
        )
        first_stage_top = first_stage[: max(self.top_k * self.expansion_factor, 0)]

        # Stage 2 — MaxSim rerank over the expanded candidate pool.
        query_text = self._build_query_text(problem, previous_attempts)

        cache = _load_embedding_cache()
        scoring_mode: str
        if cache is not None:
            # Embedding mode: cosine similarity between query and concept vectors.
            try:
                import numpy as np
                q_vec = cache["model"].encode(
                    [query_text], normalize_embeddings=True, show_progress_bar=False,
                )[0].astype(np.float32)
                emb = cache["embeddings"]
                name_to_idx = cache["name_to_idx"]
                scored: list[tuple[float, str]] = []
                for c in first_stage_top:
                    idx = name_to_idx.get(c.name)
                    if idx is None:
                        # Concept not in pre-built embeddings → use 0 (effectively
                        # ranks it after embedded ones). New concepts (e.g.
                        # post-reorg aggregates) hit this branch.
                        scored.append((0.0, c.name))
                    else:
                        # cosine sim — both are L2-normalized
                        s = float(np.dot(q_vec, emb[idx]))
                        scored.append((s, c.name))
                scoring_mode = "embedding_cosine"
            except Exception as e:
                logger.warning(f"colbert_rerank: embedding mode failed mid-call → token fallback. Error: {e}")
                cache = None  # force fallback path below

        if cache is None:
            # Token-overlap fallback (original behavior).
            query_toks = _tokenize(query_text)
            scored = []
            for c in first_stage_top:
                doc_toks = self._concept_tokens(c)
                s = _maxsim_score(query_toks, doc_toks)
                scored.append((s, c.name))
            scoring_mode = "token_maxsim"

        scored.sort(reverse=True)  # by (score, name) descending

        top = [name for _, name in scored[: max(self.top_k, 0)]]
        hint = mem.to_string(
            concept_names=top,
            include_description=self.include_description,
            skip_cues=self.skip_cues,
            skip_implementation=self.skip_implementation,
            skip_parameters=self.skip_parameters,
            skip_parameter_description=self.skip_parameter_description,
            usage_threshold=self.usage_threshold,
        )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in top],
            metadata={
                "retriever": self.name,
                "top_k": self.top_k,
                "expansion_factor": self.expansion_factor,
                "first_stage_pool": len(first_stage_top),
                "scoring_mode": scoring_mode,
                "num_concepts_total": len(mem.concepts),
                "num_selected": len(top),
                "max_score": scored[0][0] if scored else 0.0,
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

    # ----------------------------------------------------------------- #
    def _build_query_text(
        self, problem: ProblemSpec, previous_attempts: list[AttemptRecord],
    ) -> str:
        """Assemble the query representation: problem text + prior-attempt
        feedback if available. Mirrors ColBERT's query-side tokenization
        (content from the reasoning task)."""
        parts: list[str] = []
        parts.append(str(getattr(problem, "uid", "")))
        meta = getattr(problem, "metadata", {}) or {}
        for key in ("description", "instructions", "prompt", "query"):
            if key in meta and meta[key]:
                parts.append(str(meta[key]))
        # Include last feedback if present
        if previous_attempts:
            last = previous_attempts[-1]
            last_meta = getattr(last, "metadata", {}) or {}
            for key in ("feedback", "text"):
                if key in last_meta:
                    parts.append(str(last_meta[key]))
        return " \n ".join(parts)

    def _concept_tokens(self, concept) -> list[str]:
        """Concept-side token bag: name + description + cues + triggers."""
        pieces: list[str] = [concept.name]
        if concept.description:
            pieces.append(concept.description)
        for cue in getattr(concept, "cues", []) or []:
            if isinstance(cue, str):
                pieces.append(cue)
        for trig in getattr(concept, "triggers", []) or []:
            if isinstance(trig, str):
                pieces.append(trig)
        return _tokenize(" ".join(pieces))
