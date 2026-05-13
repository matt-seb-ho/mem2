"""RRMCInteractiveRetriever: multi-round probing with Coverage + convergence.

Axis C in the ablation plan. Compared against one-shot top-k retrievers.

Faithful simplification of ``workstation_00_RRMC`` for the Phase-1 axis-C
question ("does multi-round interactive retrieval help?"). The full RRMC
machinery (knowledge-graph priors, MI-estimator, structured probing) lives
elsewhere; porting it in total is out of scope for Phase-1. The core signal
— context division via coverage-gated iteration — is captured here.

Round structure:
  - Round 1: seed retrieval = top-k by co-activation degree.
  - Round t>1: pick concepts that maximize coverage (unseen concept kinds and
    unseen ``cues``), up to ``per_round_k``.
  - Stop when (a) coverage stops growing for ``convergence_patience`` rounds,
    (b) we reach ``max_rounds``, or (c) the candidate pool is exhausted.

This retriever is LLM-free at the retrieval step itself; it uses the graph
structure + concept metadata. Future RRMC variants can swap in the LLM-based
probing question generator; hook point is marked below.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
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

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "rrmc_memory_v1.json"


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


@dataclass(frozen=True, slots=True)
class CoverageState:
    kinds_seen: frozenset[str]
    cues_seen: frozenset[str]
    concepts_seen: frozenset[str]

    @property
    def score(self) -> int:
        return len(self.kinds_seen) + len(self.cues_seen)


class RRMCInteractiveRetriever:
    name = "rrmc_interactive"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        per_round_k: int = 3,
        max_rounds: int = 5,
        convergence_patience: int = 2,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
        adapted_memory_path: str | Path | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.per_round_k = int(per_round_k)
        self.max_rounds = int(max_rounds)
        self.convergence_patience = int(convergence_patience)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)
        self.adapted_memory_path = self._resolve_path(
            adapted_memory_path,
            _DEFAULT_ADAPTED_MEMORY_PATH,
        )

    # ----------------------------------------------------------------- #
    def retrieve(
        self,
        ctx: RunContext,
        memory: MemoryState,
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> RetrievalBundle:
        concept_mem = ConceptMemory.from_payload(memory.payload)
        if not concept_mem.concepts:
            return RetrievalBundle(
                problem_uid=problem.uid,
                hint_text=None,
                retrieved_items=[],
                metadata={"reason": "empty_memory"},
            )

        graph = ConceptGraph.build_from_memory(concept_mem, min_co_overlap=1)
        adapted_records, adapted_source = self._load_adapted_records(concept_mem)
        q_tokens = self._query_tokens(problem, previous_attempts)

        # Round 1 — seed: top-k by co-activation degree
        degrees = [
            (
                self._adapted_round_score(adapted_records.get(n), q_tokens, round_index=1)
                + graph.degree(n, kinds=["co_activation"]),
                n,
            )
            for n in graph.nodes
        ]
        degrees.sort(reverse=True)
        ordered_by_degree = [n for _, n in degrees]
        seen: set[str] = set()
        rounds: list[list[str]] = []

        def bump(names: list[str]) -> list[str]:
            added = []
            for n in names:
                if n in seen or n not in concept_mem.concepts:
                    continue
                seen.add(n)
                added.append(n)
                if len(seen) >= self.top_k:
                    break
            return added

        seed = bump(ordered_by_degree[: self.per_round_k])
        if seed:
            rounds.append(seed)

        # Coverage-driven rounds
        prev_cov_score = self._coverage_score(concept_mem, seen)
        patience = 0
        while (
            len(seen) < self.top_k
            and len(rounds) < self.max_rounds
            and patience < self.convergence_patience
        ):
            candidate = self._next_round_candidates(
                concept_mem,
                graph,
                seen,
                adapted_records,
                q_tokens,
            )
            added = bump(candidate)
            if not added:
                break
            rounds.append(added)
            cur_cov_score = self._coverage_score(concept_mem, seen)
            if cur_cov_score <= prev_cov_score:
                patience += 1
            else:
                patience = 0
            prev_cov_score = cur_cov_score

        names = list(seen)
        hint_text = self._render_adapted_hint(names, adapted_records) if adapted_records else ""
        if not hint_text:
            hint_text = concept_mem.to_string(
                concept_names=names,
                include_description=self.include_description,
                skip_cues=self.skip_cues,
                skip_implementation=self.skip_implementation,
                usage_threshold=self.usage_threshold,
            )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint_text or None,
            retrieved_items=[{"name": n} for n in names],
            metadata={
                "retriever": self.name,
                "scoring_mode": "rrmc_multi_round",
                "num_rounds": len(rounds),
                "per_round_counts": [len(r) for r in rounds],
                "rounds": rounds,
                "coverage_score": prev_cov_score,
                "converged": patience >= self.convergence_patience,
                "adapted_memory_source": adapted_source,
                "adapted_records_loaded": len(adapted_records),
                "adapted_selector_items_rendered": sum(1 for n in names if n in adapted_records),
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
    def _coverage_score(
        self, mem: ConceptMemory, names: set[str]
    ) -> int:
        kinds = set()
        cues = set()
        for n in names:
            c = mem.concepts.get(n)
            if not c:
                continue
            kinds.add(c.kind)
            for cue in c.cues:
                cues.add(cue)
        return len(kinds) + len(cues)

    def _next_round_candidates(
        self,
        mem: ConceptMemory,
        graph: ConceptGraph,
        seen: set[str],
        adapted_records: dict[str, dict],
        q_tokens: set[str],
    ) -> list[str]:
        """Rank unseen concepts by (unseen-kinds-contributed, unseen-cues-contributed),
        breaking ties by graph degree.

        This is the "Coverage" signal — we prefer candidates that bring NEW
        concept kinds or NEW cues over ones that merely duplicate what's
        already in context.
        """
        kinds_seen: Counter[str] = Counter()
        cues_seen: set[str] = set()
        for n in seen:
            c = mem.concepts.get(n)
            if not c:
                continue
            kinds_seen[c.kind] += 1
            cues_seen.update(c.cues)

        scored: list[tuple[tuple[int, int, float, float], str]] = []
        for name, c in mem.concepts.items():
            if name in seen:
                continue
            new_kind = 1 if kinds_seen.get(c.kind, 0) == 0 else 0
            new_cues = sum(1 for cue in c.cues if cue not in cues_seen)
            degree = graph.degree(name, kinds=["co_activation"])
            adapted_score = self._adapted_round_score(
                adapted_records.get(name),
                q_tokens,
                round_index=2,
            )
            scored.append(((new_kind, new_cues, adapted_score, degree), name))
        scored.sort(key=lambda r: r[0], reverse=True)
        return [name for _, name in scored[: self.per_round_k]]

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
            raise RuntimeError(f"invalid RRMC adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != self.name:
            raise RuntimeError(f"invalid RRMC adapted memory schema: {path}")
        records: dict[str, dict] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            probes = raw.get("probe_plan")
            if not isinstance(probes, list) or len(probes) < 2:
                raise RuntimeError(f"adapted memory missing probe_plan for {concept_id}")
            records[concept_id] = raw
        if not records:
            return {}, "flat"
        return records, "rrmc_memory_v1"

    @staticmethod
    def _query_tokens(
        problem: ProblemSpec,
        previous_attempts: list[AttemptRecord],
    ) -> set[str]:
        parts: list[str] = [str(getattr(problem, "uid", ""))]
        meta = getattr(problem, "metadata", {}) or {}
        for value in meta.values():
            if isinstance(value, str):
                parts.append(value)
        for attempt in previous_attempts or []:
            if getattr(attempt, "error", None):
                parts.append(str(attempt.error))
        return _tokenize(" ".join(parts))

    def _adapted_round_score(
        self,
        record: dict | None,
        q_tokens: set[str],
        *,
        round_index: int,
    ) -> float:
        if not record:
            return 0.0
        key = "round_1_relevance" if round_index <= 1 else "round_2_relevance"
        try:
            relevance = float(record.get(key, 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        overlap = len(q_tokens & _tokenize(self._adapted_record_text(record))) if q_tokens else 0
        return relevance + float(overlap)

    @staticmethod
    def _adapted_record_text(record: dict) -> str:
        parts: list[str] = [
            str(record.get("selector_role") or ""),
            str(record.get("convergence_signal") or ""),
            str(record.get("retrieval_notes") or ""),
        ]
        parts.extend(str(item) for item in record.get("coverage_targets") or [])
        parts.extend(str(item) for item in record.get("routing_keywords") or [])
        for probe in record.get("probe_plan") or []:
            if isinstance(probe, dict):
                parts.extend(
                    str(probe.get(key) or "")
                    for key in ("probe_question", "expected_signal", "selector_update")
                )
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _render_adapted_hint(
        names: list[str],
        adapted_records: dict[str, dict],
    ) -> str:
        blocks: list[str] = []
        for name in names:
            record = adapted_records.get(name)
            if not record:
                continue
            lines = [f"- concept: {name}"]
            lines.append(f"  rrmc_selector_role: {record.get('selector_role', 'other')}")
            lines.append(
                "  round_relevance: "
                f"r1={float(record.get('round_1_relevance', 0.0)):.2f}, "
                f"r2={float(record.get('round_2_relevance', 0.0)):.2f}"
            )
            targets = [str(t).strip() for t in record.get("coverage_targets") or [] if str(t).strip()]
            if targets:
                lines.append("  coverage_targets: " + ", ".join(targets[:5]))
            for probe in record.get("probe_plan") or []:
                if not isinstance(probe, dict):
                    continue
                round_id = probe.get("round", "?")
                question = str(probe.get("probe_question") or "").strip()
                if question:
                    lines.append(f"  round_{round_id}_probe: {question}")
            convergence = str(record.get("convergence_signal") or "").strip()
            if convergence:
                lines.append(f"  convergence_signal: {convergence}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
