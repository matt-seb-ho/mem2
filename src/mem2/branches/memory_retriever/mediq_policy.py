"""MediQ abstention-gated retriever — axis C.4.

Port of the abstention-gated interactive pattern from MediQ (Li et al., NeurIPS'24).

Paper: literature/2406.00922.pdf
Repo:  third_party/mediq/ (entry: src/expert.py::Expert + expert_functions.py::fixed_abstention_decision)

Specifically ported:
    - The multi-round "ask-or-commit" scheduling pattern where each round
      grows the context and an abstention gate decides whether to continue
      accumulating or commit to the current set.

Deliberate simplifications:
    - LLM-based confidence elicitation → rolling-window **coverage**
      improvement. If the last `window` rounds have not increased
      (unique_kinds + unique_cues) by at least `abstention_threshold`,
      abstain. Keeps retrieval deterministic-on-seed + LLM-free.
    - No patient/expert role separation.
    - No LLM confidence logprobs (paper's implicit-abstention strategy).

Distinct from `rrmc_interactive`:
    - RRMC uses "patience" (symmetric coverage patience).
    - MediQ here uses an explicit abstention threshold — tighter control
      over when to commit.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.artifacts import CONCEPT_MEMORY_DIR
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
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "mediq_memory_v1.json"


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


class MediQPolicyRetriever:
    """Abstention-gated multi-round concept retrieval.

    Round 1 = seed with top-`per_round_k` concepts by co-activation degree.
    Round r>1 = next `per_round_k` concepts that bring new kind/cue coverage.
    After each round, compute coverage gain over the prior `window` rounds.
    Abstain when gain < `abstention_threshold` (commit to current set).
    """

    name = "mediq_policy"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        per_round_k: int = 2,
        max_rounds: int = 5,
        abstention_threshold: int = 1,
        window: int = 2,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
        adapted_memory_path: str | Path | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.per_round_k = int(per_round_k)
        self.max_rounds = int(max_rounds)
        self.abstention_threshold = int(abstention_threshold)
        self.window = int(window)
        self.include_description = bool(include_description)
        self.skip_cues = bool(skip_cues)
        self.skip_implementation = bool(skip_implementation)
        self.usage_threshold = int(usage_threshold)
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

        graph = ConceptGraph.build_from_memory(mem, min_co_overlap=1)
        adapted_records, adapted_source = self._load_adapted_records(mem)
        q_tokens = self._query_tokens(problem, previous_attempts)
        # Seed order: descending co-activation degree.
        if adapted_records:
            degree_ranked = sorted(
                mem.concepts.keys(),
                key=lambda n: (
                    self._adapted_policy_score(adapted_records.get(n), q_tokens),
                    graph.degree(n, kinds=["co_activation"]),
                ),
                reverse=True,
            )
        else:
            degree_ranked = sorted(
                mem.concepts.keys(),
                key=lambda n: graph.degree(n, kinds=["co_activation"]),
                reverse=True,
            )

        seen: set[str] = set()
        rounds_log: list[dict] = []
        coverage_history: list[int] = []

        def compute_coverage(names: set[str]) -> int:
            kinds_seen: set[str] = set()
            cues_seen: set[str] = set()
            for n in names:
                c = mem.concepts.get(n)
                if not c:
                    continue
                kinds_seen.add(c.kind)
                for cue in (c.cues or []):
                    cues_seen.add(cue)
            return len(kinds_seen) + len(cues_seen)

        def pick_next_round() -> list[str]:
            """Pick `per_round_k` concepts ranked by (new_kind, new_cues, degree)."""
            kinds_seen: Counter[str] = Counter()
            cues_seen: set[str] = set()
            for n in seen:
                c = mem.concepts.get(n)
                if not c:
                    continue
                kinds_seen[c.kind] += 1
                cues_seen.update(c.cues or [])
            scored: list[tuple[tuple[int, int, float], str]] = []
            for name, c in mem.concepts.items():
                if name in seen:
                    continue
                new_kind = 1 if kinds_seen.get(c.kind, 0) == 0 else 0
                new_cues = sum(1 for cue in (c.cues or []) if cue not in cues_seen)
                degree = graph.degree(name, kinds=["co_activation"])
                policy_score = self._adapted_policy_score(adapted_records.get(name), q_tokens)
                scored.append(((new_kind, new_cues, policy_score, degree), name))
            scored.sort(key=lambda r: r[0], reverse=True)
            return [name for _, name in scored[: self.per_round_k]]

        # Seed round (round 0)
        for name in degree_ranked[: self.per_round_k]:
            if name in mem.concepts:
                seen.add(name)
        rounds_log.append({"round": 0, "added": list(seen)[:], "coverage": compute_coverage(seen)})
        coverage_history.append(compute_coverage(seen))

        abstained = False
        abstain_reason = ""
        for r in range(1, self.max_rounds + 1):
            if len(seen) >= self.top_k:
                abstain_reason = f"reached top_k cap ({self.top_k})"
                break
            candidates = pick_next_round()
            if not candidates:
                abstain_reason = "no more candidates"
                break
            for name in candidates:
                if len(seen) >= self.top_k:
                    break
                seen.add(name)
            cov = compute_coverage(seen)
            rounds_log.append({"round": r, "added": candidates, "coverage": cov})
            coverage_history.append(cov)
            # Abstention check: rolling-window coverage gain
            if len(coverage_history) > self.window:
                gain = coverage_history[-1] - coverage_history[-1 - self.window]
                if gain < self.abstention_threshold:
                    abstained = True
                    abstain_reason = (
                        f"abstained at round {r}: coverage gain over last {self.window} rounds "
                        f"({gain}) < threshold ({self.abstention_threshold})"
                    )
                    break

        selected = sorted(seen)
        hint = self._render_adapted_hint(selected, adapted_records) if adapted_records else ""
        if not hint:
            hint = mem.to_string(
                concept_names=selected,
                include_description=self.include_description,
                skip_cues=self.skip_cues,
                skip_implementation=self.skip_implementation,
                usage_threshold=self.usage_threshold,
            )
        return RetrievalBundle(
            problem_uid=problem.uid,
            hint_text=hint or None,
            retrieved_items=[{"name": n} for n in selected],
            metadata={
                "retriever": self.name,
                "scoring_mode": "mediq_policy",
                "top_k": self.top_k,
                "num_rounds": len(rounds_log),
                "rounds": rounds_log,
                "abstained": abstained,
                "abstain_reason": abstain_reason or "max_rounds",
                "coverage_history": coverage_history,
                "num_selected": len(selected),
                "adapted_memory_source": adapted_source,
                "adapted_records_loaded": len(adapted_records),
                "adapted_policy_items_rendered": sum(1 for n in selected if n in adapted_records),
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
            raise RuntimeError(f"invalid MediQ adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != self.name:
            raise RuntimeError(f"invalid MediQ adapted memory schema: {path}")
        records: dict[str, dict] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            if not raw.get("atomic_question_templates") or not raw.get("abstention_policy"):
                raise RuntimeError(f"adapted memory missing policy fields for {concept_id}")
            records[concept_id] = raw
        if not records:
            return {}, "flat"
        return records, "mediq_memory_v1"

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

    def _adapted_policy_score(self, record: dict | None, q_tokens: set[str]) -> float:
        if not record:
            return 0.0
        text = self._adapted_record_text(record)
        overlap = len(q_tokens & _tokenize(text)) if q_tokens else 0
        try:
            info_gain = float(record.get("expected_info_gain", 0.0))
        except (TypeError, ValueError):
            info_gain = 0.0
        policy = record.get("abstention_policy") if isinstance(record.get("abstention_policy"), dict) else {}
        try:
            threshold = float(policy.get("confidence_threshold_hint", 0.0))
        except (TypeError, ValueError):
            threshold = 0.0
        return float(overlap) + info_gain + 0.25 * threshold

    @staticmethod
    def _adapted_record_text(record: dict) -> str:
        parts: list[str] = [
            str(record.get("initial_assessment") or ""),
            str(record.get("question_type") or ""),
            str(record.get("evidence_integration") or ""),
            str(record.get("retrieval_notes") or ""),
        ]
        parts.extend(str(item) for item in record.get("missing_information_targets") or [])
        parts.extend(str(item) for item in record.get("atomic_question_templates") or [])
        parts.extend(str(item) for item in record.get("routing_keywords") or [])
        policy = record.get("abstention_policy")
        if isinstance(policy, dict):
            parts.extend(str(policy.get(key) or "") for key in ("ask_when", "commit_when"))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _render_adapted_hint(
        selected: list[str],
        adapted_records: dict[str, dict],
    ) -> str:
        blocks: list[str] = []
        for name in selected:
            record = adapted_records.get(name)
            if not record:
                continue
            policy = record.get("abstention_policy") if isinstance(record.get("abstention_policy"), dict) else {}
            lines = [f"- concept: {name}"]
            assessment = str(record.get("initial_assessment") or "").strip()
            if assessment:
                lines.append(f"  mediq_initial_assessment: {assessment}")
            lines.append(f"  question_type: {record.get('question_type', 'other')}")
            targets = [str(t).strip() for t in record.get("missing_information_targets") or [] if str(t).strip()]
            if targets:
                lines.append("  missing_info_targets: " + ", ".join(targets[:5]))
            questions = [str(q).strip() for q in record.get("atomic_question_templates") or [] if str(q).strip()]
            if questions:
                lines.append("  atomic_question: " + questions[0])
            if policy:
                ask_when = str(policy.get("ask_when") or "").strip()
                commit_when = str(policy.get("commit_when") or "").strip()
                if ask_when:
                    lines.append(f"  ask_when: {ask_when}")
                if commit_when:
                    lines.append(f"  commit_when: {commit_when}")
            try:
                gain = float(record.get("expected_info_gain", 0.0))
            except (TypeError, ValueError):
                gain = 0.0
            lines.append(f"  expected_info_gain: {gain:.2f}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
