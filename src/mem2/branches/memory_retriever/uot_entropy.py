"""UoT entropy-gated interactive retrieval - axis C.3.

Port of UoT (Hu et al., NeurIPS'24; arxiv 2402.03271).

Paper: literature/2402.03271.pdf
Repo:  third_party/uot/ (entry: src/uot/uot.py::UoTNode.reward_function + expected_reward)

Specifically ported:
    - The *information-gain reward function* from `UoTNode.reward_function`:
      for a candidate split ratio x in (0, 1),
          reward(x) = (-x log2 x - (1-x) log2 (1-x)) / (1 + |2x - 1| / λ)
      (Shannon entropy damped by asymmetry). Peaks at x = 0.5, zero at
      extremes. λ = 0.4 in the paper (higher λ = more symmetric).
    - The abstention-by-low-reward pattern: if no candidate round produces
      information gain above `min_gain`, stop asking (the paper's analog of
      "commit to the current guess").

Deliberate simplifications:
    - UoT's full expected-reward TREE search (`expected_reward` recursion,
      `avg_expected` / `max_expected` across n_extend_layers) is NOT ported.
      We implement the ONE-STEP reward signal - each round's candidate set
      is scored via `reward_function` on the kind-distribution entropy; if
      no candidate yields gain at least `min_gain`, abstain. This preserves the
      distinctive entropy-based abstention signal while keeping the
      retriever deterministic and LLM-free.
    - The LLM-based question generation is replaced by kind-grouped
      candidate sets (same as MediQ's approach): each "question" is
      "should we include kind K next?"; the split ratio is (concepts of K
      still unretrieved) / (concepts still unretrieved total).

C.3 vs C.4 (MediQ) vs C.2 (RRMC-interactive):
    - RRMC: symmetric patience over coverage gain.
    - MediQ: rolling-window coverage threshold.
    - UoT (this module): *Shannon-entropy* info-gain signal on the kind
      distribution. A branch with 50/50 split is rewarded highest; a
      saturated kind (all one way) gets zero reward and triggers abstention.
    - All three share the same interactive-retrieval interface - the
      abstention *mechanics* are the axis-C ablation question.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
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
_DEFAULT_ADAPTED_MEMORY_PATH = CONCEPT_MEMORY_DIR / "ports" / "uot_memory_v1.json"


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def _entropy_reward(x: float, lamb: float = 0.4) -> float:
    """UoT's damped-Shannon reward. Zero at x in {0,1}; peak near x=0.5."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    h = -x * math.log2(x) - (1 - x) * math.log2(1 - x)
    damping = 1.0 + abs(2 * x - 1) / lamb
    return h / damping


class UoTEntropyRetriever:
    """Entropy-gated multi-round retrieval with UoT reward signal."""

    name = "uot_entropy"
    COMPATIBLE_SCHEMAS = {"arcmemo_ps"}

    def __init__(
        self,
        top_k: int = 10,
        per_round_k: int = 2,
        max_rounds: int = 5,
        min_gain: float = 0.1,
        lamb: float = 0.4,
        include_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        usage_threshold: int = 1,
        adapted_memory_path: str | Path | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.per_round_k = int(per_round_k)
        self.max_rounds = int(max_rounds)
        self.min_gain = float(min_gain)
        self.lamb = float(lamb)
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
        degree_ranked = sorted(
            mem.concepts.keys(),
            key=lambda n: (
                self._adapted_uncertainty_score(adapted_records.get(n), q_tokens),
                graph.degree(n, kinds=["co_activation"]),
            ),
            reverse=True,
        )

        seen: set[str] = set()
        rounds_log: list[dict] = []
        reward_history: list[float] = []

        def kind_distribution(names: set[str]) -> Counter[str]:
            dist: Counter[str] = Counter()
            for n in names:
                c = mem.concepts.get(n)
                if c:
                    dist[c.kind] += 1
            return dist

        def pick_and_score_round() -> tuple[list[str], float]:
            """Pick `per_round_k` next candidates, score reward of round."""
            # Candidates ranked by (unseen kind contribution, degree)
            kinds_so_far = kind_distribution(seen)
            scored: list[tuple[tuple[float, int, float], str, str]] = []
            for name, c in mem.concepts.items():
                if name in seen:
                    continue
                degree = graph.degree(name, kinds=["co_activation"])
                kind_is_new = 1 if kinds_so_far.get(c.kind, 0) == 0 else 0
                adapted_score = self._adapted_uncertainty_score(
                    adapted_records.get(name),
                    q_tokens,
                )
                scored.append(((adapted_score, kind_is_new, degree), name, c.kind))
            scored.sort(key=lambda r: r[0], reverse=True)
            picks = [name for _, name, _ in scored[: self.per_round_k]]
            picked_kinds = [k for _, n, k in scored[: self.per_round_k]]
            adapted_rewards = [
                self._adapted_entropy_reward(adapted_records.get(name))
                for name in picks
                if name in adapted_records
            ]
            if adapted_rewards:
                return picks, sum(adapted_rewards) / len(adapted_rewards)

            # UoT reward: entropy over the KIND distribution AFTER picking.
            post_dist = Counter(kinds_so_far)
            for k in picked_kinds:
                post_dist[k] += 1
            total = sum(post_dist.values())
            if total <= 1:
                return picks, 0.0
            # For binary-tree UoT, reward is on one branch's ratio. We adapt
            # to multi-way: reward = mean over each kind of reward(count/total),
            # which peaks when the kind-distribution is uniform (max entropy).
            rewards = [_entropy_reward(cnt / total, self.lamb)
                        for cnt in post_dist.values()]
            return picks, (sum(rewards) / len(rewards)) if rewards else 0.0

        # Seed round (round 0)
        for name in degree_ranked[: self.per_round_k]:
            seen.add(name)
        seed_dist = kind_distribution(seen)
        total0 = sum(seed_dist.values()) or 1
        seed_reward = sum(
            _entropy_reward(cnt / total0, self.lamb) for cnt in seed_dist.values()
        ) / max(len(seed_dist), 1)
        rounds_log.append({"round": 0, "added": sorted(seen), "reward": round(seed_reward, 3)})
        reward_history.append(seed_reward)

        abstained = False
        abstain_reason = ""
        for r in range(1, self.max_rounds + 1):
            if len(seen) >= self.top_k:
                abstain_reason = f"reached top_k cap ({self.top_k})"
                break
            picks, reward = pick_and_score_round()
            if not picks:
                abstain_reason = "no more candidates"
                break
            gain = reward - reward_history[-1]
            for name in picks:
                if len(seen) >= self.top_k:
                    break
                seen.add(name)
            rounds_log.append({
                "round": r, "added": picks, "reward": round(reward, 3),
                "gain": round(gain, 3),
            })
            reward_history.append(reward)
            # UoT abstention: if expected gain < min_gain, stop.
            if gain < self.min_gain:
                abstained = True
                abstain_reason = (
                    f"abstained at round {r}: info-gain ({gain:.3f}) < "
                    f"min_gain ({self.min_gain})"
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
                "top_k": self.top_k,
                "num_rounds": len(rounds_log),
                "rounds": rounds_log,
                "abstained": abstained,
                "abstain_reason": abstain_reason or "max_rounds",
                "reward_history": [round(r, 3) for r in reward_history],
                "lamb": self.lamb,
                "min_gain": self.min_gain,
                "num_selected": len(selected),
                "adapted_memory_source": adapted_source,
                "adapted_records_loaded": len(adapted_records),
                "adapted_entropy_items_rendered": sum(1 for n in selected if n in adapted_records),
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
            raise RuntimeError(f"invalid UoT adapted memory JSON: {path}") from exc
        if data.get("schema_version") != "1" or data.get("port") != self.name:
            raise RuntimeError(f"invalid UoT adapted memory schema: {path}")
        records: dict[str, dict] = {}
        for raw in data.get("adapted_concepts") or []:
            if not isinstance(raw, dict):
                continue
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or concept_id not in mem.concepts:
                continue
            if "candidate_question" not in raw or "entropy_reward" not in raw:
                raise RuntimeError(f"adapted memory missing UoT fields for {concept_id}")
            records[concept_id] = raw
        if not records:
            return {}, "flat"
        return records, "uot_memory_v1"

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

    def _adapted_uncertainty_score(
        self,
        record: dict | None,
        q_tokens: set[str],
    ) -> float:
        if not record:
            return 0.0
        entropy_reward = self._adapted_entropy_reward(record)
        try:
            expected_yes_ratio = float(record.get("expected_yes_ratio", 0.5))
        except (TypeError, ValueError):
            expected_yes_ratio = 0.5
        split_balance = 1.0 - min(1.0, abs(expected_yes_ratio - 0.5) * 2.0)
        overlap = len(q_tokens & _tokenize(self._adapted_record_text(record))) if q_tokens else 0
        return entropy_reward + split_balance + float(overlap)

    @staticmethod
    def _adapted_entropy_reward(record: dict | None) -> float:
        if not record:
            return 0.0
        try:
            return max(0.0, min(1.0, float(record.get("entropy_reward", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _adapted_record_text(record: dict) -> str:
        parts: list[str] = [
            str(record.get("uncertainty_state") or ""),
            str(record.get("candidate_question") or ""),
            str(record.get("information_gain_target") or ""),
            str(record.get("simulation_tree_role") or ""),
            str(record.get("reward_propagation_notes") or ""),
            str(record.get("retrieval_notes") or ""),
        ]
        parts.extend(str(item) for item in record.get("yes_partition_hint") or [])
        parts.extend(str(item) for item in record.get("no_partition_hint") or [])
        parts.extend(str(item) for item in record.get("routing_keywords") or [])
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
            lines.append(
                "  uot_question: "
                + str(record.get("candidate_question") or "").strip()
            )
            lines.append(
                "  uot_expected_yes_ratio: "
                f"{float(record.get('expected_yes_ratio', 0.5)):.2f}"
            )
            lines.append(
                "  uot_entropy_reward: "
                f"{float(record.get('entropy_reward', 0.0)):.2f}"
            )
            target = str(record.get("information_gain_target") or "").strip()
            if target:
                lines.append(f"  information_gain_target: {target}")
            role = str(record.get("simulation_tree_role") or "").strip()
            if role:
                lines.append(f"  simulation_tree_role: {role}")
            yes = [str(x).strip() for x in record.get("yes_partition_hint") or [] if str(x).strip()]
            no = [str(x).strip() for x in record.get("no_partition_hint") or [] if str(x).strip()]
            if yes:
                lines.append("  yes_partition_hint: " + ", ".join(yes[:4]))
            if no:
                lines.append("  no_partition_hint: " + ", ".join(no[:4]))
            propagation = str(record.get("reward_propagation_notes") or "").strip()
            if propagation:
                lines.append(f"  reward_propagation_notes: {propagation}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
