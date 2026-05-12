"""UoT entropy-gated interactive retrieval — axis C.3.

Port of UoT (Hu et al., NeurIPS'24; arxiv 2402.03271).

Paper: literature/2402.03271.pdf
Repo:  third_party/uot/ (entry: src/uot/uot.py::UoTNode.reward_function + expected_reward)

Specifically ported:
    - The *information-gain reward function* from `UoTNode.reward_function`:
      for a candidate split ratio x ∈ (0, 1),
          reward(x) = (-x log2 x - (1-x) log2 (1-x)) / (1 + |2x - 1| / λ)
      (Shannon entropy damped by asymmetry). Peaks at x = 0.5, zero at
      extremes. λ = 0.4 in the paper (higher λ = more symmetric).
    - The abstention-by-low-reward pattern: if no candidate round produces
      information gain above `min_gain`, stop asking (the paper's analog of
      "commit to the current guess").

Deliberate simplifications:
    - UoT's full expected-reward TREE search (`expected_reward` recursion,
      `avg_expected` / `max_expected` across n_extend_layers) is NOT ported.
      We implement the ONE-STEP reward signal — each round's candidate set
      is scored via `reward_function` on the kind-distribution entropy; if
      no candidate yields gain ≥ `min_gain`, abstain. This preserves the
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
    - All three share the same interactive-retrieval interface — the
      abstention *mechanics* are the axis-C ablation question.
"""
from __future__ import annotations

import math
from collections import Counter

from mem2.concepts.graph import ConceptGraph
from mem2.concepts.memory import ConceptMemory
from mem2.core.entities import (
    AttemptRecord,
    MemoryState,
    ProblemSpec,
    RetrievalBundle,
    RunContext,
)


def _entropy_reward(x: float, lamb: float = 0.4) -> float:
    """UoT's damped-Shannon reward. Zero at x∈{0,1}; peak near x=0.5."""
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
        degree_ranked = sorted(
            mem.concepts.keys(),
            key=lambda n: graph.degree(n, kinds=["co_activation"]),
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
            scored: list[tuple[tuple[int, float], str, str]] = []
            for name, c in mem.concepts.items():
                if name in seen:
                    continue
                degree = graph.degree(name, kinds=["co_activation"])
                kind_is_new = 1 if kinds_so_far.get(c.kind, 0) == 0 else 0
                scored.append(((kind_is_new, degree), name, c.kind))
            scored.sort(key=lambda r: r[0], reverse=True)
            picks = [name for _, name, _ in scored[: self.per_round_k]]
            picked_kinds = [k for _, n, k in scored[: self.per_round_k]]

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
