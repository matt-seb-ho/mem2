"""ARC-3 interactive game adapter (SDK scaffold).

ARC-3 is a turn-based interactive game benchmark launched 2026-03-25
(SDK at ``three.arcprize.org``). No Python SDK is installable at the time
of this writing (2026-04-19); the adapter below stubs the interface so
Phase-1 code can import it, but actual rollouts require SDK integration.

Open questions (to resolve before ARC-3 runs):
  1. SDK package name and auth flow.
  2. Turn-based rollout API — sync vs async, max turns, observation format.
  3. How RRMC-style multi-round probing fits the rollout structure.

Until the SDK is wired, full experiments should use ``ArcAgiBenchmarkAdapter``
(ARC-1/2 data, already working). The sweep driver honors a ``benchmark``
config flag so we can pivot when the SDK lands.
"""
from __future__ import annotations

from mem2.core.entities import ProblemSpec, RunContext, TaskSpec


class Arc3TaskAdapter:
    """Task-spec adapter for ARC-3. Produces the TaskSpec expected by the
    inference engine; actual game-state rollout is handled by the benchmark
    adapter + SDK integration (see ``Arc3SdkBenchmark``, not yet written).
    """

    name = "arc3"
    DOMAIN_NAME = "arc"

    def __init__(self, task_name: str = "arc3_game") -> None:
        self.task_name = task_name

    def get_task_spec(self, ctx: RunContext) -> TaskSpec:
        return TaskSpec(
            task_name=self.task_name,
            task_description=(
                "Play ARC-3 interactive puzzle games. Observe the game state, "
                "issue moves, and solve the underlying concept."
            ),
            sample_format={
                "game_id": "str",
                "observation": "list[list[int]]",
                "action_space": "list[str]",
            },
            feedback_mode="sdk",
            metadata={
                "adapter": self.name,
                "sdk_url": "https://three.arcprize.org",
                "sdk_wired": False,
            },
        )

    def format_problem_sample(self, problem: ProblemSpec) -> dict:
        return {
            "uid": problem.uid,
            "game_metadata": problem.metadata,
        }


class Arc3SdkBenchmark:
    """Placeholder for the ARC-3 benchmark adapter. Raises on ``load`` until
    the SDK is wired. Import-safe so the registry can still build."""

    name = "arc3_sdk"
    DOMAIN_NAME = "arc"

    def __init__(
        self,
        sdk_api_key_env: str = "ARC3_API_KEY",
        limit: int = 5,
    ) -> None:
        self.sdk_api_key_env = sdk_api_key_env
        self.limit = int(limit)

    def load(self, ctx: RunContext) -> dict[str, ProblemSpec]:
        raise NotImplementedError(
            "ARC-3 SDK not yet integrated. "
            "Open questions: package name, auth flow, rollout API. "
            "Phase-1 runs should use 'arc_agi' (ARC-1/2) until resolved. "
            "Track progress in docs/phase1_axis_<N>_report.md."
        )

    def validate(self, problems: dict[str, ProblemSpec]) -> None:
        pass
