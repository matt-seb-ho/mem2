from __future__ import annotations

from typing import Any


class MockProviderClient:
    """Deterministic provider for local smoke runs without API keys."""

    name = "mock"
    version = "0.1"
    supports_multi_completion = True

    def __init__(self, profile_cfg: dict[str, Any] | None = None, **kwargs: Any):
        self.profile_cfg = dict(profile_cfg or {})
        self.profile_cfg.update(kwargs)
        self.calls = 0

    async def async_generate(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[str]:
        self.calls += 1
        n = int(gen_cfg.get("n", 1))
        # Intentionally simple baseline: identity transform in markdown python block.
        response = (
            "```python\n"
            "import numpy as np\n\n"
            "def transform(input_grid: np.ndarray) -> np.ndarray:\n"
            "    return input_grid\n"
            "```"
        )
        return [response for _ in range(n)]

    async def async_batch_generate(
        self,
        prompts: list[str | list[dict[str, Any]]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[list[str | None]]:
        rows = []
        for prm in prompts:
            rows.append(await self.async_generate(prm, model=model, gen_cfg=gen_cfg))
        return rows

    def get_usage_snapshot(self) -> dict[str, Any]:
        return {"provider": self.name, "calls": self.calls}
