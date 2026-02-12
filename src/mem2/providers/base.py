from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProviderProfile:
    profile_name: str
    backend: str
    provider: str
    model: str
    cache_dir: str = ".llm_cache"
    system_prompt: str | None = None
    default_max_concurrency: int = 16
    retry_attempts: int = 5
    retry_wait_min: int = 1
    retry_wait_max: int = 120
    extra_gen_defaults: dict[str, Any] | None = None
