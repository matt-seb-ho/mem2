from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mem2.providers.profiles import get_profile


class LLMPlusProviderClient:
    name = "llmplus"
    version = "0.1"
    supports_multi_completion = True

    def __init__(self, profile_cfg: dict[str, Any] | None = None, **kwargs: Any):
        merged_cfg = dict(profile_cfg or {})
        merged_cfg.update(kwargs)
        profile_name = merged_cfg.get("profile_name")
        if not profile_name:
            raise ValueError("Provider profile must include profile_name")
        profile = get_profile(profile_name)
        max_concurrency = int(
            merged_cfg.get("default_max_concurrency", profile.default_max_concurrency)
        )
        retry_attempts = int(merged_cfg.get("retry_attempts", profile.retry_attempts))
        retry_wait_min = int(merged_cfg.get("retry_wait_min", profile.retry_wait_min))
        retry_wait_max = int(merged_cfg.get("retry_wait_max", profile.retry_wait_max))
        cache_dir = str(merged_cfg.get("cache_dir", profile.cache_dir))
        system_prompt = merged_cfg.get("system_prompt", profile.system_prompt)

        # Prefer vendored llm_wrapper; fallback to workspace-level module.
        repo_root = Path(__file__).resolve().parents[3]
        vendored = repo_root / "third_party" / "llm_wrapper"
        if vendored.exists() and str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))

        # Optional explicit dotenv path; otherwise rely on shell environment.
        dotenv_cfg = merged_cfg.get("dotenv_path") or merged_cfg.get("env_file")
        if dotenv_cfg:
            dotenv_path_obj = Path(str(dotenv_cfg)).expanduser()
            if not dotenv_path_obj.is_absolute():
                dotenv_path_obj = dotenv_path_obj.resolve()
            dotenv_path: str | None = str(dotenv_path_obj)
        else:
            dotenv_path = None

        try:
            from llmplus import GenerationConfig, LLMClient, RetryConfig
            from llmplus.model_registry import Provider
        except Exception as exc:  # pragma: no cover - depends on optional provider deps
            raise RuntimeError(
                "llmplus provider backend is unavailable. "
                "Ensure third_party/llm_wrapper is present and llmplus dependencies are installed."
            ) from exc

        provider_enum = Provider[profile.provider.upper()]
        retry_cfg = RetryConfig(
            attempts=retry_attempts,
            wait_min=retry_wait_min,
            wait_max=retry_wait_max,
        )
        self._generation_cls = GenerationConfig
        self._client = LLMClient(
            provider=provider_enum,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            default_max_concurrency=max_concurrency,
            retry_cfg=retry_cfg,
            dotenv_path=dotenv_path,
        )
        self._model = profile.model
        self._default_gen = profile.extra_gen_defaults or {}
        meta = getattr(self._client, "provider_meta", None)
        self.supports_multi_completion = bool(getattr(meta, "supports_multi", True))

    async def async_generate(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[str]:
        final_cfg = dict(self._default_gen)
        final_cfg.update(gen_cfg or {})
        cfg = self._generation_cls(**final_cfg)
        model_name = model or self._model
        return await self._client.async_generate(prompt=prompt, model=model_name, gen_cfg=cfg)

    async def async_batch_generate(
        self,
        prompts: list[str | list[dict[str, Any]]],
        model: str,
        gen_cfg: dict[str, Any],
    ) -> list[list[str | None]]:
        final_cfg = dict(self._default_gen)
        final_cfg.update(gen_cfg or {})
        cfg = self._generation_cls(**final_cfg)
        model_name = model or self._model
        return await self._client.async_batch_generate(prompts=prompts, model=model_name, gen_cfg=cfg)

    def get_usage_snapshot(self) -> dict[str, Any]:
        return self._client.get_token_usage_dict()
