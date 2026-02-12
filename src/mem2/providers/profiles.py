from __future__ import annotations

from dataclasses import replace

from mem2.providers.base import ProviderProfile


_MOCK = ProviderProfile(
    profile_name="mock",
    backend="mock",
    provider="mock",
    model="mock-grid-echo",
    extra_gen_defaults={"n": 1},
)

_OPENROUTER = ProviderProfile(
    profile_name="llmplus_openrouter",
    backend="llmplus",
    provider="openrouter",
    model="qwen/qwen3-235b-a22b-2507",
    extra_gen_defaults={"n": 1, "temperature": 0.2},
)

_OPENROUTER_GEMINI25_FLASH_LITE = ProviderProfile(
    profile_name="llmplus_openrouter_gemini25_flash_lite",
    backend="llmplus",
    provider="openrouter",
    model="google/gemini-2.5-flash-lite-preview-09-2025",
    # Keep defaults simple for cross-provider compatibility.
    extra_gen_defaults={"n": 1},
)

_OPENAI = ProviderProfile(
    profile_name="llmplus_openai",
    backend="llmplus",
    provider="openai",
    model="gpt-4o",
    extra_gen_defaults={"n": 1, "temperature": 0.2},
)

_ARCMEMO_GPT41 = ProviderProfile(
    profile_name="llmplus_arcmemo_gpt41",
    backend="llmplus",
    provider="openai",
    model="gpt-4.1-2025-04-14",
    # ArcMemo default system prompt key is `default` in prompts.py.
    system_prompt=(
        "You are a world-class puzzle solver with exceptional pattern recognition "
        "skills and expertise in Python programming. Your task is to analyze "
        "puzzles and provide Python solutions."
    ),
    default_max_concurrency=32,
    retry_attempts=5,
    retry_wait_min=1,
    retry_wait_max=120,
    extra_gen_defaults={},
)

_XAI = ProviderProfile(
    profile_name="llmplus_xai",
    backend="llmplus",
    provider="xai",
    model="grok-4-fast-reasoning",
    extra_gen_defaults={"n": 1, "temperature": 0.2},
)

DEFAULT_PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "mock": _MOCK,
    "mock_v1": replace(_MOCK, profile_name="mock_v1"),
    "llmplus_openrouter": _OPENROUTER,
    "llmplus_openrouter_v1": replace(_OPENROUTER, profile_name="llmplus_openrouter_v1"),
    "llmplus_openrouter_gemini25_flash_lite": _OPENROUTER_GEMINI25_FLASH_LITE,
    "llmplus_openrouter_gemini25_flash_lite_v1": replace(
        _OPENROUTER_GEMINI25_FLASH_LITE,
        profile_name="llmplus_openrouter_gemini25_flash_lite_v1",
    ),
    "llmplus_openai": _OPENAI,
    "llmplus_openai_v1": replace(_OPENAI, profile_name="llmplus_openai_v1"),
    "llmplus_arcmemo_gpt41": _ARCMEMO_GPT41,
    "llmplus_arcmemo_gpt41_v1": replace(_ARCMEMO_GPT41, profile_name="llmplus_arcmemo_gpt41_v1"),
    "llmplus_xai": _XAI,
    "llmplus_xai_v1": replace(_XAI, profile_name="llmplus_xai_v1"),
}


def get_profile(name: str) -> ProviderProfile:
    if name not in DEFAULT_PROVIDER_PROFILES:
        raise KeyError(f"Unknown provider profile: {name}")
    return DEFAULT_PROVIDER_PROFILES[name]
