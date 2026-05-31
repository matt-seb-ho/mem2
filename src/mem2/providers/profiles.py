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
    # 2026-05-17 anti-storm: default attempts=5 with up to 120s backoff was
    # bombarding OpenRouter under sustained load. attempts=2 still recovers
    # from transient hiccups without extending sustained limit pressure.
    retry_attempts=2,
    retry_wait_min=2,
    retry_wait_max=15,
)

_OPENROUTER_GEMINI25_FLASH_LITE = ProviderProfile(
    profile_name="llmplus_openrouter_gemini25_flash_lite",
    backend="llmplus",
    provider="openrouter",
    model="google/gemini-2.5-flash-lite-preview-09-2025",
    # Keep defaults simple for cross-provider compatibility.
    extra_gen_defaults={"n": 1},
)

# COLM 2026 rebuttal — dsv4f sweep for paper's PS / DC / ACE / RB cross-method comparison.
# OpenRouter slug confirmed 2026-05-27: deepseek/deepseek-v4-flash, $0.0000001/tok prompt, 1M ctx.
_OPENROUTER_DSV4F = ProviderProfile(
    profile_name="llmplus_openrouter_dsv4f",
    backend="llmplus",
    provider="openrouter",
    model="deepseek/deepseek-v4-flash",
    # Mirror anti-storm settings from base OpenRouter profile (2026-05-17 fix).
    # ArcMemo system prompt for parity with arc_memo's batch puzzle-solving role.
    system_prompt=(
        "You are a world-class puzzle solver with exceptional pattern recognition "
        "skills and expertise in Python programming. Your task is to analyze "
        "puzzles and provide Python solutions."
    ),
    default_max_concurrency=16,
    retry_attempts=2,
    retry_wait_min=2,
    retry_wait_max=15,
    # 2026-05-28: OpenRouter's default routing today silently spreads dsv4f calls
    # across 12 providers. Empirically (probe_empty_by_provider_fast.py):
    #   100% empty: DeepSeek (canonical!), StreamLake, Baidu, Venice, Parasail
    #     0% empty: AtlasCloud, Alibaba, GMICloud
    # All 5 "empty" providers hit `finish_reason: length` — they consume the full
    # max_tokens on reasoning without emitting any visible content. Sampling shifted
    # since the model was published. Pin to the 3 clean fp8 providers and disable
    # fallback to bad ones.
    extra_gen_defaults={
        "n": 1,
        "temperature": 0.3,
        "extra_kwargs": {
            "extra_body": {
                "provider": {
                    "order": ["AtlasCloud", "Alibaba", "GMICloud"],
                    "allow_fallbacks": False,
                }
            }
        },
    },
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

# 2026-05-30: o4-mini for the cross-method comparison (the regime where PS beats
# baseline — reproduced arcmemo PS 59.5 vs baseline 56.5). low concurrency to respect
# OpenAI o4-mini rate limits (batch=16x2 -> 429 storm).
_ARCMEMO_O4MINI = ProviderProfile(
    profile_name="llmplus_arcmemo_o4mini",
    backend="llmplus",
    provider="openai",
    model="o4-mini-2025-04-16",
    system_prompt=(
        "You are a world-class puzzle solver with exceptional pattern recognition "
        "skills and expertise in Python programming. Your task is to analyze "
        "puzzles and provide Python solutions."
    ),
    default_max_concurrency=8,
    retry_attempts=8,
    retry_wait_min=2,
    retry_wait_max=60,
    extra_gen_defaults={},
)

_XAI = ProviderProfile(
    profile_name="llmplus_xai",
    backend="llmplus",
    provider="xai",
    model="grok-4-fast-reasoning",
    extra_gen_defaults={"n": 1, "temperature": 0.2},
)

_VLLM = ProviderProfile(
    profile_name="llmplus_vllm",
    backend="llmplus",
    provider="vllm",
    model="Qwen/Qwen3.5-9B",
    extra_gen_defaults={"n": 1, "temperature": 0.0},
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
    "llmplus_openrouter_dsv4f": _OPENROUTER_DSV4F,
    "llmplus_openrouter_dsv4f_v1": replace(
        _OPENROUTER_DSV4F,
        profile_name="llmplus_openrouter_dsv4f_v1",
    ),
    "llmplus_openai": _OPENAI,
    "llmplus_openai_v1": replace(_OPENAI, profile_name="llmplus_openai_v1"),
    "llmplus_arcmemo_gpt41": _ARCMEMO_GPT41,
    "llmplus_arcmemo_gpt41_v1": replace(_ARCMEMO_GPT41, profile_name="llmplus_arcmemo_gpt41_v1"),
    "llmplus_arcmemo_o4mini": _ARCMEMO_O4MINI,
    "llmplus_arcmemo_o4mini_v1": replace(_ARCMEMO_O4MINI, profile_name="llmplus_arcmemo_o4mini_v1"),
    "llmplus_xai": _XAI,
    "llmplus_xai_v1": replace(_XAI, profile_name="llmplus_xai_v1"),
    "llmplus_vllm": _VLLM,
    "llmplus_vllm_v1": replace(_VLLM, profile_name="llmplus_vllm_v1"),
}


def get_profile(name: str) -> ProviderProfile:
    if name not in DEFAULT_PROVIDER_PROFILES:
        raise KeyError(f"Unknown provider profile: {name}")
    return DEFAULT_PROVIDER_PROFILES[name]
