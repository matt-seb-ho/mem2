from mem2.providers.llmplus_client import LLMPlusProviderClient
from mem2.providers.mock_client import MockProviderClient

PROVIDERS = {
    "mock": MockProviderClient,
    "mock_v1": MockProviderClient,
    "llmplus_openrouter": LLMPlusProviderClient,
    "llmplus_openrouter_v1": LLMPlusProviderClient,
    "llmplus_openai": LLMPlusProviderClient,
    "llmplus_openai_v1": LLMPlusProviderClient,
    "llmplus_arcmemo_gpt41": LLMPlusProviderClient,
    "llmplus_arcmemo_gpt41_v1": LLMPlusProviderClient,
    "llmplus_arcmemo_o4mini": LLMPlusProviderClient,
    "llmplus_arcmemo_o4mini_v1": LLMPlusProviderClient,
    "llmplus_xai": LLMPlusProviderClient,
    "llmplus_xai_v1": LLMPlusProviderClient,
    "llmplus_vllm": LLMPlusProviderClient,
    "llmplus_vllm_v1": LLMPlusProviderClient,
    # COLM 2026 rebuttal — added 2026-05-27.
    "llmplus_openrouter_dsv4f": LLMPlusProviderClient,
    "llmplus_openrouter_dsv4f_v1": LLMPlusProviderClient,
    # Also: gemini25_flash_lite profile exists in profiles.py but was missing from
    # this registry — restoring for completeness.
    "llmplus_openrouter_gemini25_flash_lite": LLMPlusProviderClient,
    "llmplus_openrouter_gemini25_flash_lite_v1": LLMPlusProviderClient,
}
