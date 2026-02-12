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
    "llmplus_xai": LLMPlusProviderClient,
    "llmplus_xai_v1": LLMPlusProviderClient,
}
