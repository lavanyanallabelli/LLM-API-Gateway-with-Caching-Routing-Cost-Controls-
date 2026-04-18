import os
from src.providers import OpenAIProvider, AnthropicProvider
from src.providers.base import BaseProvider

def get_provider(provider_name: str) -> BaseProvider:
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. choose from {list(providers.keys())}")

    provider = providers[provider_name]()

    if not provider.is_available():
        raise ValueError(f"Provider '{provider_name}' is not available. check your API key.")

    return provider