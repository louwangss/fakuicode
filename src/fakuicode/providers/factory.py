"""Provider construction from validated configuration."""

from fakuicode.models import ProviderConfig
from fakuicode.providers.anthropic import AnthropicProvider
from fakuicode.providers.openai import OpenAIProvider


def create_provider(config: ProviderConfig):
    if config.protocol == "anthropic":
        return AnthropicProvider(config)
    return OpenAIProvider(config)
