"""Provider construction from validated configuration."""

import httpx

from fakuicode.models import ProviderConfig
from fakuicode.providers.anthropic import AnthropicProvider
from fakuicode.providers.openai import OpenAIProvider


def create_provider(config: ProviderConfig, *, client: httpx.Client | None = None):
    if config.protocol == "anthropic":
        return AnthropicProvider(config, client=client)
    return OpenAIProvider(config, client=client)
