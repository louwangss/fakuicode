from __future__ import annotations

import pytest

from fakuicode.models import ProviderConfig


class RecordingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("module_name", "provider_name", "protocol"),
    [
        ("fakuicode.providers.openai", "OpenAIProvider", "openai"),
        ("fakuicode.providers.anthropic", "AnthropicProvider", "anthropic"),
    ],
)
def test_provider_closes_only_the_http_client_it_created(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    provider_name: str,
    protocol: str,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    provider_type = getattr(module, provider_name)
    owned = RecordingClient()
    monkeypatch.setattr(module.httpx, "Client", lambda **_kwargs: owned)
    config = ProviderConfig(
        protocol,
        "model",
        "https://api.example.test",
        "secret",
    )

    provider = provider_type(config)
    provider.close()
    provider.close()

    borrowed = RecordingClient()
    shared_provider = provider_type(config, client=borrowed)
    shared_provider.close()

    assert owned.close_calls == 1
    assert borrowed.close_calls == 0
