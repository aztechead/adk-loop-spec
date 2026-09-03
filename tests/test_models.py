"""Model resolution: the four (provider, backend) combinations and credentials."""

from __future__ import annotations

import pytest

from devteam.config import Backend, ModelSpec, Provider
from devteam.models import MissingCredentialsError, litellm_id, require_credentials


@pytest.mark.parametrize(
    ("provider", "backend", "expected"),
    [
        (Provider.GEMINI, Backend.API_KEY, "gemini/m"),
        (Provider.GEMINI, Backend.AGENT_PLATFORM, "vertex_ai/m"),
        (Provider.ANTHROPIC, Backend.API_KEY, "anthropic/m"),
        (Provider.ANTHROPIC, Backend.AGENT_PLATFORM, "vertex_ai/m"),
    ],
)
def test_litellm_id_covers_every_combination(
    provider: Provider, backend: Backend, expected: str
) -> None:
    assert litellm_id(ModelSpec(provider=provider, backend=backend, model="m")) == expected


def test_missing_api_key_fails_naming_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    spec = ModelSpec(provider=Provider.ANTHROPIC, backend=Backend.API_KEY, model="m")
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        require_credentials(spec)


def test_agent_platform_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY")
    spec = ModelSpec(provider=Provider.GEMINI, backend=Backend.AGENT_PLATFORM, model="m")
    require_credentials(spec)  # ADC-authenticated: no environment variable required
