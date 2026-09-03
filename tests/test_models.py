"""Model resolution: the four (provider, backend) combinations and credentials."""

import pytest

from devteam.config import AppConfig, Backend, ModelSpec, Provider
from devteam.models import MissingCredentialsError, build_model, litellm_id, require_credentials
from tests.conftest import base_raw


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


def test_extra_and_platform_coordinates_reach_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    spec = ModelSpec(
        provider=Provider.ANTHROPIC,
        backend=Backend.AGENT_PLATFORM,
        model="claude-opus-5",
        extra={"thinking": {"type": "adaptive"}},
    )
    model = build_model(spec, AppConfig.model_validate(base_raw()))
    assert model.model == "vertex_ai/claude-opus-5"
    assert model._additional_args == {
        "thinking": {"type": "adaptive"},
        "vertex_project": "proj-1",
        "vertex_location": "us-central1",
    }
