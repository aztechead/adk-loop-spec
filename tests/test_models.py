"""Model resolution: native ADK classes on Agent Platform, LiteLLM on api-key."""

import pytest
from anthropic.lib.vertex import AsyncAnthropicVertex
from google.adk.models import AnthropicGenerateContentConfig, Gemini, LiteLlm
from google.adk.models.anthropic_llm import Claude
from google.genai import types

from devteam.config import (
    AppConfig,
    Backend,
    Effort,
    GenerationConfig,
    ModelSpec,
    Provider,
    ThinkingLevel,
)
from devteam.models import (
    MissingCredentialsError,
    build_llm,
    build_model,
    generation_config,
    model_id,
    require_credentials,
)
from tests.conftest import base_raw


def app_config() -> AppConfig:
    return AppConfig.model_validate(base_raw())


def test_agent_platform_is_the_default_backend() -> None:
    spec = ModelSpec(provider=Provider.GEMINI, model="m")
    assert spec.backend is Backend.AGENT_PLATFORM


@pytest.mark.parametrize(
    ("provider", "backend", "expected"),
    [
        (Provider.GEMINI, Backend.AGENT_PLATFORM, "m"),
        (
            Provider.ANTHROPIC,
            Backend.AGENT_PLATFORM,
            "Claude:projects/offline-project/locations/us-central1/publishers/anthropic/models/m",
        ),
        (Provider.GEMINI, Backend.API_KEY, "gemini/m"),
        (Provider.ANTHROPIC, Backend.API_KEY, "anthropic/m"),
    ],
)
def test_model_id_covers_every_combination(
    provider: Provider, backend: Backend, expected: str
) -> None:
    spec = ModelSpec(provider=provider, backend=backend, model="m")
    assert model_id(spec, app_config()) == expected


def test_missing_api_key_fails_naming_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    spec = ModelSpec(provider=Provider.ANTHROPIC, backend=Backend.API_KEY, model="m")
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        require_credentials(spec, app_config())


def test_agent_platform_needs_a_project_not_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY")
    spec = ModelSpec(provider=Provider.GEMINI, model="m")
    require_credentials(spec, app_config())  # ADC-authenticated: no key required

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    with pytest.raises(MissingCredentialsError, match="GOOGLE_CLOUD_PROJECT"):
        require_credentials(spec, app_config())


def test_gemini_on_agent_platform_is_the_native_class_on_vertex() -> None:
    llm = build_llm(
        ModelSpec(provider=Provider.GEMINI, model="gemini-3.1-pro-preview"), app_config()
    )
    assert isinstance(llm, Gemini)
    assert llm.client_kwargs == {
        "vertexai": True,
        "project": "offline-project",
        "location": "us-central1",
    }


def test_claude_on_agent_platform_uses_its_own_region() -> None:
    spec = ModelSpec(provider=Provider.ANTHROPIC, model="claude-opus-5", location="us-east5")
    llm = build_llm(spec, app_config())
    assert isinstance(llm, Claude)
    assert llm.model == "claude-opus-5"
    assert isinstance(llm.client, AsyncAnthropicVertex) and llm.client.region == "us-east5"


def test_api_key_backend_goes_through_litellm_with_extras() -> None:
    spec = ModelSpec(
        provider=Provider.ANTHROPIC,
        backend=Backend.API_KEY,
        model="claude-opus-5",
        extra={"thinking": {"type": "adaptive"}},
    )
    llm = build_llm(spec, app_config())
    assert isinstance(llm, LiteLlm)
    assert llm.model == "anthropic/claude-opus-5"
    assert llm._additional_args == {"thinking": {"type": "adaptive"}}


def test_generation_settings_become_typed_adk_config() -> None:
    gemini = ModelSpec(
        provider=Provider.GEMINI,
        model="m",
        generation=GenerationConfig(temperature=0.2, thinking_level=ThinkingLevel.HIGH),
    )
    config = generation_config(gemini)
    assert isinstance(config, types.GenerateContentConfig)
    assert config.temperature == 0.2
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is types.ThinkingLevel.HIGH

    claude = ModelSpec(
        provider=Provider.ANTHROPIC, model="m", generation=GenerationConfig(effort=Effort.XHIGH)
    )
    claude_config = generation_config(claude)
    assert isinstance(claude_config, AnthropicGenerateContentConfig)
    assert claude_config.effort == "xhigh"

    assert generation_config(ModelSpec(provider=Provider.GEMINI, model="m")) is None


def test_build_model_pairs_llm_with_generation() -> None:
    spec = ModelSpec(
        provider=Provider.GEMINI, model="m", generation=GenerationConfig(temperature=0)
    )
    model = build_model(spec, app_config())
    assert isinstance(model.llm, Gemini)
    assert model.generation is not None and model.generation.temperature == 0
