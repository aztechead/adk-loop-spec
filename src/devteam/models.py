"""Resolve config model specs into ADK model objects and registry ids.

The default backend is Google Cloud Agent Platform (formerly Vertex AI),
authenticated with Application Default Credentials (ADC) and served through
ADK's native model classes; the vendor's own API is the opt-in path, served
through LiteLLM with the vendor's key:

    gemini    + agent-platform  -> Gemini(vertexai=True, project, location)   ADC
    anthropic + agent-platform  -> Claude(AsyncAnthropicVertex(project, region)) ADC
    gemini    + api-key         -> LiteLlm("gemini/<model>")      GOOGLE_API_KEY
    anthropic + api-key         -> LiteLlm("anthropic/<model>")   ANTHROPIC_API_KEY

:func:`model_id` gives the same choice as a string ADK's model registry
resolves on its own, which is what loop-spec's phase and role routes take.
"""

import os
from typing import NamedTuple, assert_never

from anthropic.lib.vertex import AsyncAnthropicVertex
from google.adk.models import AnthropicGenerateContentConfig, BaseLlm, Gemini, LiteLlm
from google.adk.models.anthropic_llm import Claude
from google.genai import types

from devteam.config import AgentRole, AppConfig, Backend, GenerationConfig, ModelSpec, Provider

PROJECT_VAR = "GOOGLE_CLOUD_PROJECT"
ADC_HINT = "gcloud auth application-default login"

# The environment each api-key pair authenticates with. Checked up front so a
# missing key fails at build time with its name, not mid-conversation.
_API_KEY_VAR: dict[Provider, str] = {
    Provider.GEMINI: "GOOGLE_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


class MissingCredentialsError(RuntimeError):
    """A model spec needs credentials or coordinates that are not set."""


class PlatformCoordinates(NamedTuple):
    """Where an Agent Platform call lands."""

    project: str
    location: str

    @property
    def resource_prefix(self) -> str:
        return f"projects/{self.project}/locations/{self.location}"


class AgentModel(NamedTuple):
    """What an LlmAgent needs from a spec: the model object and its generation settings."""

    llm: BaseLlm
    generation: types.GenerateContentConfig | None


def project_for(config: AppConfig) -> str | None:
    """The Google Cloud project: ``gcp.project`` in YAML, else the environment."""
    return config.gcp.project or os.environ.get(PROJECT_VAR)


def coordinates_for(spec: ModelSpec, config: AppConfig) -> PlatformCoordinates:
    """The project and region an agent-platform spec is served from."""
    project = project_for(config)
    if not project:
        raise MissingCredentialsError(
            f"{spec.model} on agent-platform needs gcp.project or ${PROJECT_VAR}; "
            f"ADC supplies the identity ({ADC_HINT})"
        )
    return PlatformCoordinates(project, spec.location or config.gcp.location)


def require_credentials(spec: ModelSpec, config: AppConfig) -> None:
    """Fail loudly, naming what is missing, when a spec cannot authenticate."""
    match spec.backend:
        case Backend.AGENT_PLATFORM:
            coordinates_for(spec, config)
        case Backend.API_KEY:
            variable = _API_KEY_VAR[spec.provider]
            if not os.environ.get(variable):
                raise MissingCredentialsError(f"{model_id(spec, config)} needs ${variable}")
        case _:
            assert_never(spec.backend)


def model_id(spec: ModelSpec, config: AppConfig) -> str:
    """A model string ADK's registry resolves to the same class :func:`build_llm` uses.

    Agent-platform Gemini is a bare id (the Gemini class picks Vertex from the
    environment :mod:`devteam.loopspec` exports); agent-platform Claude is the
    full resource name behind the registry's ``Claude:`` class override, so the
    string both resolves natively and carries a ``/`` for loop-spec's route
    check. Api-key specs are LiteLLM ``provider/model`` ids.
    """
    match spec.backend:
        case Backend.AGENT_PLATFORM:
            if spec.provider is Provider.GEMINI:
                return spec.model
            prefix = coordinates_for(spec, config).resource_prefix
            return f"Claude:{prefix}/publishers/anthropic/models/{spec.model}"
        case Backend.API_KEY:
            return f"{spec.provider.value}/{spec.model}"
        case _:
            assert_never(spec.backend)


def build_llm(spec: ModelSpec, config: AppConfig) -> BaseLlm:
    """The ADK model object for a spec, authenticated for its backend."""
    require_credentials(spec, config)
    match spec.backend:
        case Backend.AGENT_PLATFORM:
            where = coordinates_for(spec, config)
            match spec.provider:
                case Provider.GEMINI:
                    return Gemini(
                        model=spec.model,
                        client_kwargs={
                            "vertexai": True,
                            "project": where.project,
                            "location": where.location,
                        },
                    )
                case Provider.ANTHROPIC:
                    return Claude(
                        model=spec.model,
                        client=AsyncAnthropicVertex(
                            project_id=where.project, region=where.location
                        ),
                    )
                case _:
                    assert_never(spec.provider)
        case Backend.API_KEY:
            if spec.provider is Provider.GEMINI:
                # Gemini through LiteLLM is the operator's explicit choice here,
                # so ADK's advisory to switch to the native class is noise.
                os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
            return LiteLlm(model=model_id(spec, config), **spec.extra)
        case _:
            assert_never(spec.backend)


def generation_config(spec: ModelSpec) -> types.GenerateContentConfig | None:
    """The spec's generation settings as ADK's typed config, or None when all default."""
    settings = spec.generation
    thinking = (
        types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(settings.thinking_level.value.upper())
            if settings.thinking_level
            else None,
            include_thoughts=settings.include_thoughts or None,
        )
        if settings.thinking_level or settings.include_thoughts
        else None
    )
    if settings == GenerationConfig():
        return None
    if spec.provider is Provider.ANTHROPIC:
        return AnthropicGenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            thinking_config=thinking,
            effort=settings.effort.value if settings.effort else None,
        )
    return types.GenerateContentConfig(
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        thinking_config=thinking,
    )


def build_model(spec: ModelSpec, config: AppConfig) -> AgentModel:
    """Everything an agent takes from one spec."""
    return AgentModel(build_llm(spec, config), generation_config(spec))


def model_for_agent(agent: AgentRole, config: AppConfig) -> AgentModel:
    """The model for one of this app's own agents (models.agents in YAML)."""
    return build_model(config.models.spec_for_agent(agent), config)
