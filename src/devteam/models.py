"""Resolve config model specs into LiteLLM model ids and ADK model objects.

Every model in this app is served through LiteLLM, so a (provider, backend)
pair maps deterministically onto one LiteLLM id prefix:

    gemini    + api-key         -> gemini/<model>       (Google AI Studio key)
    gemini    + agent-platform  -> vertex_ai/<model>
    anthropic + api-key         -> anthropic/<model>    (Anthropic 1p key)
    anthropic + agent-platform  -> vertex_ai/<model>

ADK's own registry resolves any ``provider/model`` string through LiteLLM too,
which is why :func:`litellm_id` strings are also valid loop-spec model routes.
"""

import os
from typing import assert_never

from google.adk.models.lite_llm import LiteLlm

from .config import AgentRole, AppConfig, Backend, ModelSpec, Provider

# Serving Gemini through LiteLLM is this app's deliberate design (one connector
# for every vendor and backend), so ADK's advisory to switch to native Gemini
# is noise here. An operator who exports the variable themselves still wins.
os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")

# The environment each (provider, backend) pair authenticates with. Checked up
# front so a missing key fails at build time with its name, not mid-conversation.
_REQUIRED_ENV: dict[tuple[Provider, Backend], tuple[str, ...]] = {
    (Provider.GEMINI, Backend.API_KEY): ("GOOGLE_API_KEY",),
    (Provider.ANTHROPIC, Backend.API_KEY): ("ANTHROPIC_API_KEY",),
    # Agent Platform authenticates via ADC (gcloud auth application-default
    # login); the project may come from config instead of the environment.
    (Provider.GEMINI, Backend.AGENT_PLATFORM): (),
    (Provider.ANTHROPIC, Backend.AGENT_PLATFORM): (),
}


class MissingCredentialsError(RuntimeError):
    """A model spec needs environment credentials that are not set."""


def litellm_id(spec: ModelSpec) -> str:
    """The LiteLLM model string for a spec, e.g. ``anthropic/claude-opus-5``."""
    match spec.backend:
        case Backend.AGENT_PLATFORM:
            prefix = "vertex_ai"
        case Backend.API_KEY:
            prefix = spec.provider.value
        case _:
            assert_never(spec.backend)
    return f"{prefix}/{spec.model}"


def require_credentials(spec: ModelSpec) -> None:
    """Fail loudly, naming the variable, when a spec's credentials are absent."""
    missing = [
        name for name in _REQUIRED_ENV[(spec.provider, spec.backend)] if not os.environ.get(name)
    ]
    if missing:
        raise MissingCredentialsError(
            f"{litellm_id(spec)} needs {', '.join(missing)} in the environment"
        )


def build_model(spec: ModelSpec, config: AppConfig) -> LiteLlm:
    """An ADK model object for a spec, carrying its extras and platform coordinates."""
    require_credentials(spec)
    kwargs: dict[str, object] = dict(spec.extra)
    if spec.backend is Backend.AGENT_PLATFORM:
        platform = config.services.agent_platform
        kwargs.setdefault(
            "vertex_project", platform.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        kwargs.setdefault("vertex_location", platform.location)
    return LiteLlm(model=litellm_id(spec), **kwargs)


def model_for_agent(agent: AgentRole, config: AppConfig) -> LiteLlm:
    """The model object for one of this app's own agents (models.agents in YAML)."""
    return build_model(config.models.spec_for_agent(agent), config)
