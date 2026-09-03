"""Typed schema for config/devteam.yaml, and the loader that reads it.

The YAML file is the single source of shape for the whole app; secrets stay in
the environment. Validation happens here, at load time, so every other module
receives a fully-checked :class:`AppConfig` and never re-parses anything.
"""

import enum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class Provider(enum.StrEnum):
    """Model vendors this app knows how to drive through LiteLLM."""

    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Backend(enum.StrEnum):
    """Where a provider's models are served from."""

    API_KEY = "api-key"  # the vendor's own API, authenticated by API key
    AGENT_PLATFORM = "agent-platform"  # Google Cloud Agent Platform (formerly Vertex AI)


class AgentRole(enum.StrEnum):
    """This app's own LLM agents; each needs a model in models.agents."""

    INTAKE = "intake"
    QA = "qa"


class ServiceBackend(enum.StrEnum):
    """Where sessions and long-term memory live."""

    IN_MEMORY = "in-memory"
    AGENT_PLATFORM = "agent-platform"


class LoopSpecPhase(enum.StrEnum):
    """The seven loop-spec phases that accept a model override."""

    SPEC = "spec"
    DISCUSS = "discuss"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    ITERATE = "iterate"
    DELIVER = "deliver"


class LoopSpecRole(enum.StrEnum):
    """The loop-spec dispatch roles that accept a model override."""

    SPEC_WRITER = "spec_writer"
    PLANNER = "planner"
    ADVOCATE = "advocate"
    CHALLENGER = "challenger"
    SPEC_COMPLIANCE_REVIEWER = "spec_compliance_reviewer"
    ITERATE_JUDGE = "iterate_judge"
    CODE_REVIEWER = "code_reviewer"
    IMPLEMENTER = "implementer"
    VERIFIER = "verifier"
    PATTERN_MAPPER = "pattern_mapper"


type Scalar = str | int | float | bool


class _Frozen(BaseModel):
    """Immutable, no-extras base for every config node."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(_Frozen):
    """One nameable model: a vendor, a serving backend, and a model id.

    ``extra`` is passed verbatim to LiteLLM (``thinking``, ``temperature``,
    ``api_base``, ...) so generation settings live beside the model they tune.
    """

    provider: Provider
    backend: Backend
    model: str
    extra: dict[str, Scalar | dict[str, Scalar]] = {}


class ModelsConfig(_Frozen):
    """The provider registry plus the provider each in-app agent uses."""

    providers: dict[str, ModelSpec]
    agents: dict[AgentRole, str]

    @model_validator(mode="after")
    def _every_agent_has_a_known_provider(self) -> Self:
        missing = [role for role in AgentRole if role not in self.agents]
        if missing:
            raise ValueError(f"models.agents must name a provider for: {sorted(missing)}")
        for agent, provider_key in self.agents.items():
            if provider_key not in self.providers:
                raise ValueError(
                    f"models.agents.{agent} names unknown provider {provider_key!r}; "
                    f"known providers: {sorted(self.providers)}"
                )
        return self

    def spec_for_agent(self, agent: AgentRole) -> ModelSpec:
        return self.providers[self.agents[agent]]


class AgentPlatformConfig(_Frozen):
    """Google Cloud coordinates for Agent Platform sessions and Memory Bank."""

    project: str | None = None  # None -> $GOOGLE_CLOUD_PROJECT at service build
    location: str = "us-central1"
    agent_engine_id: str | None = None


class ServicesConfig(_Frozen):
    """Which session/memory backend to run, and its coordinates."""

    backend: ServiceBackend = ServiceBackend.IN_MEMORY
    agent_platform: AgentPlatformConfig = AgentPlatformConfig()

    @model_validator(mode="after")
    def _agent_platform_needs_engine(self) -> Self:
        if (
            self.backend is ServiceBackend.AGENT_PLATFORM
            and self.agent_platform.agent_engine_id is None
        ):
            raise ValueError(
                "services.backend is agent-platform but "
                "services.agent_platform.agent_engine_id is not set"
            )
        return self


class ExposeConfig(_Frozen):
    """Where this instance serves its own A2A endpoint, and how callers prove themselves.

    ``token_env`` names the environment variable holding the bearer token every
    request (except the public agent card) must carry. Unset means no auth —
    only acceptable on a loopback host.
    """

    host: str = "127.0.0.1"
    port: int = 8001
    token_env: str | None = "DEVTEAM_A2A_TOKEN"


class PeerConfig(_Frozen):
    """One other deployed devteam instance reachable over A2A."""

    name: str
    url: str
    description: str = ""
    token_env: str | None = "DEVTEAM_A2A_TOKEN"  # bearer token we present to this peer

    @property
    def agent_card_url(self) -> str:
        return f"{self.url.rstrip('/')}/.well-known/agent-card.json"


class A2AConfig(_Frozen):
    expose: ExposeConfig = ExposeConfig()
    peers: tuple[PeerConfig, ...] = ()


class OraclePolicy(_Frozen):
    """How the supervisor answers loop-spec's interview questions.

    Options are matched by substring, first rule wins: ``halt_when`` pauses the
    cycle (loop-spec's ``halt`` answer), ``prefer`` picks a non-default option,
    and anything else takes the ``(Recommended)`` option, which loop-spec
    records as an assumed self-answer.
    """

    halt_when: tuple[str, ...] = ()
    prefer: tuple[str, ...] = ()
    pins: dict[str, str] = {}  # LOOP_SPEC_ANSWER_<KEY> pre-answers, never re-asked


class SupervisorConfig(_Frozen):
    """Policy for unattended runs: the four ports' knobs."""

    oracle: OraclePolicy = OraclePolicy()
    store_dir: Path | None = None  # mirror feature state here (state-store port)
    events_file: Path | None = None  # append every cycle event here (event-sink port)
    max_handoffs: int = 25


class LoopSpecConfig(_Frozen):
    """The loop-spec mount: where the checkout lives and which models drive it."""

    root: Path = Path("third_party/loop-spec")
    mount: Path = Path("adk_agents")  # where scripts/mount-loop-spec.sh writes the CLI agents
    agent: str = "gemini-pro"
    phases: dict[LoopSpecPhase, str] = {}
    roles: dict[LoopSpecRole, str] = {}
    supervisor: SupervisorConfig = SupervisorConfig()


class AppMeta(_Frozen):
    name: str = "devteam"
    tool_retries: int = 3  # ReflectAndRetryToolPlugin budget per tool
    resumable: bool = True  # App.resumability_config


class AppConfig(_Frozen):
    """The whole validated configuration file."""

    app: AppMeta = AppMeta()
    models: ModelsConfig
    services: ServicesConfig = ServicesConfig()
    a2a: A2AConfig = A2AConfig()
    loop_spec: LoopSpecConfig = LoopSpecConfig()

    @model_validator(mode="after")
    def _loop_spec_names_known_providers(self) -> Self:
        known = self.models.providers
        for label, key in (
            ("loop_spec.agent", self.loop_spec.agent),
            *((f"loop_spec.phases.{p}", key) for p, key in self.loop_spec.phases.items()),
            *((f"loop_spec.roles.{r}", key) for r, key in self.loop_spec.roles.items()),
        ):
            if key not in known:
                raise ValueError(
                    f"{label} names unknown provider {key!r}; known providers: {sorted(known)}"
                )
        return self


DEFAULT_CONFIG_PATH = Path("config/devteam.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read and validate the YAML config; raise with the offending path on error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"config file not found: {path}") from None
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return AppConfig.model_validate(raw)
