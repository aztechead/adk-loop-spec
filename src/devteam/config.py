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
    """Model vendors this app knows how to drive."""

    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Backend(enum.StrEnum):
    """Where a provider's models are served from.

    ``agent-platform`` is the default: Google Cloud Agent Platform (formerly
    Vertex AI), authenticated with Application Default Credentials (ADC) and
    driven through ADK's native Gemini and Claude model classes. ``api-key`` is
    the vendor's own API, driven through LiteLLM with the vendor's key.
    """

    AGENT_PLATFORM = "agent-platform"
    API_KEY = "api-key"


class AgentRole(enum.StrEnum):
    """This app's own LLM agents; each needs a model in models.agents."""

    INTAKE = "intake"
    QA = "qa"
    MANAGER = "manager"


class ServiceBackend(enum.StrEnum):
    """Where sessions and long-term memory live."""

    IN_MEMORY = "in-memory"
    AGENT_PLATFORM = "agent-platform"


class ThinkingLevel(enum.StrEnum):
    """Gemini's thinking depth (``ThinkingConfig.thinking_level``)."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Effort(enum.StrEnum):
    """Claude's adaptive thinking effort (``AnthropicGenerateContentConfig.effort``)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


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


class GcpConfig(_Frozen):
    """The Google Cloud project and region every Agent Platform call targets.

    ADC supplies the identity (``gcloud auth application-default login`` on a
    workstation, the attached service account in production); this block only
    says where. ``project`` falls back to ``$GOOGLE_CLOUD_PROJECT``.
    """

    project: str | None = None
    location: str = "us-central1"


class GenerationConfig(_Frozen):
    """Generation settings for one model, applied through ADK's typed config.

    ``thinking_level`` is Gemini-only and ``effort`` is Claude-only; the model
    spec's validator enforces the pairing so a typo never silently disables
    thinking.
    """

    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_level: ThinkingLevel | None = None
    effort: Effort | None = None
    include_thoughts: bool = False


class ModelSpec(_Frozen):
    """One nameable model: a vendor, a serving backend, and a model id.

    ``location`` overrides ``gcp.location`` for this model alone (Claude on
    Agent Platform is served from fewer regions than Gemini). ``extra`` is
    passed verbatim to LiteLLM and is therefore only valid on the api-key
    backend; the agent-platform backend takes its settings from ``generation``.
    """

    provider: Provider
    backend: Backend = Backend.AGENT_PLATFORM
    model: str
    location: str | None = None
    generation: GenerationConfig = GenerationConfig()
    extra: dict[str, Scalar | dict[str, Scalar]] = {}

    @model_validator(mode="after")
    def _settings_match_the_provider_and_backend(self) -> Self:
        if self.generation.thinking_level is not None and self.provider is not Provider.GEMINI:
            raise ValueError(f"generation.thinking_level is Gemini-only, not for {self.provider}")
        if self.generation.effort is not None and self.provider is not Provider.ANTHROPIC:
            raise ValueError(f"generation.effort is Claude-only, not for {self.provider}")
        if self.extra and self.backend is not Backend.API_KEY:
            raise ValueError(
                "extra is passed to LiteLLM and only applies to backend: api-key; "
                "use generation for the agent-platform backend"
            )
        return self


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
    """The Agent Engine that hosts sessions and Memory Bank (project and region come from gcp)."""

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


class TlsConfig(_Frozen):
    """Serve A2A over HTTPS directly (no ingress terminating TLS in front)."""

    certfile: Path
    keyfile: Path


class ExposeConfig(_Frozen):
    """Where this instance serves its own A2A endpoint, and how callers prove themselves.

    ``token_env`` names the environment variable holding the bearer token every
    request (except the public agent card and the health check) must carry.
    Unset means no auth — only acceptable on a loopback host, because the
    exposed graph includes the engineer agent and its shell.
    """

    host: str = "127.0.0.1"
    port: int = 8001
    token_env: str | None = "DEVTEAM_A2A_TOKEN"
    tls: TlsConfig | None = None

    @property
    def scheme(self) -> str:
        return "https" if self.tls else "http"


class PeerConfig(_Frozen):
    """One other deployed devteam instance reachable over A2A."""

    name: str
    url: str
    description: str = ""
    token_env: str | None = "DEVTEAM_A2A_TOKEN"  # bearer token we present to this peer
    ca_bundle: Path | None = None  # CA file that signed the peer's certificate (private PKI)

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
    """Policy for unattended runs: the oracle, store, and sink ports."""

    oracle: OraclePolicy = OraclePolicy()
    store_dir: Path | None = None  # mirror feature state here (state-store port)
    events_file: Path | None = None  # append every cycle event here (event-sink port)


class ManagerConfig(_Frozen):
    """The manager loop: one loop-spec phase per round, judged between rounds.

    ``phase_prompt`` is appended to every hand-off to the implementer. It asks
    for the phase to be done "extremely well", not "perfectly": the former
    lets the implementer close a phase once it is good enough, the latter sends
    it back into the minutiae. ``stall_rounds`` is how many rounds may pass
    with no new checklist tick before the manager is told to move the cycle on;
    ``max_rounds`` bounds the whole loop before a human is asked.
    """

    phase_prompt: str = (
        "Complete the current loop-spec phase completely and extremely well, then hand off."
    )
    stall_rounds: int = 3
    max_rounds: int = 25


class LoopSpecConfig(_Frozen):
    """The loop-spec mount: where the checkout lives and which models drive it."""

    root: Path = Path("third_party/loop-spec")
    mount: Path = Path("adk_agents")  # where scripts/mount-loop-spec.sh writes the CLI agents
    agent: str = "gemini-pro"
    phases: dict[LoopSpecPhase, str] = {}
    roles: dict[LoopSpecRole, str] = {}
    supervisor: SupervisorConfig = SupervisorConfig()
    manager: ManagerConfig = ManagerConfig()


class AppMeta(_Frozen):
    name: str = "devteam"
    tool_retries: int = 3  # ReflectAndRetryToolPlugin budget per tool
    resumable: bool = True  # App.resumability_config


class AppConfig(_Frozen):
    """The whole validated configuration file."""

    app: AppMeta = AppMeta()
    gcp: GcpConfig = GcpConfig()
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
