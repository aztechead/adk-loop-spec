"""Build the session and memory services the runner mounts.

Two backends, one seam: ``in-memory`` for local runs and tests, and
``agent-platform`` (Google Cloud Agent Platform, formerly Vertex AI) for
durable sessions plus Memory Bank, authenticated with ADC. Both sides of the
pair always come from the same backend so a session and its extracted
memories never split across stores.
"""

from typing import NamedTuple, assert_never

from google.adk.memory import BaseMemoryService, InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.sessions import BaseSessionService, InMemorySessionService, VertexAiSessionService

from devteam.config import AppConfig, ServiceBackend
from devteam.models import ADC_HINT, PROJECT_VAR, project_for


class RunnerServices(NamedTuple):
    """The service pair every Runner in this app is built from."""

    session: BaseSessionService
    memory: BaseMemoryService


def build_services(config: AppConfig) -> RunnerServices:
    """The session/memory pair for the configured backend.

    Config validation already guarantees agent_engine_id is present for the
    agent-platform backend; the project may still come from the environment.
    """
    match config.services.backend:
        case ServiceBackend.IN_MEMORY:
            return RunnerServices(InMemorySessionService(), InMemoryMemoryService())
        case ServiceBackend.AGENT_PLATFORM:
            project = project_for(config)
            if not project:
                raise RuntimeError(
                    "services.backend is agent-platform but neither gcp.project nor "
                    f"${PROJECT_VAR} is set (ADC supplies the identity: {ADC_HINT})"
                )
            engine = config.services.agent_platform.agent_engine_id
            return RunnerServices(
                VertexAiSessionService(
                    project=project, location=config.gcp.location, agent_engine_id=engine
                ),
                VertexAiMemoryBankService(
                    project=project, location=config.gcp.location, agent_engine_id=engine
                ),
            )
        case _:
            assert_never(config.services.backend)
