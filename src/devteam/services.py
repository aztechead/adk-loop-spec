"""Build the session and memory services the runner mounts.

Two backends, one seam: ``in-memory`` for local runs and tests, and
``agent-platform`` (Google Cloud Agent Platform, formerly Vertex AI) for
durable sessions plus Memory Bank. Both sides of the pair always come from the
same backend so a session and its extracted memories never split across stores.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from google.adk.memory import BaseMemoryService, InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.sessions import BaseSessionService, InMemorySessionService, VertexAiSessionService

from .config import AppConfig, ServiceBackend


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
            platform = config.services.agent_platform
            project = platform.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError(
                    "services.backend is agent-platform but neither "
                    "services.agent_platform.project nor $GOOGLE_CLOUD_PROJECT is set"
                )
            coordinates = {
                "project": project,
                "location": platform.location,
                "agent_engine_id": platform.agent_engine_id,
            }
            return RunnerServices(
                VertexAiSessionService(**coordinates),
                VertexAiMemoryBankService(**coordinates),
            )
