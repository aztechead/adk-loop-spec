"""Service construction: the in-memory pair, and agent-platform's guard rails."""

from __future__ import annotations

import pytest
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService

from devteam.config import AppConfig
from devteam.services import build_services
from tests.test_config import base_raw


def test_in_memory_pair(config: AppConfig) -> None:
    services = build_services(config)
    assert isinstance(services.session, InMemorySessionService)
    assert isinstance(services.memory, InMemoryMemoryService)


def test_agent_platform_without_project_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    raw = base_raw() | {
        "services": {"backend": "agent-platform", "agent_platform": {"agent_engine_id": "42"}}
    }
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        build_services(AppConfig.model_validate(raw))
