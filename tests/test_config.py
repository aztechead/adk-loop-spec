"""The YAML schema: what it accepts, and how it refuses bad shapes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from devteam.config import AppConfig, ServiceBackend


def base_raw() -> dict[str, object]:
    return {
        "models": {
            "providers": {
                "gemini-pro": {
                    "provider": "gemini",
                    "backend": "api-key",
                    "model": "gemini-2.5-pro",
                }
            },
            "agents": {"intake": "gemini-pro", "qa": "gemini-pro"},
        }
    }


def test_shipped_config_loads(config: AppConfig) -> None:
    assert config.app.name == "devteam"
    assert config.services.backend is ServiceBackend.IN_MEMORY
    assert config.models.spec_for_agent("intake").model


def test_unknown_agent_provider_is_refused() -> None:
    raw = base_raw()
    raw["models"]["agents"]["qa"] = "no-such-provider"  # type: ignore[index]
    with pytest.raises(ValidationError, match="unknown provider 'no-such-provider'"):
        AppConfig.model_validate(raw)


def test_unknown_loop_spec_provider_is_refused() -> None:
    raw = base_raw() | {"loop_spec": {"phases": {"plan": "no-such-provider"}}}
    with pytest.raises(ValidationError, match="loop_spec.phases.plan"):
        AppConfig.model_validate(raw)


def test_agent_platform_backend_requires_engine_id() -> None:
    raw = base_raw() | {"services": {"backend": "agent-platform"}}
    with pytest.raises(ValidationError, match="agent_engine_id"):
        AppConfig.model_validate(raw)


def test_peer_card_url() -> None:
    raw = base_raw() | {"a2a": {"peers": [{"name": "other", "url": "http://x:1/"}]}}
    parsed = AppConfig.model_validate(raw)
    assert parsed.a2a.peers[0].agent_card_url == "http://x:1/.well-known/agent-card.json"
