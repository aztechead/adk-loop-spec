"""The loop-spec mount: model routing onto its env contract, and the loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from devteam.config import AppConfig
from devteam.loopspec import build_working_agent, load_extension, model_routes
from tests.test_config import base_raw


def test_model_routes_follow_the_loop_spec_contract() -> None:
    raw = base_raw() | {
        "models": {
            "providers": {
                "gemini-pro": {
                    "provider": "gemini",
                    "backend": "api-key",
                    "model": "gemini-2.5-pro",
                },
                "claude": {
                    "provider": "anthropic",
                    "backend": "api-key",
                    "model": "claude-sonnet-4-5",
                },
            },
            "agents": {"intake": "gemini-pro", "qa": "claude"},
        },
        "loop_spec": {
            "phases": {"spec": "claude", "plan": "claude", "execute": "gemini-pro"},
            "roles": {"implementer": "gemini-pro", "code_reviewer": "claude"},
        },
    }
    routes = model_routes(AppConfig.model_validate(raw))
    assert routes == {
        "LOOP_SPEC_PHASE_MODEL_SPEC": "anthropic/claude-sonnet-4-5",
        "LOOP_SPEC_PHASE_MODEL_PLAN": "anthropic/claude-sonnet-4-5",
        "LOOP_SPEC_PHASE_MODEL_EXECUTE": "gemini/gemini-2.5-pro",
        "LOOP_SPEC_MODEL_IMPLEMENTER": "gemini/gemini-2.5-pro",
        "LOOP_SPEC_MODEL_CODE_REVIEWER": "anthropic/claude-sonnet-4-5",
    }


def test_missing_checkout_fails_with_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="git submodule update"):
        load_extension(tmp_path / "nowhere")


def test_working_agent_and_plugin_are_paired(config: AppConfig, tmp_path: Path) -> None:
    agent, plugin = build_working_agent(config, tmp_path)
    assert agent.name == "loop_spec"
    assert plugin.name == "loop_spec"
