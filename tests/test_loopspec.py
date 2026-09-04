"""The loop-spec mount: YAML onto its environment contract, and the loader."""

from pathlib import Path

import pytest

from devteam.config import AppConfig
from devteam.loopspec import AGENT_DIR_VAR, build_working_agent, environment, load_extension
from tests.conftest import base_raw


def test_environment_follows_the_loop_spec_contract(tmp_path: Path) -> None:
    raw = base_raw() | {
        "models": {
            "providers": {
                "gemini-pro": {"provider": "gemini", "model": "gemini-2.5-pro"},
                "claude": {
                    "provider": "anthropic",
                    "model": "claude-opus-5",
                    "location": "us-east5",
                },
                "claude-api": {
                    "provider": "anthropic",
                    "backend": "api-key",
                    "model": "claude-opus-5",
                },
            },
            "agents": {"intake": "gemini-pro", "qa": "claude", "manager": "gemini-pro"},
        },
        "loop_spec": {
            "phases": {"spec": "claude", "plan": "claude-api", "execute": "gemini-pro"},
            "roles": {"implementer": "gemini-pro", "code_reviewer": "claude"},
            "supervisor": {"oracle": {"pins": {"style": "compact"}}},
        },
    }
    claude_on_vertex = "Claude:projects/offline-project/locations/us-east5/publishers/anthropic/models/claude-opus-5"
    assert environment(AppConfig.model_validate(raw), tmp_path) == {
        "LOOP_SPEC_PHASE_MODEL_SPEC": claude_on_vertex,
        "LOOP_SPEC_PHASE_MODEL_PLAN": "anthropic/claude-opus-5",
        "LOOP_SPEC_PHASE_MODEL_EXECUTE": "gemini-2.5-pro",
        "LOOP_SPEC_MODEL_IMPLEMENTER": "gemini-2.5-pro",
        "LOOP_SPEC_MODEL_CODE_REVIEWER": claude_on_vertex,
        "LOOP_SPEC_ANSWER_STYLE": "compact",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "offline-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
    }


def test_api_key_only_routes_export_no_vertex_variables(tmp_path: Path) -> None:
    raw = base_raw()
    raw["models"]["providers"]["gemini-pro"]["backend"] = "api-key"  # type: ignore[index]
    env = environment(AppConfig.model_validate(raw), tmp_path)
    assert env == {}


def test_cli_mount_is_exported_when_present(config: AppConfig, tmp_path: Path) -> None:
    assert AGENT_DIR_VAR not in environment(config, tmp_path)
    mount = tmp_path / config.loop_spec.mount / "loop_spec"
    mount.mkdir(parents=True)
    (mount / "agent.py").write_text("app = None\n")
    assert environment(config, tmp_path)[AGENT_DIR_VAR] == str(mount)


def test_missing_checkout_fails_with_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="git submodule update"):
        load_extension(tmp_path / "nowhere")


def test_working_agent_and_plugin_are_paired(config: AppConfig, tmp_path: Path) -> None:
    agent, plugin = build_working_agent(config, tmp_path)
    assert agent.name == "loop_spec"
    assert plugin.name == "loop_spec"
