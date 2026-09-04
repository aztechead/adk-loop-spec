"""Mount loop-spec's ADK extension and route models onto its phases and roles.

loop-spec ships its own ADK bridge (``extensions/adk/loop_spec_adk`` in the
checkout under ``loop_spec.root``); this module is the only place that knows
where that package lives and how our YAML maps onto loop-spec's environment
contract (``docs/loop-spec/configuration.md`` in its tree):

    loop_spec.phases.<phase>      -> LOOP_SPEC_PHASE_MODEL_<PHASE>
    loop_spec.roles.<role>        -> LOOP_SPEC_MODEL_<ROLE>
    loop_spec.mount               -> LOOP_SPEC_ADK_AGENT_DIR (the fleet rung's `adk run` target)
    supervisor.oracle.pins.<key>  -> LOOP_SPEC_ANSWER_<KEY>
    gcp                           -> GOOGLE_GENAI_USE_VERTEXAI (+ its successor
                                     GOOGLE_GENAI_USE_ENTERPRISE), GOOGLE_CLOUD_PROJECT/LOCATION

Model routes are the registry ids :func:`devteam.models.model_id` produces.
Any agent-platform route also needs the Vertex variables, because loop-spec
builds its role agents from bare model strings and ADK's Gemini and Claude
classes read the project and region from the environment.
"""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from devteam.config import AppConfig, Backend, ModelSpec
from devteam.models import PROJECT_VAR, model_id, project_for, require_credentials

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.plugins import BasePlugin

_EXTENSION_SUBDIR = Path("extensions/adk")
AGENT_DIR_VAR = "LOOP_SPEC_ADK_AGENT_DIR"
USE_VERTEX_VAR = "GOOGLE_GENAI_USE_VERTEXAI"  # the name ADK 2.8 still reads, now deprecated
USE_ENTERPRISE_VAR = "GOOGLE_GENAI_USE_ENTERPRISE"  # its replacement; both are exported
LOCATION_VAR = "GOOGLE_CLOUD_LOCATION"


def load_extension(root: Path) -> ModuleType:
    """Import ``loop_spec_adk`` from the loop-spec checkout at ``root``.

    The package is not on PyPI by design (loop-spec vendors no manifests), so
    the checkout's extension directory joins ``sys.path`` here — once, in the
    one module that owns the mount.
    """
    package_dir = root.resolve() / _EXTENSION_SUBDIR
    if not (package_dir / "loop_spec_adk" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"loop-spec ADK extension not found under {package_dir}; "
            "run `git submodule update --init` to fetch the checkout"
        )
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    return importlib.import_module("loop_spec_adk")


def agent_dir(config: AppConfig, project_dir: Path) -> Path | None:
    """The mounted CLI agent directory, if scripts/mount-loop-spec.sh has written it."""
    candidate = (project_dir / config.loop_spec.mount / "loop_spec").resolve()
    return candidate if (candidate / "agent.py").is_file() else None


def routed_specs(config: AppConfig) -> list[ModelSpec]:
    """Every spec loop-spec may build an agent from: the working agent, phases, roles."""
    providers = config.models.providers
    keys = [
        config.loop_spec.agent,
        *config.loop_spec.phases.values(),
        *config.loop_spec.roles.values(),
    ]
    return [providers[key] for key in keys]


def environment(config: AppConfig, project_dir: Path) -> dict[str, str]:
    """Every variable our YAML resolves to for loop-spec's benefit."""
    providers = config.models.providers
    env: dict[str, str] = {}
    for phase, provider_key in config.loop_spec.phases.items():
        env[f"LOOP_SPEC_PHASE_MODEL_{phase.name}"] = model_id(providers[provider_key], config)
    for role, provider_key in config.loop_spec.roles.items():
        env[f"LOOP_SPEC_MODEL_{role.name}"] = model_id(providers[provider_key], config)
    for key, answer in config.loop_spec.supervisor.oracle.pins.items():
        env[f"LOOP_SPEC_ANSWER_{key.upper()}"] = answer
    if any(spec.backend is Backend.AGENT_PLATFORM for spec in routed_specs(config)):
        env[USE_VERTEX_VAR] = "true"
        env[USE_ENTERPRISE_VAR] = "true"
        env[LOCATION_VAR] = config.gcp.location
        if project := project_for(config):
            env[PROJECT_VAR] = project
    if (mounted := agent_dir(config, project_dir)) is not None:
        env[AGENT_DIR_VAR] = str(mounted)
    return env


def export_environment(config: AppConfig, project_dir: Path) -> dict[str, str]:
    """Publish the resolved variables into this process's environment.

    Process environment outranks loop-spec's profile file, and every shell the
    bridge spawns inherits it, so this one export reaches all seven phases,
    every dispatched role, and the fleet rung's ``adk run`` launcher.
    """
    env = environment(config, project_dir)
    os.environ.update(env)
    return env


def build_working_agent(config: AppConfig, project_dir: Path) -> tuple[LlmAgent, BasePlugin]:
    """The mounted loop-spec working agent and its lifecycle plugin.

    The pair shares one bridge — the plugin records the active skill directory
    that the agent's shell tool reads — so both must be installed on the same
    App together, never separately.
    """
    for spec in routed_specs(config):
        require_credentials(spec, config)
    export_environment(config, project_dir)
    extension = load_extension(config.loop_spec.root)
    bridge = extension.LoopSpecBridge(project_dir)
    agent = extension.build_agent(
        model=model_id(config.models.providers[config.loop_spec.agent], config),
        bridge=bridge,
    )
    return agent, extension.LoopSpecPlugin(bridge)
