"""Mount loop-spec's ADK extension and route models onto its phases and roles.

loop-spec ships its own ADK bridge (``extensions/adk/loop_spec_adk`` in the
checkout under ``loop_spec.root``); this module is the only place that knows
where that package lives and how our YAML model routing maps onto loop-spec's
environment contract (``docs/loop-spec/configuration.md`` in its tree):

    loop_spec.phases.<phase> -> LOOP_SPEC_PHASE_MODEL_<PHASE>
    loop_spec.roles.<role>   -> LOOP_SPEC_MODEL_<ROLE>

Both take LiteLLM ``provider/model`` ids, which ADK dispatch consumes natively.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from .config import AppConfig
from .models import litellm_id

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.plugins import BasePlugin

_EXTENSION_SUBDIR = Path("extensions/adk")


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


def model_routes(config: AppConfig) -> dict[str, str]:
    """The LOOP_SPEC_* model variables our YAML routing resolves to."""
    providers = config.models.providers
    routes: dict[str, str] = {}
    for phase, provider_key in config.loop_spec.phases.items():
        routes[f"LOOP_SPEC_PHASE_MODEL_{phase.name}"] = litellm_id(providers[provider_key])
    for role, provider_key in config.loop_spec.roles.items():
        routes[f"LOOP_SPEC_MODEL_{role.name}"] = litellm_id(providers[provider_key])
    return routes


def export_model_routes(config: AppConfig) -> dict[str, str]:
    """Publish the model routes into this process's environment.

    Process environment outranks loop-spec's profile file, and every shell the
    bridge spawns inherits it, so this one export reaches all seven phases and
    every dispatched role.
    """
    routes = model_routes(config)
    os.environ.update(routes)
    return routes


def build_working_agent(config: AppConfig, project_dir: Path) -> tuple[LlmAgent, BasePlugin]:
    """The mounted loop-spec working agent and its lifecycle plugin.

    The pair shares one bridge — the plugin records the active skill directory
    that the agent's shell tool reads — so both must be installed on the same
    App together, never separately.
    """
    export_model_routes(config)
    extension = load_extension(config.loop_spec.root)
    bridge = extension.LoopSpecBridge(project_dir)
    agent = extension.build_agent(
        model=litellm_id(config.models.providers[config.loop_spec.agent]),
        bridge=bridge,
    )
    return agent, extension.LoopSpecPlugin(bridge)
