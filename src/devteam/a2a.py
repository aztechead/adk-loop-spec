"""Expose this dev-team instance to its peers over A2A.

``build_a2a_app`` wraps the App's root agent in a Starlette application that
publishes an agent card at /.well-known/agent-card.json and speaks the A2A
protocol. Peers configured in other instances' YAML point their
``RemoteA2aAgent`` entries at this endpoint — that pairing is how several
deployed devteam instances work as one team.
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.applications import Starlette

from .app import build_runner
from .config import AppConfig


def build_a2a_app(config: AppConfig, project_dir: Path | None = None) -> Starlette:
    """The A2A server application, backed by the configured services.

    The Runner is built here rather than left to ``to_a2a``'s default so the
    exposed agent shares the same session and memory backend as local runs.
    """
    runner = build_runner(config, project_dir)
    return to_a2a(
        runner.agent,
        host=config.a2a.expose.host,
        port=config.a2a.expose.port,
        runner=runner,
    )


def serve(config: AppConfig, project_dir: Path | None = None) -> None:
    """Run the A2A endpoint until interrupted."""
    uvicorn.run(
        build_a2a_app(config, project_dir),
        host=config.a2a.expose.host,
        port=config.a2a.expose.port,
    )
