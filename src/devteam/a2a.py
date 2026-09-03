"""Expose this dev-team instance to its peers over A2A.

Peers reach the Q&A agent only. The working agent behind the CHANGE route
holds an unsandboxed shell, and loop-spec's harness contract says never to
expose it to untrusted callers — so the A2A surface is the memory-backed
answerer, and every request except the public agent card must present the
bearer token named by ``a2a.expose.token_env``.
"""

import os
from pathlib import Path

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.apps import App
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .agents import build_qa_agent
from .app import MemoryCommitPlugin, runner_for
from .config import AppConfig


class BearerAuth:
    """ASGI middleware: reject any request to the agent endpoint without the shared token."""

    def __init__(self, app: ASGIApp, token: str, public_paths: frozenset[str]) -> None:
        self._app = app
        self._token = token
        self._public = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] not in self._public:
            header = Request(scope).headers.get("authorization", "")
            if header != f"Bearer {self._token}":
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


def peer_facing_app(config: AppConfig) -> tuple[App, LlmAgent]:
    """What peers may talk to: the Q&A agent, no shell, no loop-spec mount."""
    qa_agent = build_qa_agent(config)
    return App(name=config.app.name, root_agent=qa_agent, plugins=[MemoryCommitPlugin()]), qa_agent


def build_a2a_app(config: AppConfig, project_dir: Path | None = None) -> Starlette | BearerAuth:
    """The A2A server application, backed by the configured services.

    The Runner is built here rather than left to ``to_a2a``'s default so the
    exposed agent shares the same session and memory backend as local runs.
    """
    expose = config.a2a.expose
    app, qa_agent = peer_facing_app(config)
    runner = runner_for(config, app)
    server = to_a2a(qa_agent, host=expose.host, port=expose.port, runner=runner)
    token = os.environ.get(expose.token_env) if expose.token_env else None
    if token:
        return BearerAuth(server, token, frozenset({AGENT_CARD_WELL_KNOWN_PATH}))
    if expose.host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            f"a2a.expose binds {expose.host} without a bearer token; set "
            f"${expose.token_env or 'a2a.expose.token_env'} or bind to loopback"
        )
    return server


def serve(config: AppConfig, project_dir: Path | None = None) -> None:
    """Run the A2A endpoint until interrupted."""
    uvicorn.run(
        build_a2a_app(config, project_dir),
        host=config.a2a.expose.host,
        port=config.a2a.expose.port,
    )
