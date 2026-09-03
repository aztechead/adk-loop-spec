"""Expose this dev-team instance to its peers over A2A.

Peers reach the whole graph — intake, the loop-spec engineer, and Q&A — so a
feature filed at one instance can be shipped by the instance that owns the
repository. That surface includes an unsandboxed shell, which is why every
request except the public agent card must present the bearer token named by
``a2a.expose.token_env``, and why serving without one is refused off
loopback. ``a2a.expose.tls`` serves HTTPS directly for container-to-container
traffic with no ingress in front.
"""

import os
from pathlib import Path

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .app import build_app, runner_for
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


def build_a2a_app(config: AppConfig, project_dir: Path | None = None) -> Starlette | BearerAuth:
    """The A2A server application over the full dev-team graph.

    The Runner is built here rather than left to ``to_a2a``'s default so the
    exposed graph shares the same session and memory backend as local runs.
    """
    expose = config.a2a.expose
    app = build_app(config, project_dir)
    runner = runner_for(config, app)
    assert app.root_agent is not None
    server = to_a2a(
        app.root_agent, host=expose.host, port=expose.port, protocol=expose.scheme, runner=runner
    )
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
    """Run the A2A endpoint until interrupted, over HTTPS when tls is configured."""
    expose = config.a2a.expose
    uvicorn.run(
        build_a2a_app(config, project_dir),
        host=expose.host,
        port=expose.port,
        ssl_certfile=str(expose.tls.certfile) if expose.tls else None,
        ssl_keyfile=str(expose.tls.keyfile) if expose.tls else None,
    )
