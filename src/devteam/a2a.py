"""Expose this dev-team instance to its peers over A2A, on FastAPI.

Peers reach the whole graph — intake, the manager loop with its loop-spec
engineer, and Q&A — so a feature filed at one instance can be shipped by the
instance that owns the repository. That surface includes an unsandboxed
shell, which is why every request except the public agent card and the
health check must present the bearer token named by ``a2a.expose.token_env``,
and why serving without one is refused off loopback. ``a2a.expose.tls``
serves HTTPS directly for container-to-container traffic with no ingress.

The app is the same FastAPI application ADK's own server family builds on:
the A2A JSON-RPC route and agent-card route from the a2a-sdk, an
``A2aAgentExecutor`` over our Runner, and a progress page for the manager
loop's checklist.
"""

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.agents.base_agent import BaseAgent
from google.adk.runners import Runner
from google.adk.workflow import Workflow

from devteam.app import build_app, runner_for
from devteam.config import AppConfig, ExposeConfig
from devteam.cycle import CycleResult
from devteam.manager import REPORT_KEY, RESULT_KEY, RoundReport, render_progress

HEALTH_PATH = "/healthz"
PUBLIC_PATHS = frozenset({AGENT_CARD_WELL_KNOWN_PATH, HEALTH_PATH})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

type Next = Callable[[Request], Awaitable[Response]]


def build_api(
    root: BaseAgent | Workflow, runner: Runner, expose: ExposeConfig, token: str | None
) -> FastAPI:
    """The FastAPI application serving ``root`` over A2A through ``runner``.

    The agent card is built asynchronously, so the A2A routes attach in the
    lifespan; the health check and the progress page are ordinary routes.
    """
    card_builder = AgentCardBuilder(
        agent=root, rpc_url=f"{expose.scheme}://{expose.host}:{expose.port}/"
    )
    executor = A2aAgentExecutor(runner=runner)
    task_store = InMemoryTaskStore()
    push_store = InMemoryPushNotificationConfigStore()

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        card = await card_builder.build()
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
            push_config_store=push_store,
            agent_card=card,
        )
        add_a2a_routes_to_fastapi(
            api,
            agent_card_routes=create_agent_card_routes(card),
            jsonrpc_routes=create_jsonrpc_routes(handler, "/", enable_v0_3_compat=True),
        )
        yield

    api = FastAPI(title=f"{runner.app_name} A2A", lifespan=lifespan, docs_url=None, redoc_url=None)

    @api.get(HEALTH_PATH)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "app": runner.app_name}

    @api.get("/progress/{user_id}/{session_id}", response_class=HTMLResponse)
    async def progress(user_id: str, session_id: str) -> str:
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="no such session")
        report = session.state.get(REPORT_KEY)
        result = session.state.get(RESULT_KEY)
        return render_progress(
            RoundReport.model_validate(report) if report else None,
            CycleResult.model_validate(result) if result else None,
        )

    if token:

        @api.middleware("http")
        async def require_bearer(request: Request, call_next: Next) -> Response:
            if request.url.path in PUBLIC_PATHS:
                return await call_next(request)
            if request.headers.get("authorization") != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    return api


def build_a2a_app(config: AppConfig, project_dir: Path | None = None) -> FastAPI:
    """The A2A server application over the full dev-team graph.

    The Runner is built here so the exposed graph shares the same session and
    memory backend as local runs.
    """
    expose = config.a2a.expose
    token = os.environ.get(expose.token_env) if expose.token_env else None
    if not token and expose.host not in LOOPBACK_HOSTS:
        raise RuntimeError(
            f"a2a.expose binds {expose.host} without a bearer token; set "
            f"${expose.token_env or 'a2a.expose.token_env'} or bind to loopback"
        )
    app = build_app(config, project_dir)
    runner = runner_for(config, app)
    assert app.root_agent is not None
    return build_api(app.root_agent, runner, expose, token)


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
