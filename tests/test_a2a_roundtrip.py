"""A real A2A round trip on localhost: expose one agent, consume it as a peer.

The exposed agent is a deterministic function-only Workflow so the whole
protocol — card discovery, bearer auth, task submission, response — runs
offline. This is the same wiring `devteam serve` and the YAML peer list use,
minus the LLMs.
"""

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from google.adk import Event, Workflow
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from devteam.a2a import BearerAuth
from devteam.agents import build_peer_agent
from devteam.config import PeerConfig

HOST = "127.0.0.1"
TOKEN = "peer-secret"
TOKEN_ENV = "TEST_PEER_TOKEN"


def echo(node_input: str) -> Event:
    return Event(
        content=types.Content(role="model", parts=[types.Part(text=f"peer echo: {node_input}")])
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


@pytest.fixture()
def peer_url() -> Iterator[str]:
    """Serve the echo workflow over token-protected A2A for one test."""
    port = free_port()
    workflow = Workflow(name="echo_peer", edges=[("START", echo)])
    server_app = BearerAuth(
        to_a2a(workflow, host=HOST, port=port), TOKEN, frozenset({AGENT_CARD_WELL_KNOWN_PATH})
    )
    server = uvicorn.Server(uvicorn.Config(server_app, host=HOST, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://{HOST}:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if httpx.get(url + AGENT_CARD_WELL_KNOWN_PATH, timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.05)
    else:
        pytest.fail("A2A server never published its agent card")
    yield url
    server.should_exit = True
    thread.join(timeout=10)


async def ask(peer: PeerConfig, text: str) -> list[str]:
    runner = Runner(
        agent=build_peer_agent(peer), app_name="consumer", session_service=InMemorySessionService()
    )
    session = await runner.session_service.create_session(app_name="consumer", user_id="tester")
    texts: list[str] = []
    async for event in runner.run_async(
        user_id="tester",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        if event.content:
            texts.extend(part.text for part in event.content.parts or [] if part.text)
    return texts


async def test_remote_peer_answers_with_the_token(
    peer_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    texts = await ask(PeerConfig(name="echo_peer", url=peer_url, token_env=TOKEN_ENV), "hello team")
    assert any("peer echo" in t and "hello team" in t for t in texts), texts


def test_agent_card_is_public_but_the_endpoint_is_not(peer_url: str) -> None:
    assert httpx.get(peer_url + AGENT_CARD_WELL_KNOWN_PATH).status_code == 200
    denied = httpx.post(peer_url + "/", json={"jsonrpc": "2.0", "id": 1, "method": "x"})
    assert denied.status_code == 401
    allowed = httpx.post(
        peer_url + "/",
        json={"jsonrpc": "2.0", "id": 1, "method": "x"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert allowed.status_code != 401
