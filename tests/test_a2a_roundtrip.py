"""A real A2A round trip on localhost: expose one agent, consume it as a peer.

The exposed agent is a deterministic function-only Workflow so the whole
protocol — card discovery, task submission, response — runs offline. This is
the same wiring `devteam serve` and the YAML peer list use, minus the LLMs.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from google.adk import Event, Workflow
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

HOST = "127.0.0.1"


def echo(node_input: str) -> Event:
    return Event(message=f"peer echo: {node_input}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


@pytest.fixture()
def peer_url() -> Iterator[str]:
    """Serve the echo workflow over A2A for the duration of one test."""
    port = free_port()
    workflow = Workflow(name="echo_peer", edges=[("START", echo)])
    server = uvicorn.Server(
        uvicorn.Config(
            to_a2a(workflow, host=HOST, port=port), host=HOST, port=port, log_level="error"
        )
    )
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


async def test_remote_peer_answers_over_a2a(peer_url: str) -> None:
    remote = RemoteA2aAgent(
        name="echo_peer",
        agent_card=peer_url + AGENT_CARD_WELL_KNOWN_PATH,
        description="The peer instance under test.",
    )
    runner = Runner(agent=remote, app_name="consumer", session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="consumer", user_id="tester")

    texts: list[str] = []
    async for event in runner.run_async(
        user_id="tester",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="hello team")]),
    ):
        if event.content:
            texts.extend(part.text for part in event.content.parts or [] if part.text)

    assert any("peer echo" in text and "hello team" in text for text in texts), texts
