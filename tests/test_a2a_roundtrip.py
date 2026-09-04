"""Real A2A round trips on localhost: expose one instance, consume it as a peer.

The exposed agent is a deterministic function-only Workflow so the whole
protocol — card discovery, bearer auth, TLS, task submission, response — runs
offline on the same FastAPI application `devteam serve` builds, minus the
LLMs. The last test drives the real dev-team graph with a scripted intake that
assigns the request to the peer, proving the PEER route forwards the user's
words to the other instance.
"""

import datetime as dt
import ipaddress
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.adk import Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from devteam.a2a import HEALTH_PATH, build_api
from devteam.agents import build_peer_agent
from devteam.config import ExposeConfig, PeerConfig, TlsConfig
from devteam.graph import make_router, peer_route
from tests.conftest import ScriptedLlm

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


def self_signed(directory: Path) -> tuple[Path, Path]:
    """A throwaway certificate for 127.0.0.1, the shape a private PKI would issue."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "devteam-test")])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=1))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(HOST))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certfile, keyfile = directory / "cert.pem", directory / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certfile, keyfile


@dataclass(frozen=True)
class Peer:
    url: str
    ca_bundle: Path | None

    def config(self, token_env: str | None = TOKEN_ENV) -> PeerConfig:
        return PeerConfig(
            name="echo_peer", url=self.url, token_env=token_env, ca_bundle=self.ca_bundle
        )

    def verify(self) -> str | bool:
        return str(self.ca_bundle) if self.ca_bundle else True


@pytest.fixture(params=["http", "https"])
def peer(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Peer]:
    """Serve the echo workflow over token-protected A2A, plain or TLS, for one test."""
    port = free_port()
    tls = request.param == "https"
    certfile, keyfile = self_signed(tmp_path) if tls else (None, None)
    workflow = Workflow(name="echo_peer", edges=[("START", echo)])
    expose = ExposeConfig(
        host=HOST,
        port=port,
        tls=TlsConfig(certfile=certfile, keyfile=keyfile) if certfile and keyfile else None,
    )
    runner = Runner(node=workflow, app_name="echo_peer", session_service=InMemorySessionService())
    server_app = build_api(workflow, runner, expose, TOKEN)
    server = uvicorn.Server(
        uvicorn.Config(
            server_app,
            host=HOST,
            port=port,
            log_level="error",
            ssl_certfile=str(certfile) if certfile else None,
            ssl_keyfile=str(keyfile) if keyfile else None,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    handle = Peer(url=f"{request.param}://{HOST}:{port}", ca_bundle=certfile)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            card = httpx.get(
                handle.url + AGENT_CARD_WELL_KNOWN_PATH, timeout=1, verify=handle.verify()
            )
            if card.status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.05)
    else:
        pytest.fail("A2A server never published its agent card")
    yield handle
    server.should_exit = True
    thread.join(timeout=10)


async def collect(runner: Runner, text: str) -> list[str]:
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
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    runner = Runner(
        agent=build_peer_agent(peer.config()),
        app_name="consumer",
        session_service=InMemorySessionService(),
    )
    texts = await collect(runner, "hello team")
    assert any("peer echo" in t and "hello team" in t for t in texts), texts


def test_card_and_health_are_public_but_the_endpoint_is_not(peer: Peer) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "x"}
    assert httpx.get(peer.url + AGENT_CARD_WELL_KNOWN_PATH, verify=peer.verify()).status_code == 200
    health = httpx.get(peer.url + HEALTH_PATH, verify=peer.verify())
    assert health.status_code == 200 and health.json() == {"status": "ok", "app": "echo_peer"}
    assert httpx.post(peer.url + "/", json=body, verify=peer.verify()).status_code == 401
    assert httpx.get(peer.url + "/progress/u/s", verify=peer.verify()).status_code == 401
    allowed = httpx.post(
        peer.url + "/",
        json=body,
        headers={"Authorization": f"Bearer {TOKEN}"},
        verify=peer.verify(),
    )
    assert allowed.status_code != 401
    missing = httpx.get(
        peer.url + "/progress/u/s",
        headers={"Authorization": f"Bearer {TOKEN}"},
        verify=peer.verify(),
    )
    assert missing.status_code == 404


async def test_graph_forwards_a_peers_request_to_that_peer(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PEER route: intake names the owning team, the graph hands the request over A2A."""
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    request = "add a healthcheck endpoint to the platform service"
    intake = LlmAgent(
        name="intake",
        model=ScriptedLlm(
            model="scripted",
            script={
                "Classify": json.dumps(
                    {"category": "FEATURE", "request": request, "team": "echo_peer"}
                )
            },
        ),
        instruction="Classify the request",
    )
    remote = build_peer_agent(peer.config())
    route_request = make_router(frozenset({"echo_peer"}))
    workflow = Workflow(
        name="delegating",
        edges=[
            ("START", intake, route_request),
            (route_request, {peer_route("echo_peer"): remote}),
        ],
    )
    runner = Runner(node=workflow, app_name="consumer", session_service=InMemorySessionService())
    texts = await collect(runner, request)
    assert any("peer echo" in t and request in t for t in texts), texts
