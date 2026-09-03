"""The dev-team's own agents: intake classifier, Q&A answerer, and A2A peers.

Each builder takes the validated config and returns a ready agent; model
choice always flows through :mod:`devteam.models` so YAML stays the single
switch for vendors and backends.
"""

import enum
import os

import httpx
from fastapi.openapi.models import HTTPBearer
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.auth.auth_credential import (
    AuthCredential,
    AuthCredentialTypes,
    HttpAuth,
    HttpCredentials,
)
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.tools import load_memory, preload_memory
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field

from .config import AgentRole, AppConfig, PeerConfig
from .models import model_for_agent


class Category(enum.StrEnum):
    """What the intake classifier may label a request."""

    FEATURE = "FEATURE"
    BUG = "BUG"
    QUESTION = "QUESTION"


class IntakeResult(BaseModel):
    """The classifier's structured verdict: a label plus the request it labeled.

    Carrying the request forward matters because a Workflow hands each node
    only the previous node's output — the agents downstream need the user's
    words, not just the label.
    """

    category: Category = Field(description="Exactly one label for the request.")
    request: str = Field(description="The user's request, restated verbatim.")
    team: str | None = Field(
        default=None,
        description="The peer team that owns this request, or null when this team does.",
    )


def build_intake_agent(config: AppConfig) -> LlmAgent:
    """Classify the incoming request and name the team that owns it.

    The graph routes on the label and the team alone; both are deterministic
    code once this agent has spoken.
    """
    labels = ", ".join(Category)
    peers = config.a2a.peers
    team_note = (
        "Peer teams and what they own:\n"
        + "\n".join(f"- {peer.name}: {peer.description or 'no description'}" for peer in peers)
        + "\nSet team to the peer that owns the request, or null when it is ours."
        if peers
        else "There are no peer teams; team is always null."
    )
    return LlmAgent(
        name="intake",
        model=model_for_agent(AgentRole.INTAKE, config),
        description="Classifies a dev-team request as a feature, a bug, or a question.",
        instruction=(
            f"Classify the user's request as exactly one of: {labels}.\n"
            "FEATURE: new behavior or a change to build.\n"
            "BUG: something existing is broken and needs a fix.\n"
            "QUESTION: the user wants information, not a code change.\n"
            "Return the label and the user's request text verbatim.\n" + team_note
        ),
        output_schema=IntakeResult,
    )


def _bearer(token_env: str | None) -> tuple[AuthScheme | None, AuthCredential | None]:
    """The auth pair presented to a peer, or nothing when no token is configured."""
    token = os.environ.get(token_env) if token_env else None
    if not token:
        return None, None
    return (
        HTTPBearer(),
        AuthCredential(
            auth_type=AuthCredentialTypes.HTTP,
            http=HttpAuth(scheme="bearer", credentials=HttpCredentials(token=token)),
        ),
    )


def build_peer_agent(peer: PeerConfig) -> RemoteA2aAgent:
    """One remote agent for a configured peer — another deployed devteam instance.

    ``use_legacy=False`` activates ADK's A2A extension, which fixes message
    duplication and nested-output loss in streaming exchanges.
    """
    scheme, credential = _bearer(peer.token_env)
    return RemoteA2aAgent(
        name=peer.name,
        agent_card=peer.agent_card_url,
        description=peer.description or f"The {peer.name} devteam instance, over A2A.",
        use_legacy=False,
        auth_scheme=scheme,
        auth_credential=credential,
        # A private CA is the norm between containers; httpx verifies against it.
        httpx_client=httpx.AsyncClient(verify=str(peer.ca_bundle)) if peer.ca_bundle else None,
    )


def build_qa_agent(config: AppConfig) -> LlmAgent:
    """Answer questions from long-term memory, consulting peer teams when needed.

    Memory Bank supplies recall (preload for ambient context, load_memory for
    explicit search); each A2A peer is wrapped as a tool so one question can be
    handed to another deployed instance and its answer folded into ours.
    This is also the only agent exposed to peers over A2A: it holds no shell.
    """
    peers = [build_peer_agent(peer) for peer in config.a2a.peers]
    peer_note = (
        "When the question concerns another team's system, ask that team's "
        f"agent tool ({', '.join(peer.name for peer in peers)}) and credit its answer."
        if peers
        else "No peer teams are configured; answer from memory and this conversation."
    )
    return LlmAgent(
        name="qa",
        model=model_for_agent(AgentRole.QA, config),
        description="Answers dev-team questions from project memory and peer teams.",
        instruction=(
            "Answer the user's question about this project and team.\n"
            "Search long-term memory (load_memory) for relevant past decisions "
            "before answering; say so plainly when memory holds nothing relevant.\n" + peer_note
        ),
        tools=[preload_memory, load_memory, *(AgentTool(agent=peer) for peer in peers)],
    )
