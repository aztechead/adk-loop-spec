"""The dev-team's own agents: intake classifier, Q&A answerer, and A2A peers.

Each builder takes the validated config and returns a ready agent; model
choice always flows through :mod:`devteam.models` so YAML stays the single
switch for vendors and backends.
"""

from __future__ import annotations

import enum

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools import load_memory, preload_memory
from google.adk.tools.agent_tool import AgentTool

from .config import AppConfig
from .models import model_for_agent


class Category(enum.StrEnum):
    """What the intake classifier may label a request."""

    FEATURE = "FEATURE"
    BUG = "BUG"
    QUESTION = "QUESTION"


def build_intake_agent(config: AppConfig) -> LlmAgent:
    """Classify the incoming request; the graph routes on this label alone."""
    labels = ", ".join(Category)
    return LlmAgent(
        name="intake",
        model=model_for_agent("intake", config),
        description="Classifies a dev-team request as a feature, a bug, or a question.",
        instruction=(
            f"Classify the user's request as exactly one of: {labels}.\n"
            "FEATURE: new behavior or a change to build.\n"
            "BUG: something existing is broken and needs a fix.\n"
            "QUESTION: the user wants information, not a code change.\n"
            "Reply with the single label only, nothing else."
        ),
        output_schema=str,
    )


def build_peer_agents(config: AppConfig) -> list[RemoteA2aAgent]:
    """One remote agent per configured peer — other deployed devteam instances."""
    return [
        RemoteA2aAgent(
            name=peer.name,
            agent_card=peer.agent_card_url,
            description=peer.description or f"The {peer.name} devteam instance, over A2A.",
        )
        for peer in config.a2a.peers
    ]


def build_qa_agent(config: AppConfig) -> LlmAgent:
    """Answer questions from long-term memory, consulting peer teams when needed.

    Memory Bank supplies recall (preload for ambient context, load_memory for
    explicit search); each A2A peer is wrapped as a tool so one question can be
    handed to another deployed instance and its answer folded into ours.
    """
    peers = build_peer_agents(config)
    peer_note = (
        "When the question concerns another team's system, ask that team's "
        f"agent tool ({', '.join(peer.name for peer in peers)}) and credit its answer."
        if peers
        else "No peer teams are configured; answer from memory and this conversation."
    )
    return LlmAgent(
        name="qa",
        model=model_for_agent("qa", config),
        description="Answers dev-team questions from project memory and peer teams.",
        instruction=(
            "Answer the user's question about this project and team.\n"
            "Search long-term memory (load_memory) for relevant past decisions "
            "before answering; say so plainly when memory holds nothing relevant.\n" + peer_note
        ),
        tools=[preload_memory, load_memory, *(AgentTool(agent=peer) for peer in peers)],
    )
